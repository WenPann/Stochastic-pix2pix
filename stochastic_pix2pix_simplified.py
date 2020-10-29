# Author: Wen Pan, Michael Pyrcz, Carlos Torres-Verdin
# Reference: Stochastic Pix2Pix: A New Machine Learning Method for Geophysical and Well Conditioning of Rule-Based Channel Reservoir Models
from __future__ import print_function, division
import scipy


from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply,RepeatVector,Lambda, Add
from keras.layers import BatchNormalization, Activation,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
from keras import backend as K
from scipy import ndimage
from scipy.ndimage import maximum_filter,zoom,median_filter

# when the conditioning reproduction weight is set to be 0, it becomes to be GAN 
weight_JS=1 #weight for Jensen Shannon divergence
weight_well=2 # weight for well data
weight_seismic=1 # weight for seismic
weight_dir='./weight/'
img_dir='./img/'
class Stochastic_pix2pix():
    def __init__(self):
        # Input param
        self.batch_size=8
        #training,output image shape
        self.img_rows = 56
        self.img_cols = 56
        # # of latent variables
        self.latent=20
        # # of facies
        self.channels = 5
        #input and output image shape
        self.img_shape1 = (self.img_rows, self.img_cols, 1)#B seismic outline
        self.img_shape2 = (self.img_rows, self.img_cols, self.channels)#A training,output image shape
        # well location
        self.well_loc=np.array([[28,28],[1,4],[32,29],[33,33],[12,13],[4,46],[44,40],[20,40],[51,15],[28,47]])
        # regions corresponding to largest template size
        # Note we assume nonstationarity (regional stationarity) if they are larger than 1
        self.disc_patch = (1)
        # number of filters in the first layer of G and D, D's filter size can be infered from the MPS density function.   
        self.gf = 20
        self.df = 5
        # weight optimization param
        optimizer = Adam(0.0008, 0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images (params)
        img_B = Input(shape=self.img_shape1) #seismic outline
        well=Input(shape=self.img_shape1) # well data map
        myrandom=Input(shape=(self.latent,)) # random latent variables

        # Generate conditional realization fake_A, reproduced well data, and seismic outline with generator
        fake_A,fake_well,fake_seismic= self.generator([img_B,myrandom,well])

        # to train the generator to fool the discriminator we will only train the generator
        self.discriminator.trainable = False

        # Discriminator calculates the multi-point statistics and determines if the statistics calculated from training images are reproduced in the realization.
        valid = self.discriminator(fake_A)

        # specify the relative weight for each loss functions
        self.combined = Model(inputs=[img_B,myrandom,well], outputs=[valid,fake_well,fake_seismic])
        self.combined.compile(loss=['binary_crossentropy','categorical_crossentropy','binary_crossentropy'],
                              loss_weights=[weight_JS,weight_well,weight_seismic], 
                              optimizer=optimizer)


    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u


        # Image input (seismic)
        d0 = Input(shape=self.img_shape1)
        # latent variables
        myrandom=Input(shape=(self.latent,))
        myrandom2=Dense(self.img_rows*self.img_cols)(myrandom)        
        myrandom2=Reshape((self.img_rows,self.img_cols,1))(myrandom2)
        # combine latent variables with seismic input
        combine2=Concatenate()([d0, myrandom2])
        #input well data
        mywell=Input(shape=self.img_shape1)
        #combine all 3 inputs
        # combine2=Concatenate(axis=-1)([mywell,combine2])
        combine2=Concatenate()([mywell,combine2])
        # Encoder-Decoder Structure
        d1 = conv2d(combine2, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        # Start of Decoding
        u8 = deconv2d(d3, d2, self.gf*2)
        u9 = deconv2d(u8, d1, self.gf)
        u10 = UpSampling2D(size=2)(u9)
        u11=Concatenate()([u10,mywell])
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='softmax')(u11)

        # Back calculation of seismic data
        seismic0=Lambda( lambda x: x[:,:,:,1:])(output_img)
        seismic=Lambda( lambda x: K.sum(x, axis=-1),name='seismic')(seismic0)
        
        # Back calculation of well data
        wells=[]
        wells.append(Lambda(lambda x: x[:,13:14,40,:])(output_img))
        wells.append(Lambda(lambda x: x[:,19:20,4,:])(output_img))
        wells.append(Lambda(lambda x: x[:,32:33,29,:])(output_img))
        wells.append(Lambda(lambda x: x[:,33:34,33,:])(output_img))
        wells.append(Lambda(lambda x: x[:,12:13,13,:])(output_img))
        wells.append(Lambda(lambda x: x[:,4:5,46,:])(output_img))
        wells.append(Lambda(lambda x: x[:,44:45,40,:])(output_img))
        wells.append(Lambda(lambda x: x[:,20:21,40,:])(output_img))
        wells.append(Lambda(lambda x: x[:,51:52,15,:])(output_img))
        wells.append(Lambda(lambda x: x[:,28:29,47,:])(output_img))            
        well=Concatenate(axis=1)(wells)

        return Model(inputs=[d0,myrandom,mywell], outputs=[output_img,well,seismic])







    def build_discriminator(self):
        # define convolutional down sampling layer
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d


        #input training image/output from generator
        img_A = Input(shape=self.img_shape2)
        
        # MPS extraction
        d1 = d_layer(img_A, self.df, bn=False)#128
        d2 = d_layer(d1, self.df*2)#64
        d3 = d_layer(d2, self.df*4)#32
        # Force output between 0 and 1
#        validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d3)
        d3=Flatten()(d3)
        validity=Dense(1, activation='sigmoid')(d3)
        return Model(img_A, validity)

    def train(self, epochs, batch_size=8, sample_interval=50):

        # Adversarial loss ground truths
        batch_size=self.batch_size
#        valid = np.ones((batch_size,) + self.disc_patch)
#        fake = np.zeros((batch_size,) + self.disc_patch) 
        valid = np.ones(batch_size)
        fake = np.zeros(batch_size)         
        # load training images      
        imgs=np.load('train_dat.npy')   
        imgs_B_all=imgs.copy()
        imgs_B_all[imgs<=0]=0
        imgs_B_all[imgs>0]=1
        imgs_A_all=imgs[:,:,:,0].copy()
        
       
        # keep record of training process
#        self.process=[[],[],[],[]]# discriminator accuracy for training image; realization; generator accuracy; discriminator loss; generator loss.
#        # imgs_A_all: training images
#        imgs_A_all=a.copy()
#        # seismic_all: seismic conditioing data for loss calculation
#        seismic_all=(a>0).astype(int)
#        # imgs_B_all: seismic conditioing data for input
#        imgs_B_all=seismic_all.copy()
#        imgs_B_all=np.expand_dims(imgs_B_all,axis=-1)
        # well_all: well data for input
        well_all=np.zeros(imgs_B_all.shape)
        for mm in range(self.well_loc.shape[0]):
            well_all[:,self.well_loc[mm,0],self.well_loc[mm,1],0]=(imgs_A_all[:,self.well_loc[mm,0],self.well_loc[mm,1]])/2#input
        # well_dat_all" well data fro loss calculation
        well_dat_all=np.zeros((imgs_B_all.shape[0],self.well_loc.shape[0]))
        for mm in range(self.well_loc.shape[0]):
            well_dat_all[:,mm]=imgs_A_all[:,self.well_loc[mm,0],self.well_loc[mm,1]]#output
        well_dat_all=to_categorical(well_dat_all,self.channels)
        # one-hot encoding for training images
        imgs_A_all=to_categorical(imgs_A_all,self.channels)  
        
        # start of training
        for jj in range(1000):
            for epoch in range(5): 
                # generate random latent input variables
                res_all=np.random.normal(0,1,size=(imgs_B_all.shape[0],self.latent)) 
                ids=np.arange(imgs_B_all.shape[0])
                np.random.shuffle(ids)
                for ii in np.arange(imgs_B_all.shape[0]//batch_size):
                    # divide training data according to their batch size
                    imgs_A=imgs_A_all[ids[int(ii*batch_size):int((ii+1)*batch_size)]]
                    imgs_B=imgs_B_all[ids[int(ii*batch_size):int((ii+1)*batch_size)]]
                    well_dat=well_dat_all[ids[int(ii*batch_size):int((ii+1)*batch_size)]]
                    well=well_all[ids[int(ii*batch_size):int((ii+1)*batch_size)]]
                    res=res_all[ids[int(ii*batch_size):int((ii+1)*batch_size)]]

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    # generate fake realization and its associated conditioning data
                    fake_A,fake_well,fake_seismic= self.generator.predict([imgs_B,res,well])
                    # train discrminator to calculate J-S divergence
                    d_loss_real = self.discriminator.train_on_batch(imgs_A, valid)
                    d_loss_fake = self.discriminator.train_on_batch(fake_A, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # -----------------
                    #  Train Generator
                    # -----------------
    
                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_B,res,well], [valid,well_dat,imgs_B[:,:,:,0]])
                    
                    # record of trainign process
#                    # accuracy
#                    self.process[0].append(d_loss_real[1])
#                    self.process[1].append(d_loss_fake[1])
#                    # loss
#                    self.process[2].append(d_loss[0]) 
#                    self.process[3].append(g_loss[0])   

                    # output the training progress
                    if ii%20==0:
                        # calculate well data mismatch
                        myfakewell=np.argmax(fake_well,axis=-1)
                        well_dat2=np.argmax(well_dat,axis=-1)
                        error=np.sum((myfakewell!=well_dat2))/myfakewell.shape[0]

                        print ("[Epoch %d,%d] [D loss: %f, D acc: %3d%%] [G loss: %f][well error:%3d%%] " % (jj, epoch,
                                                                                d_loss[0], 100*d_loss[1],
                                                                                g_loss[0],error*10))  
    
                    # save generated image samples
                    if ii==0:
                        self.save_imgs(jj,imgB=imgs_B[0:1,:,:,:],imgA=imgs_A[0],well=well[0:1])
                    # save weight and training process
                    if ii==0 and epoch==0:
                        self.generator.save_weights(weight_dir+'pos_predictor_full_seismic_filt2_sand_whole2_'+str(jj)+'.h5')
                        self.combined.save_weights(weight_dir+'pos_combined_full_seismic_filt2_sand_whole2_'+str(jj)+'.h5')
                        self.discriminator.save_weights(weight_dir+'pos_discriminator_full_seismic_filt2_sand_whole2_'+str(jj)+'.h5')
#                        self.save_process()

#    def save_process(self):
#        plt.figure()
#        plt.plot(self.process[0])
#        plt.plot(self.process[1])
#        plt.ylabel('Accuracy')
#        plt.xlabel('Iteration')
#        plt.legend(['D_real','D_fake'])
#        plt.savefig(mydir+'acc_plot.png',dpi=100)
#        plt.close('all')
#
#        plt.figure()
#        plt.plot(self.process[2])
#        plt.plot(self.process[3])
#        plt.ylabel('Loss')
#        plt.xlabel('Iteration')
#        plt.legend(['D','G'])
#        plt.savefig(mydir+'loss_plot.png',dpi=100)
#        plt.close('all')        




    def save_imgs(self, ii,imgB,imgA,well):
        # training image
        plt.figure()
        zz=np.argmax(imgA[:,:,:],axis=-1)
        plt.imshow(np.argmax(imgA[:,:,:],axis=-1))
        plt.scatter(self.well_loc[:,0],self.well_loc[:,1],color='red')
        plt.title('Truth Model')
        plt.savefig(img_dir+str(ii)+'_true.png',dpi=100)
        plt.close('all')
        # realization 1
        plt.figure()
        res=np.random.normal(0,1,size=[1,self.latent])
        fake_A,wellfake,seismic = self.generator.predict([imgB,res,well])        
        plt.imshow(np.argmax(fake_A[0,:,:,:],axis=-1))
        plt.title('Realization1')
        plt.savefig(img_dir+str(ii)+'_realization1.png',dpi=100)        
        plt.close('all')
        # realization 2
        plt.figure()
        res=np.random.normal(0,1,size=[1,self.latent])
        fake_A,wellfake,seismic = self.generator.predict([imgB,res,well]) 
        z=np.argmax(fake_A[0,:,:,:],axis=-1)
        plt.imshow(z)
        plt.title('Realization2')
        plt.savefig(img_dir+str(ii)+'_realization2.png',dpi=100)        
        plt.close('all')
        # wel data mismatch map
        plt.figure()
        res=np.random.normal(0,1,size=[1,self.latent])
        fake_A,wellfake,seismic = self.generator.predict([imgB,res,well]) 
        z=np.argmax(fake_A[0,:,:,:],axis=-1)
        z2=np.ones(z.shape)*np.nan
        for i in range(self.well_loc.shape[0]):
            z2[self.well_loc[i,1],self.well_loc[i,0]]=z[self.well_loc[i,0],self.well_loc[i,1]]==zz[self.well_loc[i,0],self.well_loc[i,1]]
        plt.imshow(z2)
        plt.title('Well Data Mismatch')
        plt.savefig(img_dir+str(ii)+'_well_dat.png',dpi=100)        
        plt.close('all')
        return 0

if __name__ == '__main__':
    gan = Stochastic_pix2pix()
    # gan.generator.load_weights(weight_dir+'pos_predictor_full_seismic_filt2_sand_whole2.h5')
    # gan.discriminator.load_weights(weight_dir+'pos_discriminator_full_seismic_filt2_sand_whole2.h5')
    # gan.combined.load_weights(weight_dir+'pos_combined_full_seismic_filt2_sand_whole2.h5')
    gan.train(epochs=1)
    
