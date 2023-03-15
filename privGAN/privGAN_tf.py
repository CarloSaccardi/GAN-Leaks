## Implementation of privGAN

import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras import Model, Sequential
from keras.layers import Reshape, Dense, Dropout, Flatten, LeakyReLU, Conv2D, MaxPool2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.optimizers import Adam

from keras.datasets import mnist,cifar10
from keras.optimizers import Adam
from keras import initializers
from scipy import stats
import warnings
import pandas as pd 
from keras.datasets import mnist
from keras import backend as K
import gzip
import pickle 

gan_models_dir =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gan_models'))
sys.path.append(gan_models_dir)
from dcgan.model_torch import Generator, Discriminator


def MNIST_Generator(randomDim = 100, optim = Adam(lr=0.0002, beta_1=0.5)):
    """Creates a generator for MNIST dataset
    Args:
        randomDim (int, optional): input shape. Defaults to 100.
        optim ([Adam], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    
    generator = Sequential()
    generator.add(Dense(512, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02),
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Dense(512,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Dense(1024,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Dense(784, activation='tanh',
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.compile(loss='binary_crossentropy', optimizer=optim)
    
    return generator


def MNIST_Discriminator(optim = Adam(lr=0.0002, beta_1=0.5)):
    """Discriminator for MNIST dataset
    Args:
        optim ([Adam], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    
    discriminator = Sequential()
    discriminator.add(Dense(2048, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02),
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(512,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(256,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(1, activation='sigmoid',
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.compile(loss='binary_crossentropy', optimizer=optim)
    
    return discriminator

def MNIST_DiscriminatorPrivate(OutSize = 2, optim = Adam(lr=0.0002, beta_1=0.5)):
    """The discriminator designed to guess which Generator generated the data
    Args:
        OutSize (int, optional): [description]. Defaults to 2.
        optim ([type], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    
    discriminator = Sequential()
    discriminator.add(Dense(2048, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02),
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(512,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(256,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(OutSize, activation='softmax',
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=optim)
    
    return discriminator



def privGAN(X_train, generators = [MNIST_Generator(),MNIST_Generator()], 
            discriminators = [MNIST_Discriminator(),MNIST_Discriminator()],
            pDisc = MNIST_DiscriminatorPrivate(OutSize = 2), 
            randomDim=100, disc_epochs = 50, epochs=200, dp_delay = 100, 
            batchSize=128, optim = Adam(lr=0.0002, beta_1=0.5), verbose = 1, 
            lSmooth = 0.95, privacy_ratio = 1.0, SplitTF = False):
    
    
    #make sure the number of generators is the same as the number of discriminators 
    if len(generators) != len(discriminators):
        print('Different number of generators and discriminators')
        return()
    else:
        n_reps = len(generators)
        
    #throw error if n_reps = 1 
    if n_reps == 1:
        print('You cannot have only one generator-discriminator pair')
        return()
    
    
    X = []
    t = len(X_train)//n_reps
    y_train = []
    
    for i in range(n_reps):
        if i<n_reps-1:
            X += [X_train[i*t:(i+1)*t]]
            y_train += [i]*t
        else:
            X += [X_train[i*t:]]
            y_train += [i]*len(X_train[i*t:])
    
    y_train = np.array(y_train) + 0.0 
    
    pDisc2 = pDisc
        
    pDisc2.fit(X_train, y_train,
          batch_size=batchSize,
          epochs=disc_epochs,
          verbose=verbose,
          validation_data=(X_train, y_train))
    
    yp= np.argmax(pDisc2.predict(X_train), axis = 1)
    print('dp-Accuracy:',np.sum(y_train == yp)/len(yp))

    
    
    #define combined model 
    outputs = []    
    ganInput = Input(shape=(randomDim,))
    loss = ['binary_crossentropy']*n_reps + ['sparse_categorical_crossentropy']*n_reps
    Pout = []
    loss_weights = [1.0]*n_reps + [1.0*privacy_ratio]*n_reps
    
    pDisc2.trainable = False

    for i in range(n_reps):
        discriminators[i].trainable = False
        outputs += [discriminators[i](generators[i](ganInput))]
        Pout += [pDisc2(generators[i](ganInput))]
        
        
    #specify the combined GAN 
    outputs += Pout
    gan = Model(inputs = ganInput, outputs = outputs)       
    gan.compile(loss = loss, loss_weights = loss_weights, optimizer=optim)

            
    #Get batchcount
    batchCount = int(t // batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    
    dLosses = np.zeros((n_reps,epochs))
    dpLosses = np.zeros(epochs)
    gLosses = np.zeros(epochs)

    for e in range(epochs):
        d_t = np.zeros((n_reps,batchCount))
        dp_t = np.zeros(batchCount)
        g_t = np.zeros(batchCount)
        d_t3acc = np.zeros(batchCount)
        
        for i in range(batchCount):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = []
            generatedImages = []
            yDis2 = []
            yDis2f = []
            pDisc2.trainable = False
            
            
            for j in range(n_reps):
                imageBatch = X[j][np.random.randint(0, len(X[j]), size=batchSize)]
                generatedImages += [generators[j].predict(noise)]

                yDis = np.zeros(2*batchSize)
                yDis[:batchSize] = lSmooth
                discriminators[j].trainable = True
                
                if SplitTF:                    
                    d_r = discriminators[j].train_on_batch(imageBatch, lSmooth*np.ones(batchSize))
                    d_f = discriminators[j].train_on_batch(generatedImages[j],np.zeros(batchSize))
                    d_t[j,i] = d_r + d_f
                else:
                    X_temp = np.concatenate([imageBatch, generatedImages[j]])
                    d_t[j,i] = discriminators[j].train_on_batch(X_temp, yDis)
                    
                discriminators[j].trainable = False
                l = list(range(n_reps))
                del(l[j])
                yDis2 += [j]*batchSize
                yDis2f += [np.random.choice(l,(batchSize,))]
            
            yDis2 = np.array(yDis2)
            
            #Train privacy discriminator
            generatedImages = np.concatenate(generatedImages)

            
            if e >= dp_delay: 
                pDisc2.trainable = True
                dp_t[i] = pDisc2.train_on_batch(generatedImages, yDis2)
                pDisc2.trainable = False
                
            
            yGen = [np.ones(batchSize)]*n_reps + yDis2f
            
            #Train combined model
            g_t[i] = gan.train_on_batch(noise, yGen)[0]
            
            if verbose == 1:
                print(
                    'epoch = %d/%d, batch = %d/%d' % (e, epochs, i, batchCount),
                    100*' ',
                    end='\r'
                )

                      

        # Store loss of most recent batch from this epoch
        dLosses[:,e] = np.mean(d_t, axis = 1)
        dpLosses[e] = np.mean(dp_t)
        gLosses[e] = np.mean(g_t)
        
        if e%verbose == 0:
            print('epoch =',e)
            print('dLosses =', np.mean(d_t, axis = 1))
            print('dpLosses =', np.mean(dp_t))
            print('gLosses =', np.mean(g_t))
            yp= np.argmax(pDisc2.predict(generatedImages), axis = 1)
            print('dp-Accuracy:',np.sum(yDis2 == yp)/len(yp))
            
    return (generators, discriminators, pDisc2, dLosses, dpLosses, gLosses)





if __name__ == '__main__':

    # Load MNIST data and concatenate the train+test set
    f = gzip.open('privGAN\mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()
    (X_train, y_train), (X_test, y_test) = data


    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_test = (X_test.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    X_all = np.concatenate((X_train,X_test))


    #Generate training test split
    frac = 0.1 
    n = int(frac*len(X_all))
    l = np.array(range(len(X_all)))
    l = np.random.choice(l,len(l),replace = False)
    X = X_all[l[:n]]
    X_comp = X_all[l[n:]]

    print('training set size:',X.shape)
    print('test set size:',X_comp.shape)

    #########################################################################
    
    K.clear_session()
    optim = Adam(lr=0.0002, beta_1=0.5)
    generators = [MNIST_Generator(optim = Adam(lr=0.0002, beta_1=0.5)),
                MNIST_Generator(optim = Adam(lr=0.0002, beta_1=0.5))]
    discriminators = [MNIST_Discriminator(optim = Adam(lr=0.0002, beta_1=0.5))
                    ,MNIST_Discriminator(optim = Adam(lr=0.0002, beta_1=0.5))]
    pDisc = MNIST_DiscriminatorPrivate(OutSize = 2, 
                                        optim = Adam(lr=0.0002, beta_1=0.5))

    (generators, discriminators, _, dLosses, dpLosses, gLosses)= privGAN(X, epochs = 1, 
                                                                            disc_epochs=1,
                                                                            batchSize=256,
                                                                            generators = generators, 
                                                                            discriminators = discriminators,
                                                                            pDisc = pDisc,
                                                                            optim = optim,
                                                                            privacy_ratio = 1.0)

