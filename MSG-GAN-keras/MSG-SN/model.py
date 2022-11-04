import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import Model, Sequential
#from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
Adam=adam_v2.Adam
from SNbase import ConvSN2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config= ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def hinge_G_loss(y_true, y_pred):
    return -K.mean(y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))

#from keras.optimizers import Adam
class PixelNorm(Layer):   
  def __init__(self, channel_axis=-1, **kwargs):
    self.channel_axis = channel_axis
    super().__init__() 

  def call(self, x, epsilon=1e-8):
    return x / K.sqrt(K.mean(K.square(x), axis=len(x.shape)-1, keepdims=True) + epsilon)

  def compute_output_shape(self, input_shape):
    return input_shape
  
  def get_config(self):
    return {
        'channel_axis': self.channel_axis,
        **super().get_config()
    }
class MinibatchStddev(Layer):
    def __init__(self,group_size=4,num_new_features=1,**kwargs):
        super().__init__(**kwargs)
        self.group_size=group_size
        self.num_new_features=num_new_features
    
    def call(self,x):
        x=tf.transpose(x,perm=[0,3,1,2])# [NHWC]->[NCHW]
        group_size=tf.minimum(self.group_size,tf.shape(x)[0])
        s=x.shape
        y=tf.reshape(x,
                    [group_size,-1,self.num_new_features,s[1]//self.num_new_features,s[2],s[3]])
        y=tf.cast(y,tf.float32)
        y-=tf.reduce_mean(y,axis=0,keepdims=True)
        y=tf.reduce_mean(tf.square(y),axis=0)
        y=tf.sqrt(y+1e-8)
        y=tf.reduce_mean(y,axis=[2,3,4],keepdims=True)
        y=tf.reduce_mean(y,axis=[2])
        y=tf.cast(y,x.dtype)
        y=tf.tile(y,[self.group_size,1,s[2],s[3]])
        x=tf.concat([x,y],axis=1)
        x=tf.transpose(x,perm=[0,2,3,1])
        return x
    def compute_output_shape(self, input_shape):
        output_shape=input_shape
        output_shape[-1]=input_shape[-1]+1
        return output_shape
    def get_config(self):
        return {
            'group_size': self.group_size,
            "num_new_features": self.num_new_features,
            **super().get_config()
        }

randomDim = 512

#### Creare Generator ###

adam = Adam(lr=0.003)

inp_noise=Input(shape=(randomDim,))

# Block 1
model=Reshape((1,1,512))(inp_noise)
model=Conv2DTranspose(512,(4,4))(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r1
oup_4=Conv2D(3,(1,1),activation="linear",padding='same')(model)

#Block 2
model=UpSampling2D()(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r2
oup_8=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 3
model=UpSampling2D()(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r3
oup_16=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 4
model=UpSampling2D()(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(512,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r4
oup_32=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 5
model=UpSampling2D()(model)
model=Conv2D(256,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(256,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r5
oup_64=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 6
model=UpSampling2D()(model)
model=Conv2D(128,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(128,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r6
oup_128=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 7
model=UpSampling2D()(model)
model=Conv2D(64,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(64,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r7
oup_256=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 8
model=UpSampling2D()(model)
model=Conv2D(32,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(32,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r7
oup_512=Conv2D(3,(1,1),activation="linear",padding='same')(model)

# Block 9
model=UpSampling2D()(model)
model=Conv2D(16,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=Conv2D(16,(3,3),padding='same')(model)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
# r7
oup_1024=Conv2D(3,(1,1),activation="linear",padding='same')(model)



generator=Model(inputs=inp_noise,outputs=[oup_4,oup_8,oup_16,oup_32,oup_64,oup_128,oup_256,oup_512,oup_1024])
                     
generator.summary()

## Create Discriminator ###


inp_1024=Input(shape=(1024,1024,3))

model=ConvSN2D(16,(3,3),padding='same')(inp_1024)
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)

# D Block 1
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(16,(3,3),padding='same')(model)#                   Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(32,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

# D Block 2
inp_512=Input(shape=(512,512,3))

model=Concatenate()([inp_512,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(32,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(64,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

# D Block 3
inp_256=Input(shape=(256,256,3))

model=Concatenate()([inp_256,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(64,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(128,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

# D Block 4
inp_128=Input(shape=(128,128,3))

model=Concatenate()([inp_128,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(128,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(256,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

# D Block 5
inp_64=Input(shape=(64,64,3))

model=Concatenate()([inp_64,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(256,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

# D Block 6
inp_32=Input(shape=(32,32,3))

model=Concatenate()([inp_32,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

# D Block 7
inp_16=Input(shape=(16,16,3))

model=Concatenate()([inp_16,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

#D Block 8
inp_8=Input(shape=(8,8,3))

model=Concatenate()([inp_8,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=AveragePooling2D()(model)#                                AveragePooling

#D Block 9
inp_4=Input(shape=(4,4,3))

model=Concatenate()([inp_4,model])#                            Combine Function
model=MinibatchStddev()(model) #                                MiniBatchStd 
model=ConvSN2D(512,(3,3),padding='same')(model)#                  Conv
model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)
model=ConvSN2D(512,(4,4),padding='same')(model)#                  Conv
#model=PixelNorm()(model)
model=LeakyReLU(0.2)(model)

model=Flatten()(model)
oup=Dense(1,activation="linear")(model)

discriminator=Model(inputs=[inp_4,inp_8,inp_16,inp_32,inp_64,inp_128,inp_256,inp_512,inp_1024],outputs=oup)
discriminator.summary()

