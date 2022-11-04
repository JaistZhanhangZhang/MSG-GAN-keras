import os
from model import generator, discriminator
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop,Adam
import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

from dataloader import dataloader
config= ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)

##### hyperparamaters ##########
dataset_rootdir="../images1024x1024/"
epochs=500000
batchsize=4
latent=512
gp_weight=10.
d_optimizer=Adam(learning_rate=0.003)
g_optimizer=Adam(learning_rate=0.003)
################################
adversial_epoch=0


def hinge_G_loss(y_pred):
    return -tf.reduce_mean(y_pred)

def hinge_D_real_loss(y_pred):
    return tf.reduce_mean(tf.nn.relu(1-y_pred))

def hinge_D_fake_loss(y_pred):
    return tf.reduce_mean(tf.nn.relu(1+y_pred))

def plotGeneratedImages(epoch, examples=25, dim=(5 , 5), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent])
    generatedImages = generator.predict(noise)
    generatedImages=generatedImages[-1].reshape((-1,1024,1024,3))
    generatedImages = (generatedImages + 1.0)*0.5
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest',cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/MSG-SN_image_epoch_%d.png' % epoch)


def load_weights(generator_dir,discriminator_dir):
    generator.load_weights(generator_dir)
    discriminator.load_weights(discriminator_dir)

def save_weights(generator_dir,discriminator_dir):
    generator.save_weights(generator_dir)
    discriminator.save_weights(discriminator_dir)

def gradient_penalty(batch_size, real_images, fake_images):
    """Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    
    interpolated=[]
    
    diff = fake_images[0] - real_images[0]
    interpolated.append( real_images[0] + alpha * diff )

    diff = fake_images[1] - real_images[1]
    interpolated.append( real_images[1] + alpha * diff )

    diff = fake_images[2] - real_images[2]
    interpolated.append( real_images[2] + alpha * diff )

    diff = fake_images[3] - real_images[3]
    interpolated.append( real_images[3] + alpha * diff )

    diff = fake_images[4] - real_images[4]
    interpolated.append( real_images[4] + alpha * diff )

    diff = fake_images[5] - real_images[5]
    interpolated.append( real_images[5] + alpha * diff )

    diff = fake_images[6] - real_images[6]
    interpolated.append( real_images[6] + alpha * diff )

    diff = fake_images[7] - real_images[7]
    interpolated.append( real_images[7] + alpha * diff )

    diff = fake_images[8] - real_images[8]
    interpolated.append( real_images[8] + alpha * diff )
    

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated[4]])[0]########  ?????????? How to calculate the gradient when multi-images-input?
    #  3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def train_each_epoch(Y_train,batchsize=batchsize):

    ##### train D ###########
    rdm=np.random.normal(0.0,1.0,size=(batchsize,latent))
    generator.trainable=False
    discriminator.trainable=True
    with tf.GradientTape() as tape:
        fake_images = generator(rdm)
        fake_images_predict=discriminator(fake_images)
        real_images_predict=discriminator(Y_train)

        d_cost = hinge_D_real_loss(real_images_predict) + hinge_D_fake_loss(fake_images_predict)

        #### gradient penalty ########
        #gp = gradient_penalty(batchsize, Y_train_per_batch, fake_images)

        #d_loss=d_cost+ gp * gp_weight
        d_loss=d_cost
    
    d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
    d_optimizer.apply_gradients(
            zip(d_gradient, discriminator.trainable_variables)
        )

    ##### train G ############
    generator.trainable=True
    discriminator.trainable=False
    rdm=np.random.normal(0.0,1.0,size=(batchsize,latent))
    with tf.GradientTape() as tape:
        fake_images = generator(rdm)
        fake_images_predict=discriminator(fake_images)
        g_loss = hinge_G_loss(fake_images_predict)

    gen_gradient=tape.gradient(g_loss,generator.trainable_variables)
    g_optimizer.apply_gradients(
            zip(gen_gradient, generator.trainable_variables)
        )
    #print("###############################################################################")
    return{"d_loss":d_loss,"g_loss":g_loss}

def train(epochs=epochs,checkpoints=None):
    if checkpoints is not None:
        load_weights("models/generator_"+str(checkpoints)+".h5", "models/discriminator_"+str(checkpoints)+".h5")

    filelist=os.listdir(dataset_rootdir)
    Y_train=dataloader(batch_size=batchsize)

    for epoch in range(0,epochs):
        #print(epoch)
        #batch_data=next(Y_train)
        #Y_train=np.asarray(Y_train)
        #Y_train=Y_train.transpose([1,0,2,3,4])
        #print(Y_train.shape)
        result=train_each_epoch(next(Y_train),batchsize=batchsize)
        print("epoch: " + str(epoch) + "     d_loss: " + str(result["d_loss"])+"        g_loss: "+ str(result["g_loss"]))

        if epoch %1000 == 0:
            plotGeneratedImages(epoch)
            save_weights("models/generator_"+str(epoch)+".h5", "models/discriminator_"+str(epoch)+".h5")


train()

        

