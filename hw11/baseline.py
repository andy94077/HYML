import os, sys, argparse
import importlib
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, Flatten, Activation, GlobalAveragePooling2D, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
import cv2

import utils

def build_model(img_shape, in_dim):
    def build_generator(in_dim):
        generator = Sequential([
            Dense(512*4*4, activation='relu', input_shape=(in_dim,), use_bias=False, kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [4, 4, 512]
            Reshape((4, 4, 512)),

            Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [8, 8, 512]
            
            Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [16, 16, 256]
            
            Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [32, 32, 256]
            
            Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh', kernel_initializer=RandomNormal(0, 0.02)), # [64, 64, 3]
        ], name='generator')
        return generator
    
    def conv_block(filters, kernel_size, strides=1, padding='same'):
        return Sequential([
            Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),
            LeakyReLU(0.2)
        ])

    def build_discriminator(img_shape):
        discriminator = Sequential([
            Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape, kernel_initializer=RandomNormal(0, 0.02)),
            LeakyReLU(0.2),

            conv_block(128, 3, strides=2),
            conv_block(256, 3, strides=2),
            conv_block(512, 3, strides=2),

            Conv2D(1, 4, activation='sigmoid', kernel_initializer=RandomNormal(0, 0.02)),
            Flatten()
        ], name='discriminator')
        return discriminator
    
    generator = build_generator(in_dim)
    discriminator = build_discriminator(img_shape)
    gan = Sequential([
        generator,
        discriminator
    ], 'gan')

    discriminator.compile(Adam(2e-4, beta_1=0.5), loss='binary_crossentropy')
    discriminator.trainable = False
    gan.compile(Adam(2e-4, beta_1=0.5), loss='binary_crossentropy')

    return gan, generator, discriminator

def build_lsgan(img_shape, in_dim):
    def build_generator(in_dim):
        generator = Sequential([
            Dense(512*4*4, activation='relu', input_shape=(in_dim,), use_bias=False, kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [4, 4, 512]
            Reshape((4, 4, 512)),

            Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [8, 8, 512]
            
            Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [16, 16, 256]
            
            Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [32, 32, 256]
            
            Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh', kernel_initializer=RandomNormal(0, 0.02)), # [64, 64, 3]
        ], name='generator')
        return generator
    
    def conv_block(filters, kernel_size, strides=1, padding='same'):
        return Sequential([
            Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),
            LeakyReLU(0.2)
        ])

    def build_discriminator(img_shape):
        discriminator = Sequential([
            Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape, kernel_initializer=RandomNormal(0, 0.02)),
            LeakyReLU(0.2),

            conv_block(128, 3, strides=2),
            conv_block(256, 3, strides=2),
            conv_block(512, 3, strides=2),

            Conv2D(1, 4, kernel_initializer=RandomNormal(0, 0.02)),
            Flatten()
        ], name='discriminator')
        return discriminator
    
    generator = build_generator(in_dim)
    discriminator = build_discriminator(img_shape)
    gan = Sequential([
        generator,
        discriminator
    ], 'gan')

    discriminator.compile(Adam(2e-4, beta_1=0.5), loss='mse')
    discriminator.trainable = False
    gan.compile(Adam(2e-4, beta_1=0.5), loss='mse')

    return gan, generator, discriminator


def train(model_dir, gan, generator, discriminator, X, batch_size, epochs, training_ratio=3, generate_imgs_frequency=None, generate_csv=False, seed=880301):
    root = model_dir
    img_dir = os.path.join(model_dir, 'imgs')
    model_dir = os.path.join(root, 'models')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if generate_csv:
        csv = open(os.path.join(root, 'logs.csv'), 'w')
        print('epoch,step,G_loss,D_loss', file=csv)

    np.random.seed(seed)
    step = 1
    training_ratio_list = training_ratio
    if not isinstance(training_ratio, (list, tuple)):
        training_ratio_list = [training_ratio] * epochs
    for epoch, training_ratio in zip(range(epochs), training_ratio_list):
        # pad the size of X to make it be the multiple of the batch size
        idx = np.concatenate([np.random.permutation((X.shape[0] + batch_size - 1) // batch_size * batch_size) % X.shape[0] for _ in range(training_ratio)])
        discriminator_trained_times = 0
        for i in range(0, idx.shape[0], batch_size):
            realX = X[idx[i:i+batch_size]]
            noise = np.random.uniform(size=(realX.shape[0],) + gan.input.shape[1:])
            
            fakeX = generator.predict_on_batch(noise)
            
            real_loss = discriminator.train_on_batch(realX, np.ones(realX.shape[0]))
            fake_loss = discriminator.train_on_batch(fakeX, np.zeros(fakeX.shape[0]))
            if discriminator_trained_times < training_ratio - 1:
                discriminator_trained_times += 1
                continue
            discriminator_trained_times = 0
            discriminator_loss = (real_loss + fake_loss) / 2

            noise = np.random.uniform(size=(realX.shape[0],) + gan.input.shape[1:])
            generator_loss = gan.train_on_batch(noise, np.ones(noise.shape[0]))
            print(f'[{min((i+batch_size) // training_ratio, X.shape[0])}/{X.shape[0]}], epoch: {epoch+1:0{len(str(epochs))}}/{epochs}, G_loss: {generator_loss:.5f}, D_loss: {discriminator_loss:.5f}', end='\r')
            if generate_imgs_frequency is not None and step % generate_imgs_frequency == 0:
                gan.save_weights(os.path.join(model_dir, f'{epoch+1:0{len(str(epochs))}}_{step:05}.h5'))
                utils.generate_grid_img(os.path.join(img_dir, f'{epoch+1:0{len(str(epochs))}}_{step:05}.jpg'), generator)
                if generate_csv:
                    print(f'{epoch+1},{step},{generator_loss},{discriminator_loss}', file=csv)
            step += 1
        print('')

        if generate_imgs_frequency is None:
            gan.save_weights(os.path.join(model_dir, f'{epoch+1:0{len(str(epochs))}}.h5'))
            utils.generate_grid_img(os.path.join(img_dir, f'{epoch+1:0{len(str(epochs))}}.jpg'), generator)
            if generate_csv:
                print(f'{epoch+1},{step},{generator_loss},{discriminator_loss}', file=csv)
