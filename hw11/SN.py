import os, sys, argparse
import importlib
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, Flatten, Activation, GlobalAveragePooling2D, LeakyReLU, Reshape, AveragePooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
import tensorflow.compat.v1 as tf
import cv2

import utils

class SpectralNormalization(Constraint):
    def __init__(self, power_iters=1):
        self.n_iters = power_iters

    def __call__(self, w):
        flattened_w = tf.reshape(w, [w.shape[0], -1])
        u = tf.random.normal([flattened_w.shape[0]])
        v = tf.random.normal([flattened_w.shape[1]])
        for _ in range(self.n_iters):
            v = tf.linalg.matvec(tf.transpose(flattened_w), u)
            v /= tf.linalg.norm(v + 1e-8)
            u = tf.linalg.matvec(flattened_w, v)
            u /= tf.linalg.norm(u + 1e-8)
        sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
        return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}

class ResBlockTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', use_bias=True, **kwargs):
        super(ResBlockTranspose, self).__init__(**kwargs)
        self.layers = [
            BatchNormalization(), Activation('relu'),
            UpSampling2D(),
            Conv2DTranspose(filters, kernel_size, padding=padding, use_bias=use_bias),
            BatchNormalization(), Activation('relu'),
            Conv2DTranspose(filters, kernel_size, padding=padding, use_bias=use_bias),
        ]
        self.shortcut_layers = [
            Conv2DTranspose(filters, 1, padding=padding, use_bias=use_bias),
            UpSampling2D()
        ]

    def call(self, input_tensor):
        out = input_tensor
        for layer in self.layers:
            out = layer(out)
        
        shortcut = input_tensor
        for layer in self.shortcut_layers:
            shortcut = layer(shortcut)
        
        return out + shortcut

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.layers = [
            Activation('relu'),
            Conv2D(filters, kernel_size, padding=padding, activation='relu', kernel_constraint=SpectralNormalization()),
            Conv2D(filters, kernel_size, padding=padding, kernel_constraint=SpectralNormalization()),
            AveragePooling2D()
        ]
        self.shortcut_layers = [
            AveragePooling2D(),
            Conv2D(filters, 1, padding=padding, kernel_constraint=SpectralNormalization()),
        ]

    def call(self, input_tensor):
        out = input_tensor
        for layer in self.layers:
            out = layer(out)
        
        shortcut = input_tensor
        for layer in self.shortcut_layers:
            shortcut = layer(shortcut)
        
        return out + shortcut

class OptimizedBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', **kwargs):
        super(OptimizedBlock, self).__init__(**kwargs)
        self.layers = [
            Conv2D(filters, kernel_size, padding=padding, activation='relu',
                   kernel_constraint=SpectralNormalization()),
            Conv2D(filters, kernel_size, padding=padding, kernel_constraint=SpectralNormalization()),
            AveragePooling2D()
        ]
        self.shortcut_layers = [
            AveragePooling2D(),
            Conv2D(filters, 1, padding=padding, kernel_constraint=SpectralNormalization())
        ]

    def call(self, input_tensor):
        out = input_tensor
        for layer in self.layers:
            out = layer(out)
        
        shortcut = input_tensor
        for layer in self.shortcut_layers:
            shortcut = layer(shortcut)
        
        return out + shortcut

    
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
            Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(0, 0.02), kernel_constraint=SpectralNormalization()),
            BatchNormalization(),
            LeakyReLU(0.2)
        ])

    def build_discriminator(img_shape):
        discriminator = Sequential([
            Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape, kernel_initializer=RandomNormal(0, 0.02), kernel_constraint=SpectralNormalization()),
            LeakyReLU(0.2),

            conv_block(128, 3, strides=2),
            conv_block(256, 3, strides=2),
            conv_block(512, 3, strides=2),

            Conv2D(1, 4, activation='sigmoid', kernel_initializer=RandomNormal(0, 0.02), kernel_constraint=SpectralNormalization()),
            Flatten()
        ], name='discriminator')
        return discriminator
    
    generator = build_generator(in_dim)
    discriminator = build_discriminator(img_shape)

    generator_in = Input((in_dim,), name='generator_in')
    generator_out = generator(generator_in)
    discriminator_out = discriminator(generator_out)
    gan = Model(generator_in, discriminator_out, name='gan')

    discriminator.compile(Adam(1e-4, beta_1=0.5), loss='binary_crossentropy')
    discriminator.trainable = False
    gan.compile(Adam(1e-4, beta_1=0.5), loss='binary_crossentropy')

    return gan, generator, discriminator

def build_resnet_model(img_shape, in_dim):
    def build_generator(in_dim):
        generator = Sequential([
            Dense(512* 4* 4, activation='relu', input_shape=(in_dim,), use_bias=False),
            Reshape((4, 4, 512)),  # [4, 4, 512]

            ResBlockTranspose(512, 3, padding='same', use_bias=False), # [8, 8, 512]
            ResBlockTranspose(256, 3, padding='same', use_bias=False), # [16, 16, 256]
            ResBlockTranspose(128, 3, padding='same', use_bias=False), # [32, 32, 256]
            ResBlockTranspose(64, 3, padding='same', use_bias=False),  # [64, 64, 256]
            
            BatchNormalization(),
            Conv2DTranspose(3, 3, padding='same', activation='tanh'), # [64, 64, 3]
        ], name='generator')
        return generator
    
    def build_discriminator(img_shape):
        discriminator = Sequential([
            OptimizedBlock(64, 3, input_shape=img_shape),  # [32, 32, 64]

            ResBlock(128, 3),  # [16, 16, 128]
            ResBlock(256, 3),  # [8, 8, 256]
            ResBlock(512, 3),  # [4, 4, 512]
            ResBlock(1024, 3),  # [2, 2, 512]

            Activation('relu'),
            GlobalAveragePooling2D(),
            Dense(1, kernel_constraint=SpectralNormalization())
        ], name='discriminator')
        return discriminator
    
    generator = build_generator(in_dim)
    discriminator = build_discriminator(img_shape)

    generator_in = Input((in_dim,), name='generator_in')
    generator_out = generator(generator_in)

    discriminator_out = discriminator(generator_out)
    gan = Model(generator_in, discriminator_out, name='gan')

    discriminator.compile(Adam(2e-4, beta_1=0.5), loss='hinge')
    discriminator.trainable = False
    gan.compile(Adam(2e-4, beta_1=0.5), loss='hinge')

    return gan, generator, discriminator

def build_model2(img_shape, in_dim):
    def build_generator(in_dim):
        generator = Sequential([
            Dense(512*4*4, activation='relu', input_shape=(in_dim,), use_bias=False, kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [4, 4, 512]
            Reshape((4, 4, 512)),

            Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [8, 8, 512]
            
            Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [16, 16, 128]
            
            Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [32, 32, 64]
            
            Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False, activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [64, 64, 64]
            
            Conv2DTranspose(3, 4, padding='same', activation='tanh', kernel_initializer=RandomNormal(0, 0.02)), # [64, 64, 3]
        ], name='generator')
        return generator
    
    def conv_block(filters, kernel_size, strides=1, padding='same'):
        return Sequential([
            Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(0, 0.02), kernel_constraint=SpectralNormalization()),
            BatchNormalization(),
            LeakyReLU(0.2)
        ])

    def build_discriminator(img_shape):
        discriminator = Sequential([
            Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape,
                   kernel_initializer=RandomNormal(0, 0.02), kernel_constraint=SpectralNormalization()),
            LeakyReLU(0.2),

            conv_block(64, 3, strides=2),
            conv_block(128, 3, strides=2),
            conv_block(256, 3, strides=2),
            conv_block(512, 3, strides=2),

            Conv2D(1, 2, kernel_initializer=RandomNormal(0, 0.02),
                   kernel_constraint=SpectralNormalization()),
            Flatten()
        ], name='discriminator')
        return discriminator
    
    generator = build_generator(in_dim)
    discriminator = build_discriminator(img_shape)

    generator_in = Input((in_dim,), name='generator_in')
    generator_out = generator(generator_in)
    discriminator_out = discriminator(generator_out)
    gan = Model(generator_in, discriminator_out, name='gan')

    discriminator.compile(Adam(1e-4, beta_1=0.5), loss='hinge')
    discriminator.trainable = False
    gan.compile(Adam(1e-4, beta_1=0.5), loss='hinge')

    return gan, generator, discriminator


def train(model_dir, gan, generator, discriminator, X, batch_size, epochs, training_ratio=3, generate_imgs_frequency=None, generate_csv=False, seed=880301):
    root = model_dir
    img_dir = os.path.join(model_dir, 'imgs')
    model_dir = os.path.join(root, 'models')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if len(gan.metrics_names) == 1:
        print_str = f'[{{}}/{X.shape[0]}], epoch: {{:0{len(str(epochs))}}}/{epochs}, ' + \
                'G_loss: {:.5f}, D_loss: {:.5f}'
    else:
        print_str = f'[{{}}/{X.shape[0]}], epoch: {{:0{len(str(epochs))}}}/{epochs}, ' + \
                'G_loss: {:.5f}, G_clf_loss: {:.5f}, G_tot_var_loss: {:.5f}, ' + \
                'D_loss: {:.5f}'
    if generate_csv:
        csv = open(os.path.join(root, 'logs.csv'), 'w')
        if len(gan.metrics_names) == 1:
            print('epoch,step,G_loss,D_loss', file=csv)
        else:
            print('epoch,step,G_loss,G_clf_loss,G_tot_var_loss,D_loss', file=csv)

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
            generator_losses = gan.train_on_batch(noise, [np.ones(noise.shape[0])] * max(1, len(gan.metrics_names) - 1))
            if len(gan.metrics_names) == 1:
                generator_losses = [generator_losses]
            
            print(print_str.format(min((i+batch_size) // training_ratio, X.shape[0]), epoch+1, *generator_losses, discriminator_loss), end='\r')
            if generate_imgs_frequency is not None and step % generate_imgs_frequency == 0:
                gan.save_weights(os.path.join(model_dir, f'{epoch+1:0{len(str(epochs))}}_{step:05}.h5'))
                utils.generate_grid_img(os.path.join(img_dir, f'{epoch+1:0{len(str(epochs))}}_{step:05}.jpg'), generator)
                if generate_csv:
                    print(epoch + 1, step, *generator_losses, discriminator_loss, sep=',', file=csv)
            step += 1
        print('')

        if generate_imgs_frequency is None:
            gan.save_weights(os.path.join(model_dir, f'{epoch+1:0{len(str(epochs))}}.h5'))
            utils.generate_grid_img(os.path.join(img_dir, f'{epoch+1:0{len(str(epochs))}}.jpg'), generator)
            if generate_csv:
                print(epoch + 1, step, *generator_losses, discriminator_loss, sep=',', file=csv)
