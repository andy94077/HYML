import os, sys, argparse
import importlib
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, Flatten, Activation, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf

import utils

def build_model(img_shape, in_dim):
    def build_generator(in_dim):
        generator = Sequential([
            Conv2DTranspose(1024, 6, strides=6, padding='same', activation='relu', input_shape=(1, 1, in_dim), kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [6, 6, 1024]

            Conv2DTranspose(512, 5, strides=2, padding='same', activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [12, 12, 512]
            
            Conv2DTranspose(256, 5, strides=2, padding='same', activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(), # [24, 24, 256]
            
            Conv2DTranspose(128, 5, strides=2, padding='same', activation='relu', kernel_initializer=RandomNormal(0, 0.02)),
            BatchNormalization(),  # [48, 48, 256]
            
            Conv2DTranspose(3, 5, strides=2, padding='same', activation='tanh', kernel_initializer=RandomNormal(0, 0.02)), # [96, 96, 3]
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
            Conv2D(64, 3, padding='same', input_shape=img_shape, kernel_initializer=RandomNormal(0, 0.02)),
            LeakyReLU(0.2),

            conv_block(64, 3),
            conv_block(64, 3, strides=2),

            conv_block(128, 3),
            conv_block(128, 3, strides=2),
            
            conv_block(256, 3, strides=2),
            conv_block(256, 3, strides=2),

            conv_block(512, 3, strides=2),

            Conv2D(1, 3, activation='sigmoid', kernel_initializer=RandomNormal(0, 0.02)),
            Flatten()
        ], name='discriminator')
        return discriminator
    
    generator = build_generator(in_dim)
    discriminator = build_discriminator(img_shape)
    gan = Sequential([
        generator,
        discriminator
    ], 'gan')

    return gan, generator, discriminator

def train(model_path, gan, generator, discriminator, X, batch_size, epochs, training_ratio=3, seed=880301):
    img_dir = model_path[:model_path.rfind('.h5')] + '_imgs'
    os.makedirs(img_dir, exist_ok=True)
    
    dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(20 * batch_size).batch(batch_size).repeat(training_ratio).prefetch(tf.data.experimental.AUTOTUNE)
    np.random.seed(seed)
    discriminator_trained_times = 0
    for epoch in range(epochs):
        for realX in dataset:
            noise = np.random.uniform(size=(realX.shape[0], gan.input.shape[1]))

            fakeX = generator.predict_on_batch(noise)

            discriminator.trainable = True
            real_loss = discriminator.train_on_batch(realX, np.ones(realX.shape[0]))
            fake_loss = discriminator.train_on_batch(fakeX, np.zeros(fakeX.shape[0]))
            if discriminator_trained_times < training_ratio:
                discriminator_trained_times += 1
                continue
            discriminator_trained_times = 0
            discriminator_loss = (real_loss + fake_loss) / 2

            discriminator.trainable = False
            generator_loss = gan.train_on_batch(noise, np.ones(noise.shape[0]))

            print(f'epoch: {epoch:0{len(str(epochs))}}/{epochs}, gen_loss: {generator_loss:.5f}, dis_loss: {discriminator_loss:.5f}', end='\r')
        print('')

        gan.save_weights(model_path)
        generate_grid_img(os.path.join(img_dir, f'{i:0{len(str(epochs))}}.jpg', generator)

def generate_grid_img(img_path, generator, grid_size=(2, 8), seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.uniform(size=(grid_size[0] * grid_size[1], generator.input[1:]))
    imgs = generator.predict_on_batch(noise)
    imgs = imgs.reshape(grid_size+imgs.shape[1:])
    grid = []
    for i in range(imgs.shape[0]):
        grid.append(np.concatenate(imgs[i:i + 1], axis=1))
    grid = np.concatenate(grid, axis=0)
    cv2.imwrite(img_path, grid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-r', '--seed', type=int, default=880301, help='random seed')
    parser.add_argument('-g', '--gpu', default='3', help='available gpu device')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    data_dir = args.data_dir
    model_path = args.model_path
    training = not args.no_training
    test = args.test
    function = args.model_function
    seed = args.seed
    input_shape = (96, 96)
    in_dim = 100

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model, generator, discriminator = globals()[function](input_shape + (3,), in_dim)
    discriminator.compile(Adam(2e-4, beta_1=0.5), loss='binary_crossentropy')
    discriminator.trainable = False
    model.compile(Adam(2e-4, beta_1=0.5), loss='binary_crossentropy')
    discriminator.summary()
    model.summary()
    
    if training:
        trainX = utils.load_data(data_dir, input_shape)
        print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')

        batch_size = 128
        train(model_path, model, generator, discriminator, batch_size=batch_size, epochs=100)
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        generate_grid_img(test, generator, seed=seed)
