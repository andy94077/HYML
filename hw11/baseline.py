import os, sys, argparse
import importlib
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, Flatten, Activation, GlobalAveragePooling2D, LeakyReLU
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-r', '--seed', type=int, default=880301, help='random seed')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
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
    ensemble = args.ensemble
    function = args.model_function
    seed = args.seed
    input_shape = (96, 96)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model, generator, discriminator = globals()[function](input_shape + (3,), 100)
    discriminator.compile(Adam(2e-4, beta_1=0.5), loss='binary_crossentropy')
    model.compile(Adam(2e-4, beta_1=0.5), loss='binary_crossentropy')
    discriminator.trainable = False
    discriminator.summary()
    model.summary()
    
    if training:
        trainX = utils.load_data(data_dir, input_shape)
        print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')

        batch_size = 128
        checkpoint = ModelCheckpoint(model_path, 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_acc', 0.8, 5, verbose=1, min_lr=1e-4)
        #logger = CSVLogger(model_path+'.csv')
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        model.fit_generator(train_gen.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=trainX.shape[0]//batch_size, epochs=50, validation_data=(validX, validY), verbose=1, callbacks=[checkpoint, reduce_lr])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = np.random.uniform(-1, 1, size=input_shape + (3,))
        pred = model.predict(testX)
        if ensemble:
            np.save(test, pred)
        else:
            utils.generate_csv(pred, test)
    else:
        if not training:
            trainX = utils.load_data(data_dir, input_shape)
            print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')
        
