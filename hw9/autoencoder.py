import os, argparse
import importlib
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Conv2DTranspose, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf

import utils

def build_autoencoder(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape):
        encoder = Sequential([
            Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
            MaxPooling2D(),

            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D()
        ])
        return encoder

    def build_decoder(hidden_shape):
        decoder = Sequential([
            Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu', input_shape=hidden_shape),
            Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh')
        ])
        return decoder

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[-3:])
    autoencoder = Sequential([
        encoder,
        decoder
    ])
    return encoder, decoder, autoencoder

def build_autoencoder2(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape):
        encoder = Sequential([
            Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
            MaxPooling2D(),

            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(),

            Conv2D(512, 3, padding='same', activation='relu'),
            MaxPooling2D()
        ])
        return encoder

    def build_decoder(hidden_shape):
        decoder = Sequential([
            Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu', input_shape=hidden_shape),
            Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh')
        ])
        return decoder

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[-3:])
    autoencoder = Sequential([
        encoder,
        decoder
    ])
    return encoder, decoder, autoencoder

def build_autoencoder3(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape):
        encoder = Sequential([
            Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
            MaxPooling2D(),

            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(),

            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D()
        ])
        return encoder

    def build_decoder(hidden_shape):
        decoder = Sequential([
            Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu', input_shape=hidden_shape),
            Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh')
        ])
        return decoder

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[-3:])
    autoencoder = Sequential([
        encoder,
        decoder
    ])
    return encoder, decoder, autoencoder

def build_autoencoder4(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape):
        encoder = Sequential([
            Conv2D(64, 3, strides=2, padding='same', activation='relu', input_shape=input_shape),
            Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            Conv2D(256, 3, strides=2, padding='same', activation='relu'),
            Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        ])
        return encoder

    def build_decoder(hidden_shape):
        decoder = Sequential([
            Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu', input_shape=hidden_shape),
            Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh')
        ])
        return decoder

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[-3:])
    autoencoder = Sequential([
        encoder,
        decoder
    ])
    return encoder, decoder, autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file', help='training data')
    parser.add_argument('-f', '--model-function', default='build_autoencoder')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, type=str, help='testing file and predicted file')
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-g', '--gpu', default='3', help='available gpu device')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_path = args.model_path
    trainX_path = args.training_file
    function = args.model_function
    training = not args.no_training
    test = args.test
    input_shape = (32, 32)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, model = globals()[function](input_shape + (3,))  # ignore the decoder model
    model.compile(Adam(1e-3), loss='mse')
    model.summary()

    if training:
        trainX = utils.load_data(trainX_path, normalize=True)
        print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')

        batch_size = 256
        checkpoint = ModelCheckpoint(model_path, 'loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('loss', 0.8, 5, verbose=1, min_lr=1e-5)
        #logger = CSVLogger(model_path+'.csv')
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        model.fit(trainX, trainX, batch_size=batch_size, epochs=args.epoch, verbose=1, callbacks=[checkpoint, reduce_lr])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = utils.load_data(test[0], normalize=True)
        pred = encoder.predict(testX, batch_size=256)
        np.save(test[1], pred)
    else:
        if not training:
            trainX = utils.load_data(trainX_path, normalize=True)
            print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')
        print(f'\033[32;1mTraining loss: {model.evaluate(trainX, trainX, batch_size=256, verbose=0)}\033[0m')

