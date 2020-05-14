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
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import utils

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file', help='training data')
    parser.add_argument('-v', '--validation-file', help='validation data')
    parser.add_argument('-f', '--model-function', default='build_autoencoder3')
    parser.add_argument('-g', '--gpu', default='3', help='available gpu device')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_path = args.model_path
    trainX_path = args.training_file
    validX_path = args.validation_file
    function = args.model_function
    input_shape = (32, 32)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, model = globals()[function](input_shape + (3,))  # ignore the decoder model
    model.compile(Adam(1e-3), loss='mse')
    model.summary()

    trainX = utils.load_data(trainX_path, normalize=True, preprocessing=False)
    if validX_path is not None:
        validX = utils.load_data(validX_path, normalize=True, preprocessing=False)
    else:
        trainX, validX = train_test_split(trainX, test_size=0.1, random_state=880301)
    print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')

    batch_size = 256
    checkpoint = ModelCheckpoint(model_path[:model_path.rfind('.h5')]+'_epoch_{epoch}.h5', 'loss', verbose=1, save_best_only=False, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('loss', 0.8, 5, verbose=1, min_lr=1e-5)
    logger = CSVLogger(model_path+'.csv')
    history = model.fit(trainX, trainX, batch_size=batch_size, epochs=100, validation_data=(validX, validX), verbose=1, callbacks=[checkpoint, reduce_lr, logger])

