import os, argparse
import importlib
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Conv2DTranspose, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    parser.add_argument('training_file')
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
    function = args.model_function
    input_shape = (32, 32)

    trainX = utils.load_data(trainX_path, normalize=True)[[1, 2, 3, 6, 7, 9]]

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, model = globals()[function](input_shape + (3,))  # ignore the decoder model
    model.compile(Adam(1e-3), loss='mse')
    model.summary()

    print('\033[32;1mLoading Model\033[0m')
    model.load_weights(model_path)
    decoder_out=(model.predict(trainX) * 128 + 127.5).clip(0, 255).astype(np.uint8)
    trainX = (trainX * 128 + 127.5).astype(np.uint8)

    fig, axs = plt.subplots(2, 6, figsize=(10, 3))
    for i in range(trainX.shape[0]):
        axs[0, i].imshow(trainX[i])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].imshow(decoder_out[i])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
    
    fig.savefig('report_2.jpg', bbox_inches='tight')
        
