import os, argparse
import importlib
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Conv2DTranspose, GlobalAveragePooling2D, Reshape, Add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
import cv2

import utils

def build_baseline(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape):
        encoder = Sequential([
            Conv2D(12, 4, strides=2, padding='same', activation='relu', input_shape=input_shape),
            Conv2D(24, 4, strides=2, padding='same', activation='relu'),
            Conv2D(48, 4, strides=2, padding='same', activation='relu'),
        ], name='encoder')
        return encoder

    def build_decoder(hidden_shape):
        decoder = Sequential([
            Conv2DTranspose(24, 4, strides=2, padding='same', activation='relu', input_shape=hidden_shape),
            Conv2DTranspose(12, 4, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
        ], name='decoder')
        return decoder

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[-3:])
    autoencoder = Sequential([
        encoder,
        decoder
    ])
    autoencoder.compile(Adam(1e-3, decay=5e-4), loss='mse')

    return encoder, decoder, autoencoder


def build_autoencoder(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape):
        encoder = Sequential([
            Conv2D(512, 4, strides=2, padding='same', activation='relu', input_shape=input_shape),
            Conv2D(128, 4, strides=2, padding='same', activation='relu'),
            Conv2D(64, 4, strides=2, padding='same', activation='relu'),
            Conv2D(64, 4, strides=2, padding='same', activation='relu'),
        ])
        return encoder

    def build_decoder(hidden_shape):
        decoder = Sequential([
            Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu', input_shape=hidden_shape),
            Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
        ])
        return decoder

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[-3:])
    autoencoder = Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(Adam(1e-3, decay=5e-4), loss='mse')

    return encoder, decoder, autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('test_file')
    parser.add_argument('output_image', help='Output original image and reconstructed image')
    parser.add_argument('-f', '--model-function', default='build_autoencoder')
    parser.add_argument('-g', '--gpu', default='3', help='Available gpu device')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_path = args.model_path
    testX_path = args.test_file
    output_image = args.output_image
    function = args.model_function
    input_shape = (32, 32)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, model = globals()[function](input_shape + (3,))  # ignore the decoder model
    model.summary()

    print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    testX = np.load(testX_path)

    pred = model.predict(testX, batch_size=512)

    mse = np.mean((pred - testX)** 2, axis=(1, 2, 3))
    min_mse2_idx = np.argpartition(mse, 2)[:2]
    max_mse2_idx = np.argpartition(mse, -2)[-2:]
    idx = np.concatenate([min_mse2_idx, max_mse2_idx])
    print(mse[idx])

    testX = np.concatenate(testX[idx] * 128 + 128, axis=1).astype(np.uint8)
    pred = np.concatenate(pred[idx] * 128 + 128, axis=1).astype(np.uint8)
    cv2.imwrite(output_image, np.concatenate([testX, pred], axis=0))

