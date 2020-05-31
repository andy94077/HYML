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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_dir')
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
    model_dir = args.model_dir
    training = not args.no_training
    test = args.test
    function = args.model_function
    seed = args.seed
    input_shape = (64, 64)
    in_dim = 100

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
        globals()['train'] = getattr(importlib.import_module(function[:function.rfind('.')]), 'train')
    model, generator, discriminator = globals()[function](input_shape + (3,), in_dim)
    discriminator.summary()
    model.summary()
    
    if training:
        trainX = utils.load_data(data_dir, input_shape)
        print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')

        batch_size = 128
        if 'baseline' in function:
            kwargs = dict(epochs=40, training_ratio=[3] * 30 + [2] * 10)
        else:
            kwargs = dict(epochs=100, training_ratio=[3] * 40 + [2] * 40 + [1] * 20)
        globals()['train'](model_dir, model, generator, discriminator, trainX, batch_size=batch_size, generate_imgs_frequency=200, generate_csv=True, seed=seed, **kwargs)
    else:
        print('\033[32;1mLoading Model\033[0m')
        model.load_weights(model_dir)
    
    if test:
        utils.generate_grid_img(test, generator, grid_size=(3, 8), seed=seed)
