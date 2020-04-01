import os, sys, argparse
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, GlobalAveragePooling2D, Add
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import tensorflow as tf

import utils

def residual_block(input_x, filters, kernel_size=3, strides=1, match_dim=False):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if match_dim:
        input_x = Conv2D(filters, 1, padding='same', strides=strides)(input_x)
    x = Add()([x, input_x])
    x = Activation('relu')(x)
    return x

def Resnet18(input_shape, output_dim):
    input_tensor = Input(input_shape)
    x = Conv2D(64, 3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, strides=2, match_dim=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, strides=2, match_dim=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, strides=2, match_dim=True)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    out = Dense(output_dim, activation='softmax')(x)
    return Model(input_tensor, out)

