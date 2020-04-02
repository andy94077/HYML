import os, sys, argparse
import importlib
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf

import utils

def build_model2(input_shape, output_dim):
    model = Sequential([
        Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, 3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(output_dim, activation='softmax')
        ])
    return model

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-f', '--model-function', default='build_model2')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-a', '--augment', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    training = not args.no_training
    function = args.model_function
    normalize = args.normalize
    augment = args.augment
    input_shape = (128, 128)

    trainX, trainY = utils.load_train_data(data_dir, input_shape, normalize=normalize, preprocessing=True)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model = globals()[function](input_shape + (3,), 11)
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    if training:
        batch_size = 128
        checkpoint = ModelCheckpoint(model_path, 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_acc', 0.8, 5, verbose=1, min_lr=1e-4)
        #logger = CSVLogger(model_path+'.csv')
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        if augment:
            train_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='constant', cval=0)
            model.fit_generator(train_gen.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=trainX.shape[0]//batch_size, epochs=50, validation_data=(validX, validY), verbose=1, callbacks=[checkpoint, reduce_lr])
        else:
            model.fit(trainX, trainY, batch_size=batch_size, epochs=50, validation_data=(validX, validY), verbose=1, callbacks=[checkpoint, reduce_lr])

    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    print(f'\033[32;1mTraining score: {model.evaluate(trainX, trainY, batch_size=128, verbose=0)}\033[0m')
    print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=128, verbose=0)}\033[0m')
        
