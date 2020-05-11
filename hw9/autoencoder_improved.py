import os, argparse
import importlib
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import cv2

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

def projective_transform(img):
    src_points = np.array([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]], np.float32)
    X = np.random.randint(0, int(0.125 * img.shape[1]), size=4)
    Y = np.random.randint(0, int(0.125 * img.shape[0]), size=4)
    dst_points = np.array([[X[0], Y[0]], [img.shape[1] - 1 - X[1], Y[1]], [X[2], img.shape[0] - 1 - Y[2]], [img.shape[1] - 1 - X[3], img.shape[0] - 1 - Y[3]]], np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    output = cv2.warpPerspective(img, projective_matrix, img.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file', help='training data')
    parser.add_argument('-v', '--validation-file', help='validation data')
    parser.add_argument('-y', '--validation-labels', help='validation label')
    parser.add_argument('-f', '--model-function', default='build_autoencoder')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, type=str, help='testing file and predicted file')
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
    validY_path = args.validation_labels
    function = args.model_function
    training = not args.no_training
    test = args.test
    input_shape = (32, 32)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, model = globals()[function](input_shape + (3,))  # ignore the decoder model
    model.compile(Adam(1e-3, decay=5e-4), loss='mse')#SGD(0.1, momentum=0.9, decay=5e-4), loss='mse')#
    model.summary()

    if training:
        trainX = utils.load_data(trainX_path, normalize=True, preprocessing=False)
        if validX_path is not None:
            validX = utils.load_data(validX_path, normalize=True, preprocessing=False)
        else:
            trainX, validX = train_test_split(trainX, test_size=0.1, random_state=880301)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')
        train_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1, preprocessing_function=projective_transform)

        batch_size = 256
        checkpoint = ModelCheckpoint(model_path, 'loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('loss', 0.5, 10, verbose=1, min_lr=1e-5)
        #logger = CSVLogger(model_path+'.csv')
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        model.fit_generator(train_gen.flow(trainX, trainX, batch_size=batch_size), steps_per_epoch=trainX.shape[0]/batch_size, epochs=500, validation_data=(validX, validX), verbose=1, callbacks=[checkpoint, reduce_lr])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = utils.load_data(test[0], normalize=True, preprocessing=False)
        pred = encoder.predict(testX, batch_size=256)
        np.save(test[1], pred)
    else:
        if not training:
            trainX = utils.load_data(trainX_path, normalize=True, preprocessing=False)
            validX = utils.load_data(validX_path, normalize=True, preprocessing=False)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')
        print(f'\033[32;1mTraining loss: {model.evaluate(trainX, trainX, batch_size=256, verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton loss: {model.evaluate(validX, validX, batch_size=256, verbose=0)}\033[0m')
        if validY_path:
            validY = np.load(validY_path)
            latentX = encoder.predict(validX, batch_size=256).reshape(validX.shape[0], -1)
            clf = LinearSVC(max_iter=2000).fit(latentX, validY)
            print(f'\033[32;1mValidaiton acc: {clf.score(latentX, validY)}\033[0m')

