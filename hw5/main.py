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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

def build_model(input_shape, output_dim):
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

def plot_saliency_map(model, X, fig_name):
    K.set_learning_phase(0)
    X = X.copy()
    saliency_map = []
    Y = np.argmax(model.predict_on_batch(X), axis=-1)
    for i, x in enumerate(X):
        grad_tensor = K.gradients(model.output[:, Y[i]], model.input)[0]
        get_grad = K.function([model.input], [grad_tensor])

        grad = np.abs(get_grad([[x]])[0][0])
        saliency_map.append((grad - np.min(grad, axis=(1, 2), keepdims=True)) / (np.max(grad, axis=(1, 2), keepdims=True) - np.min(grad, axis=(1, 2), keepdims=True) + 1e-8))
    saliency_map = np.array(saliency_map)
    saliency_map = np.clip((saliency_map) * 255, 0, 255).astype(np.uint8)

    X **= 1.5 # restore gamma correction
    X = np.clip(X, 0, 255).astype(np.uint8)

    ## plot figures
    fig, axs = plt.subplots(2, X.shape[0], figsize=(15, 8))
    for row, target in enumerate([X, saliency_map]):
        for col, img in enumerate(target):
            axs[row][col].imshow(img[..., ::-1]) # BGR to RGB
    plt.title('saliency map')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return saliency_map

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-f', '--model-function', default='build_model')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    function = args.model_function
    input_shape = (128, 128)

    trainX, trainY = utils.load_train_data(data_dir, input_shape, normalize=False, preprocessing=True)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model = globals()[function](input_shape + (3,), 11)
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    print('\033[32;1mLoading Model\033[0m')
    model.load_weights(model_path)

    images = trainX[[83, 4218, 4707, 8598]]
    plot_saliency_map(model, images, '1.jpg')
