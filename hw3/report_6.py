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
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    parser.add_argument('-T', '--no-training', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    training = not args.no_training
    input_shape = (128, 128)

    trainX, trainY = utils.load_train_data(data_dir, input_shape, normalize=False, preprocessing=True)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')

    model = build_model2(input_shape + (3,), 11)
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    print('\033[32;1mLoading Model\033[0m')
    model.load_weights(model_path)
    y_pred = np.argmax(model.predict(validX), axis=1)
    matrix = confusion_matrix(np.argmax(validY, axis=1), y_pred, normalize='true')
    labels = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
    ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='nipy_spectral', xticklabels=labels, yticklabels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('report_6.jpg', bbox_inches='tight')

