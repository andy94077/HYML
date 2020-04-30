import os, sys, argparse
import importlib
import numpy as np
from tqdm import tqdm
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

def load_train_data(data_dir, input_shape):
    val_filename = os.listdir(os.path.join(data_dir, 'validation'))
    train_filename = sorted(os.listdir(os.path.join(data_dir, 'training')) + [name[:-3] + '_.jpg' for name in val_filename])
    X = np.array([cv2.resize(cv2.imread(os.path.join(data_dir, 'training', n) if n[-5] != '_' else os.path.join(data_dir, 'validation', n[:-5]+'jpg')), input_shape) for n in tqdm(train_filename)])
    Y = np.array([int(n.split('_')[0]) for n in train_filename])

    X = X.astype(np.float32)
    utils.data_preprocessing(X)
    Y = to_categorical(Y)
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-o', '--output-dir', default='.')
    parser.add_argument('-g', '--gpu', default='3')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    training = not args.no_training
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    input_shape = (128, 128)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    trainX, trainY = load_train_data(data_dir, input_shape)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')

    model = build_model(input_shape + (3,), 11)
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    
    if training:
        train_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='constant', cval=0)

        batch_size = 128
        checkpoint = ModelCheckpoint(model_path, 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_acc', 0.8, 5, verbose=1, min_lr=1e-4)
        model.fit_generator(train_gen.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=trainX.shape[0] // batch_size, epochs=50, validation_data=(validX, validY), verbose=1, callbacks=[checkpoint, reduce_lr])
        
        model.load_weights(model_path)
        print(f'\033[32;1mTraining score: {model.evaluate(trainX, trainY, batch_size=128, verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=128, verbose=0)}\033[0m')
    else:
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.jpg'), bbox_inches='tight')

