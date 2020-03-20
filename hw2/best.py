import os, sys, argparse
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf

import utils

def build_model(input_dim):
    ii = Input((input_dim,), name='input')
    x = Dense(32, activation='sigmoid')(ii)
    out = Dense(1, activation='sigmoid', name='output')(x)
    return Model(ii, out)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file', nargs=2, help='trainX and trainY')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, help='testing file and the predicted file')
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.training_file
    training = not args.no_training
    test = args.test

    trainX, trainY, mean, std = utils.load_train_data(train_file[0], train_file[1])
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')

    model = build_model(trainX.shape[1])
    model.compile(Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    if training:
        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 4, verbose=1, min_lr=1e-6)
        #logger = CSVLogger(model_path+'.csv')
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        model.fit(trainX, trainY, batch_size=128, epochs=100, validation_data=(validX, validY), verbose=1, callbacks=[checkpoint, reduce_lr])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = utils.load_test_data(test[0], mean, std)
        pred = model.predict(testX)
        utils.generate_csv(pred, test[1])
        np.save(test[1] + '.npy', pred)
    else:
        print(f'\033[32;1mTraining score: {model.evaluate(trainX, trainY, batch_size=256, verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=256, verbose=0)}\033[0m')
        
