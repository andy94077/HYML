import os, sys, argparse
import numpy as np
from tqdm import trange
from keras.models import Model
from keras.layers import Input, GRU, Dense, RepeatVector, Lambda, Concatenate, Flatten, CuDNNGRU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf

import utils

def build_model(hidden_dim):
    gru_in = Input((9, 18), name='input')
    #x = Flatten()(gru_in)
    #x = Dense(162, kernel_regularizer=l2(1e-2))(x)
    x = CuDNNGRU(hidden_dim, kernel_regularizer=l2(5e-3))(gru_in)
    #x = GRU(hidden_dim, activation=None, kernel_regularizer=l2(1e-5))(gru_in)
    #gru_out = Dense(hidden_dim, activation='relu')(x)
    out = Dense(1, name='output')(x)#, kernel_regularizer=l2(1e-3))(gru_out)
    return Model(gru_in, out)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, help='testing file and the predicted file')
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.training_file
    training = not args.no_training
    test = args.test

    model = build_model(64)
    #model.compile(Adam(1e-3), loss='mse')#, metrics=['mse'])
    model.compile('rmsprop', loss='mse')#, metrics=['mse'])

    if os.path.exists(model_path):
        print('\033[32;1mLoading Model\033[0m')
        model.load_weights(model_path)
        mean = np.load('mean_best.npy')
        std = np.load('std_best.npy')
    if training:
        trainX, trainY, mean, std = utils.load_train_data_RNN(train_file, 9)
        trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        np.save('mean_best.npy', mean)
        np.save('std_best.npy', std)

        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 20, verbose=1, min_lr=1e-6)
        logger = CSVLogger(model_path+'.csv', append=True)
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        model.fit(trainX, trainY, batch_size=256, epochs=500, validation_data=(validX, validY), verbose=2, callbacks=[checkpoint, reduce_lr, logger])

    model.load_weights(model_path)
    if test:
        testX = utils.load_test_data_RNN(test[0], mean, std)
        utils.generate_csv(model.predict(testX), test[1])
    else:
        if not training:
            trainX, trainY, mean, std = utils.load_train_data_RNN(train_file, 9)
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        print(f'\033[32;1mTraining score: {model.evaluate(trainX, trainY, batch_size=256, verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=256, verbose=0)}\033[0m')
        
