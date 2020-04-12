import os, sys, argparse
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, CuDNNGRU, Dense, Embedding, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.preprocessing.text import text_to_word_sequence
from keras import backend as K
import tensorflow as tf

import utils
from word2vec import Word2Vec

def bag_of_word(X, vocabulary_size):
    return np.apply_along_axis(lambda a: np.histogram(a, bins=vocabulary_size, range=(0, vocabulary_size - 1))[0], axis=1, arr=X).astype(np.float32)

def bag_of_word_generator(X, Y, vocabulary_size, batch_size):
    i = 0
    while True:
        yield np.apply_along_axis(lambda a: np.histogram(a, bins=vocabulary_size, range=(0, vocabulary_size - 1))[0], axis=1, arr=X[i:i + batch_size]).astype(np.float32), Y[i:i + batch_size]
        i = (i + batch_size) % len(X)

def build_model(input_shape):
    model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
        ])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word2vec_model_path')
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-files', nargs=2, help='labeled training file and unlabeled training file')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-g', '--gpu', type=str, default='3')
    args = parser.parse_args()

    word2vec_model_path = args.word2vec_model_path
    model_path = args.model_path
    labeled_path, unlabeled_path = args.training_files if args.training_files else [None, None]
    training = not args.no_training
    function = args.model_function

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    max_seq_len = 32
    w2v_model = Word2Vec().load(word2vec_model_path)
    word2idx = w2v_model.get_word2idx()
    vocabulary_size = len(word2idx)
    print(f'\033[32;1mvocabulary_size: {vocabulary_size}\033[0m')

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model = globals()[function]((len(word2idx),))
    model.compile(Adam(1e-3), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    if training:
        trainX, trainY = utils.load_train_data(labeled_path, word2idx, max_seq_len)
        trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
        validX = bag_of_word(validX, vocabulary_size)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 2, verbose=1, min_lr=1e-5)
        #logger = CSVLogger(model_path+'.csv', append=True)
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, write_images=True, update_freq=512)
        model.fit_generator(bag_of_word_generator(trainX, trainY, vocabulary_size, batch_size=256), steps_per_epoch=int(np.ceil(trainX.shape[0] / 256)), validation_data=(validX, validY), epochs=10, callbacks=[checkpoint, reduce_lr])

    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    #sentences = [text_to_word_sequence('today is a good day, but it is hot'), text_to_word_sequence('today is hot, but it is a good day')]
    #sentences = utils.data_preprocessing(sentences, word2idx, max_seq_len)
    #sentences = bag_of_word(sentences, vocabulary_size)
    #print(model.predict(sentences))
    if not training:
        trainX, trainY = utils.load_train_data(labeled_path, word2idx, max_seq_len)
        trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
        validX = bag_of_word(validX, vocabulary_size)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
    print(f'\033[32;1mTraining score: {model.evaluate_generator(bag_of_word_generator(trainX, trainY, vocabulary_size, batch_size=256), steps=int(np.ceil(trainX.shape[0] / 256)), verbose=0)}\033[0m')
    print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=256, verbose=0)}\033[0m')
