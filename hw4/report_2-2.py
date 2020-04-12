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


def build_model(embedding):
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
        ])
    return model

def build_model2(embedding):
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dense(1, activation='sigmoid')
        ])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word2vec_model_path')
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-files', nargs=2, help='labeled training file and unlabeled training file')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, type=str, help='predicted file')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-g', '--gpu', type=str, default='3')
    args = parser.parse_args()

    word2vec_model_path = args.word2vec_model_path
    model_path = args.model_path
    labeled_path, unlabeled_path = args.training_files if args.training_files else [None, None]
    training = not args.no_training
    test = args.test
    ensemble = args.ensemble
    function = args.model_function
    lr = args.lr

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    max_seq_len = 32
    w2v_model = Word2Vec().load(word2vec_model_path)
    word2idx = w2v_model.get_word2idx()
    embedding = w2v_model.get_embedding()
    vocabulary_size = len(word2idx)
    print(f'\033[32;1mvocabulary_size: {vocabulary_size}\033[0m')

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model = globals()[function](embedding)
    model.compile(Adam(lr), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    if training:
        trainX, trainY = utils.load_train_data(labeled_path, word2idx, max_seq_len)
        trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 2, verbose=1, min_lr=1e-5)
        logger = CSVLogger(model_path+'.csv', append=True)
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, write_images=True, update_freq=512)
        model.fit(trainX, trainY, validation_data=(validX, validY), batch_size=256, epochs=10, callbacks=[checkpoint, reduce_lr, logger])

    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    sentences = [text_to_word_sequence('today is a good day, but it is hot'), text_to_word_sequence('today is hot, but it is a good day')]
    sentences = utils.data_preprocessing(sentences, word2idx, max_seq_len)
    print(model.predict(sentences))
