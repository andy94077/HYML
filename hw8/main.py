import os, sys, argparse
import numpy as np
from tqdm import trange
from keras.models import Sequential, Model
from keras.layers import Input, GRU, Dense, RepeatVector, Lambda, Concatenate, Bidirectional, Embedding
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Mean, SparseCategoricalAccuracy
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf
import nltk

import utils

def train_data_preprocessing(inputX, word2idx, max_seq_len):
    X = [[word2idx['<BOS>']]+[word2idx[w] for w in ll[:min(len(ll), max_seq_len-2)]]+[word2idx['<EOS>']] for ll in inputX]
    trainX, Y = X[:-1], X[1:]
    line_len = np.array([min(len(i), max_seq_len) for i in Y])
    trainX = pad_sequences(trainX, max_seq_len, padding='post', truncating='post', value=word2idx[''])
    Y = pad_sequences(Y, max_seq_len+1, padding='post', truncating='post', value=word2idx[''])
    np.random.seed(880301)
    idx = np.random.permutation(trainX.shape[0])
    train_seq, valid_seq = idx[:int(trainX.shape[0]*0.9)], idx[int(trainX.shape[0]*0.9):]
    trainX, validX = trainX[train_seq], trainX[valid_seq]
    trainY_SOS, validY_SOS = Y[train_seq, :-1], Y[valid_seq, :-1]
    trainY, validY = Y[train_seq, 1:], Y[valid_seq, 1:]
    line_len, valid_line_len = line_len[train_seq], line_len[valid_seq]
    return trainX, validX, trainY_SOS, validY_SOS, trainY, validY, line_len, valid_line_len


def build_model(hidden_dim, max_seq_len, vocabulary_size):
    def build_encoder(embedding_dim, hidden_dim, vocabulary_size):
        return Sequential([
            Embedding(input_shape=(max_seq_len,))
        ])
    ## encoder Input and layers
    encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
    ith_str = Input((1,), dtype='int32', name='ith_str')
    word = Input((1,), dtype='int32', name='word')
    OneHot = Lambda(lambda x: K.one_hot(x, vocabulary_size_en), name='OneHot')

    ## building encoder
    encoder_in_and_word = Concatenate()([ith_str, word, encoder_in])
    encoder_out, state = GRU(hidden_dim, return_state=True)(OneHot(encoder_in_and_word))
    encoder_out_dup = RepeatVector(max_seq_len)(encoder_out)

    ## decoder Input and layers
    decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
    ith = Input((1,), dtype='int32', name='ith')
    decoder_GRU = GRU(hidden_dim, return_sequences=True, return_state=True)
    decoder_Dense = Dense(vocabulary_size_en, activation='softmax', name='decoder_out')

    ## building decoder
    ith_dup = RepeatVector(max_seq_len)(K.cast(ith, 'float'))
    word_dup = K.reshape(RepeatVector(max_seq_len)(word), (-1, max_seq_len))
    x = Concatenate()([ith_dup, OneHot(word_dup), OneHot(decoder_in), encoder_out_dup])
    x, _ = decoder_GRU(x, initial_state=state)
    decoder_out = decoder_Dense(x)

    ## get the specific word
    gather = K.concatenate([K.reshape(tf.range(K.shape(decoder_out)[0]), (-1, 1)), ith])
    specific_word = tf.gather_nd(decoder_out, gather)
    specific_word = Lambda(tf.identity, name='word_out')(specific_word) # Add this layer because the name of tf.gather_nd is too ugly

    model = Model([encoder_in, decoder_in, ith, ith_str, word], [decoder_out, specific_word])
    encoder = Model([encoder_in, ith_str, word], [encoder_out, state])

    ## building decoder model given encoder_out and states
    decoder_in_one_word = Input((1,), dtype='int32', name='decoder_in_one_word')
    decoder_state_in = Input((hidden_dim,), name='decoder_state_in')
    encoder_out = Input((hidden_dim,), name='decoder_encoder_out')
    x = Concatenate()([K.cast(ith, 'float')[:, tf.newaxis], OneHot(word), OneHot(decoder_in_one_word), encoder_out[:, tf.newaxis]])
    x, decoder_state = decoder_GRU(x, initial_state=decoder_state_in)
    decoder_out = decoder_Dense(x)
    decoder = Model([decoder_in_one_word, encoder_out, decoder_state_in, ith, word], [decoder_out, decoder_state])
    return model, encoder, decoder


def decode_sequence(encoder, decoder, testX, max_seq_len, word2idx, beam_search=False):
    '''Return list of sequences outputed by the decoder given batched input sequences.
    
    Args:
        encoder: An encoder model.
        decoder: An decoder model.
        testX: A batched list of sequences. i.e. [[w_0, w_1, ..., w_n], ...].
            The size of `testX` should not be too large.
        max_seq_len: The maximum sequence length.
        word2idx: A dict with words as keys and indexs as values.
        beam_search: A bool represents whether beam search is enabled.
    
    Returns:
        A list of sequences padded to `max_seq_len`.
        Note that it does not contain i(<BOS>) in the beginning, bus it does contain i(<EOS>).
    '''
    encoder_out, state = encoder.predict_on_batch(testX)
    target_seq = np.full((testX.shape[0], max_seq_len + 1), word2idx['<PAD>'], np.int32)
    target_seq[:, 0] = word2idx['<BOS>']  # add <BOS> in the beginning
    seq_eos = np.zeros(testX.shape[0], np.bool)
    for i in range(max_seq_len):
        decoder_out, state = decoder.predict_on_batch([target_seq[:, i:i + 1], encoder_out, state])
        #TODO beam search
        target_seq[:, i+1] = np.argmax(decoder_out[:, 0], axis=-1)
        seq_eos[target_seq[:, i+1] == word2idx['<EOS>']] = True
        if np.all(seq_eos):
            break
    else:
        target_seq[np.logical_not(seq_eos), -1] = word2idx['<EOS>']
    return target_seq[:, 1:] # does not return <BOS>

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, type=str, help='testing file and predicted file')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('-g', '--gpu', type=str, default='3')
    parser.add_argument('-t', '--teacher-forcing-ratio', type=float, default=1.0)
    parser.add_argument('-b', '--beam-search', type=int, default=False, help='Number of branches to do beam search. Disabled by default.')
    parser.add_argument('-a', '--attention', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    data_dir = args.data_dir
    model_path = args.model_path
    training = not args.no_training
    function = args.model_function
    test = args.test
    ensemble = args.ensemble
    teacher_forcing_ratio = args.teacher_forcing_ratio
    beam_search_enabled = args.beam_search
    attention_enabled = args.attention

    max_seq_len = 32 + 2  # contain <BOS> and <EOS>
    word2idx_en = utils.get_word2idx(os.path.join(data_dir, 'word2int_en.json'))
    word2idx_cn = utils.get_word2idx(os.path.join(data_dir, 'word2int_cn.json'))
    idx2word_en = utils.get_idx2word(os.path.join(data_dir, 'int2word_en.json'))
    idx2word_cn = utils.get_idx2word(os.path.join(data_dir, 'int2word_cn.json'))
    vocabulary_size_en, vocabulary_size_cn = len(word2idx_en), len(word2idx_cn)
    print(f'\033[32;1men vocabulary_size: {vocabulary_size_en}, cn vocabulary_size: {vocabulary_size_cn}\033[0m')

    hidden_dim = 128
    model, encoder, decoder= build_model(hidden_dim, max_seq_len, vocabulary_size_en)
    model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.summary()

    if training:
        trainX, validX, trainY_SOS, validY_SOS, trainY, validY, line_len, valid_line_len = train_data_preprocessing(inputX, word2idx, max_seq_len)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}, trainY_SOS: {trainY_SOS.shape}, validY_SOS: {validY_SOS.shape}\033[0m')
        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 3, verbose=1, min_lr=1e-6)
        logger = CSVLogger(model_path+'.csv', append=True)
        tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, write_images=True, update_freq=512)
        epochs = 10
        for epoch in range(epochs):
            print(f'\033[32;1mepoch: {epoch+1}/{epochs}\033[0m')
            ith, ith_str, word = generate_word(trainY, line_len)
            valid_ith, valid_ith_str, valid_word = generate_word(validY, valid_line_len)
           
            model.fit([trainX, trainY_SOS, ith, ith_str, word], [trainY, word], validation_data=([validX, validY_SOS, valid_ith, valid_ith_str, valid_word], [validY, valid_word]), batch_size=256, epochs=1, callbacks=[checkpoint, reduce_lr, logger, tensorboard])

        ith, ith_str, word = generate_word(trainY, line_len)
        valid_ith, valid_ith_str, valid_word = generate_word(validY, valid_line_len)

        print(f'Training score: {model.evaluate([trainX, trainY_SOS, ith, ith_str, word], [trainY, word], batch_size=256, verbose=0)}')
        print(f'Validaiton score: {model.evaluate([validX, validY_SOS, valid_ith, valid_ith_str, valid_word], [validY, valid_word], batch_size=256, verbose=0)}')
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = utils.load_data(test[0], word2idx, max_seq_len, label=False)
        pred = model.predict(testX, batch_size=256)
        if ensemble:
            np.save(test[1], pred)
        else:
            utils.generate_csv(pred, test[1])
    else:
        if not training:
            trainX, trainY = utils.load_train_data(labeled_path, word2idx, max_seq_len)
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
        print(f'\033[32;1mTraining score: {model.evaluate(trainX, trainY, batch_size=256, verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=256, verbose=0)}\033[0m')

    if submit:
        with open('output.txt', 'w') as f, open(submit, 'r') as t:
            input_testX = [line.split() for line in t.readlines()]
            testX = [[word2idx[w] if w in word2idx else word2idx[''] for w in ll[:-2]] for ll in input_testX]
            testX = pad_sequences(testX, max_seq_len, padding='post', truncating='post', value=word2idx[''])
            
            #the index of the word in testX starts from 1, which is different from our model
            test_ith = np.array([[int(ll[-2])-1] for ll in input_testX], dtype=np.int32)
            test_ith_str = np.array([[word2idx[str(i)]] for i in test_ith.ravel()], dtype=np.int32)
            test_word = np.array([[word2idx[ll[-1]] if ll[-1] in word2idx else word2idx['']] for ll in input_testX], dtype=np.int32)

            batch_size = 1024
            for i in trange(0, testX.shape[0], batch_size):
                decoder_seq = decode_sequence(encoder, decoder, testX[i:i+batch_size], test_ith[i:i+batch_size], test_ith_str[i:i+batch_size], test_word[i:i+batch_size], max_seq_len, word2idx)
                print(*[' '.join([idx2word[idx] for idx in ll]).strip() for ll in decoder_seq], sep='\n', file=f)
            os.system(f'python3 data/hw2.1_evaluate.py --training_file {submit} --result_file output.txt')

    if not training and not submit:
        trainX, validX, trainY_SOS, validY_SOS, trainY, validY, line_len, valid_line_len = train_data_preprocessing(inputX, word2idx, max_seq_len)
        ith, ith_str, word = generate_word(trainY, line_len)
        valid_ith, valid_ith_str, valid_word = generate_word(validY, valid_line_len)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}, trainY_SOS: {trainY_SOS.shape}, validY_SOS: {validY_SOS.shape}\033[0m')
        print(f'Training score: {model.evaluate([trainX, trainY_SOS, ith, ith_str, word], [trainY, word], batch_size=256, verbose=0)}')
        print(f'Validaiton score: {model.evaluate([validX, validY_SOS, valid_ith, valid_ith_str, valid_word], [validY, valid_word], batch_size=256, verbose=0)}')
