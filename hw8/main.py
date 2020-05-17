import os, sys, argparse
import numpy as np
from tqdm import trange
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, Lambda, Concatenate, Bidirectional, Embedding, Activation, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
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


def build_model(hidden_dim, max_seq_len, vocabulary_size_en, vocabulary_size_cn, with_attention=False, teacher_forcing_ratio=1.0):
    def build_encoder(embedding_dim, hidden_dim, max_seq_len, vocabulary_size):
        encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_seq_len)(encoder_in)
        
        states1, states2 = [], []
        x = embedding
        for _ in range(3):
            x, state1, state2 = Bidirectional(GRU(hidden_dim, return_sequences=True, return_state=True), merge_mode='concat')(x)
            states1.append(state1[:, tf.newaxis])
            states2.append(state2[:, tf.newaxis])
        encoder_out = x
        states1 = Concatenate(axis=1)(states1)
        states2 = Concatenate(axis=1)(states2)
        return Model(encoder_in, [encoder_out, states1, states2], name='encoder')
    
    def build_attention(units, hidden_dim, max_seq_len):
        hidden_state = Input((hidden_dim,), name='attention_hidden_state')
        encoder_out = Input((max_seq_len, hidden_dim), name='attention_encoder_out')

        hidden_state_expand = Lambda(lambda x: K.expand_dims(x, 1))(hidden_state)
        score_1 = Dense(units, use_bias=False)(hidden_state_expand)
        score_2 = Dense(units, use_bias=False)(encoder_out)
        added = Activation('tanh')(score_1 + score_2)
        alpha = Dense(1, activation='softmax', use_bias=False)(added)
        print(alpha)
        context_vector = K.sum(alpha * encoder_out, axis=1, keepdims=True)
        print(context_vector)
        return Model([hidden_state, encoder_out], context_vector, name='attention')

    def build_decoder(embedding_dim, hidden_dim, max_seq_len, vocabulary_size, with_attention=False):
        decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
        decoder_states1_in = Input((3, hidden_dim), name='decoder_states1_in')
        decoder_states2_in = Input((3, hidden_dim), name='decoder_states2_in')
        encoder_out = Input((max_seq_len, hidden_dim*2), name='decoder_encoder_out')
        teacher_forcing_ratio = Input(tuple(), name='teacher_forcing_ratio')
        random = K.random_uniform((K.shape(decoder_in)[0], max_seq_len))
        print(random)

        attention = build_attention(128, hidden_dim*2, max_seq_len)
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_seq_len)
        grus = [Bidirectional(GRU(hidden_dim, return_sequences=True, return_state=True), merge_mode='concat') for _ in range(3)]
        dense = Dense(vocabulary_size, activation='softmax')

        inputs, states1, states2 = decoder_in[:, 0:1], [decoder_states1_in[:, i] for i in range(3)], [decoder_states2_in[:, i] for i in range(3)]
        outputs = []
        for i in range(max_seq_len):
            embed = embedding(inputs)

            if with_attention:
                states_concat = Concatenate()([states1[0], states2[0]])
                context_vector = attention([states_concat, encoder_out])
            else:
                context_vector = encoder_out[:, -1:]
            print(embed, context_vector)
            x = Concatenate()([embed, context_vector])

            for j in range(len(grus)):
                x, states1[j], states2[j] = grus[j](x, initial_state = [states1[j], states2[j]])

            out = dense(x)
            outputs.append(out)
            out_argmax = tf.cast(K.argmax(out), tf.int32)
            if i != max_seq_len - 1:
                inputs = tf.where(random[:, i] < teacher_forcing_ratio, decoder_in[:, i+1:i+2], out_argmax)

        decoder_out = Concatenate(axis=1)(outputs)
        decoder_train = Model([decoder_in, decoder_states1_in, decoder_states2_in, encoder_out, teacher_forcing_ratio], decoder_out, name='decoder_train')
        
        ## inference model
        decoder_in_one_word = Input((1,), dtype='int32', name='decoder_in_one_word')
        embed = embedding(decoder_in_one_word)
        if with_attention:
            states_concat = Concatenate()([decoder_states1_in[:, 0], decoder_states2_in[:, 0]])
            context_vector = attention([states_concat, encoder_out])
        else:
            context_vector = encoder_out[:, -1:]
        x = Concatenate()([embed, context_vector])

        states1, states2 = [], []
        for j in range(len(grus)):
            x, state1, state2 = grus[j](x, initial_state=[decoder_states1_in[:, j], decoder_states2_in[:, j]])
            states1.append(state1)
            states2.append(state2)
        decoder_out = dense(x)
        states1 = Concatenate(axis=1)(states1)
        states2 = Concatenate(axis=1)(states2)
        
        decoder_infer = Model([decoder_in_one_word, decoder_states1_in, decoder_states2_in, encoder_out],
                                [decoder_out, states1, states2], name='decoder_infer')
        return decoder_train, decoder_infer

    encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
    encoder = build_encoder(256, hidden_dim, max_seq_len, vocabulary_size_en)
    encoder_out, states1, states2 = encoder(encoder_in)
    
    decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
    teacher_forcing_ratio = Input(tuple(), dtype='float32', name='teacher_forcing_ratio')
    decoder_train, decoder_infer = build_decoder(256, hidden_dim, max_seq_len,
        vocabulary_size_cn, with_attention=with_attention)
    decoder_out = decoder_train([decoder_in, states1, states2, encoder_out, teacher_forcing_ratio])
    
    model = Model([encoder_in, decoder_in, teacher_forcing_ratio], decoder_out, name='seq2seq')
    return model, encoder, decoder_infer

    '''
    ## encoder Input and layers
    ith_str = Input((1,), dtype='int32', name='ith_str')
    word = Input((1,), dtype='int32', name='word')
    OneHot = Lambda(lambda x: K.one_hot(x, vocabulary_size_en), name='OneHot')

    ## building encoder
    encoder_in_and_word = Concatenate()([ith_str, word, encoder_in])
    encoder_out, state = GRU(hidden_dim, return_state=True)(OneHot(encoder_in_and_word))
    encoder_out_dup = RepeatVector(max_seq_len)(encoder_out)

    ## decoder Input and layers
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
    x = Concatenate()([K.cast(ith, 'float')[:, tf.newaxis], OneHot(word), OneHot(decoder_in_one_word), encoder_out[:, tf.newaxis]])
    x, decoder_state = decoder_GRU(x, initial_state=decoder_state_in)
    decoder_out = decoder_Dense(x)
    decoder = Model([decoder_in_one_word, encoder_out, decoder_state_in, ith, word], [decoder_out, decoder_state])
    return model, encoder, decoder
    '''


def decode_sequence(encoder, decoder, X, max_seq_len, word2idx, beam_search=False, return_distribution=False):
    '''Return list of sequences outputed by the decoder given batched input sequences.
    
    Args:
        encoder: An encoder model.
        decoder: An decoder model.
        X: A batched list of sequences. i.e. [[w_0, w_1, ..., w_n], ...].
            The size of `X` should not be too large.
        max_seq_len: The maximum sequence length.
        word2idx: A dict with words as keys and indexs as values.
        beam_search: A bool represents whether beam search is enabled.
    
    Returns:
        A list of sequences padded to `max_seq_len`.
        Note that it does not contain i(<BOS>) in the beginning, bus it does contain i(<EOS>).
    '''
    encoder_out, states1, states2 = encoder.predict_on_batch(X)
    target_seq = np.full((X.shape[0], max_seq_len + 1), word2idx['<PAD>'], np.int32)
    target_seq[:, 0] = word2idx['<BOS>']  # add <BOS> in the beginning
    seq_eos = np.zeros(X.shape[0], np.bool)
    for i in range(max_seq_len):
        decoder_out, states1, states2 = decoder.predict_on_batch([target_seq[:, i:i + 1], states1, states2, encoder_out])
        #TODO beam search
        target_seq[:, i+1] = np.argmax(decoder_out[:, 0], axis=-1)
        seq_eos[target_seq[:, i+1] == word2idx['<EOS>']] = True
        if np.all(seq_eos):
            break
    else:
        target_seq[np.logical_not(seq_eos), -1] = word2idx['<EOS>']
    return to_categorical(target_seq[:, 1:], len(word2idx)) if return_distribution else target_seq[:, 1:] # does not return <BOS>

def predict(encoder, decoder, X, word2idx, beam_search=False, batch_size=256, return_distribution=False):
    sequences = []
    for i in trange(0, X.shape[0], batch_size):
        decoder_seq = decode_sequence(encoder, decoder, X[i:i+batch_size], max_seq_len, word2idx, beam_search=beam_search)
        sequences.extend(decoder_seq)
    return np.array(sequences)

def evaluate(model, encoder, decoder, X, Y, Y_raw, word2idx, idx2word, beam_search=False, batch_size=256):
    sequences = predict(encoder, decoder, X, word2idx, beam_search=beam_search, batch_size=batch_size)
    sentences = utils.seq2sent(sequences, idx2word)
    return K.sparse_categorical_crossentropy(Y, sentences), utils.bleu_score(sentences, Y_raw)
    return model.evaluate([X, np.zeros_like(X), np.zeros(X.shape[0])], Y, verbose=0), utils.bleu_score(sentences, Y_raw)

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

    max_seq_len = 2#32 + 2  # contain <BOS> and <EOS>
    word2idx_en = utils.get_word2idx(os.path.join(data_dir, 'word2int_en.json'))
    word2idx_cn = utils.get_word2idx(os.path.join(data_dir, 'word2int_cn.json'))
    idx2word_en = utils.get_idx2word(os.path.join(data_dir, 'int2word_en.json'))
    idx2word_cn = utils.get_idx2word(os.path.join(data_dir, 'int2word_cn.json'))
    vocabulary_size_en, vocabulary_size_cn = len(word2idx_en), len(word2idx_cn)
    print(f'\033[32;1men vocabulary_size: {vocabulary_size_en}, cn vocabulary_size: {vocabulary_size_cn}\033[0m')

    hidden_dim = 128
    model, encoder, decoder = build_model(hidden_dim, max_seq_len, vocabulary_size_en,
        vocabulary_size_cn, with_attention=attention_enabled)
    
    model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy')
    model.summary()
    
    if training:
        trainX, trainY_decoder_in, trainY, trainY_raw = utils.load_data(os.path.join(data_dir, 'training.txt'), word2idx_en, max_seq_len, label=True, word2idx_Y=word2idx_cn)
        validX, validY_decoder_in, validY, validY_raw = utils.load_data(os.path.join(data_dir, 'validation.txt'), word2idx_en, max_seq_len, label=True, word2idx_Y=word2idx_cn)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
        
        train_teacher_forcing_ratio = np.full(trainX.shape[0], teacher_forcing_ratio)
        valid_teacher_forcing_ratio = np.full(validX.shape[0], teacher_forcing_ratio)

        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 3, verbose=1, min_lr=1e-6)
        logger = CSVLogger(model_path+'.csv')
        tensorboard = TensorBoard(os.path.join('logs', os.path.basename(model_path[:model_path.rfind('.')])), write_grads=True, update_freq='epoch')
        model.fit([trainX, trainY_decoder_in, train_teacher_forcing_ratio], trainY, validation_data=([validX, validY_decoder_in, valid_teacher_forcing_ratio], validY), batch_size=256, epochs=10, callbacks=[checkpoint, reduce_lr, logger, tensorboard])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = utils.load_data(test[0], word2idx_en, max_seq_len, label=False)
        pred = predict(encoder, decoder, testX, word2idx_cn, beam_search=beam_search_enabled)
        if ensemble:
            np.save(test[1], pred)
        else:
            utils.generate_csv(pred, idx2word_cn, test[1])
    else:
        if not training:
            trainX, trainY_decoder_in, trainY, trainY_raw = utils.load_data(os.path.join(data_dir, 'training.txt'), word2idx_en, max_seq_len, label=True, word2idx_Y=word2idx_cn)
            validX, validY_decoder_in, validY, validY_raw = utils.load_data(os.path.join(data_dir, 'validation.txt'), word2idx_en, max_seq_len, label=True, word2idx_Y=word2idx_cn)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

        print(f'\033[32;1mTraining score: {evaluate(model, encoder, decoder, trainX, trainY, trainY_raw, word2idx_cn, idx2word_cn, beam_search=beam_search_enabled)}\033[0m')
        print(f'\033[32;1mValidaiton score: {evaluate(model, encoder, decoder, validX, validY, validY_raw, word2idx_cn, idx2word_cn, beam_search=beam_search_enabled)}\033[0m')
    '''
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
    '''