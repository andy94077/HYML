import os, sys, argparse
import numpy as np
from tqdm import trange
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, Lambda, Concatenate, Bidirectional, Embedding, Activation, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf

import utils

def build_model(embedding_dim, hidden_dim, max_seq_len, vocabulary_size_en, vocabulary_size_cn, n_gru_layers=3, with_attention=False, attention_dim=128, teacher_forcing_ratio=1.0):
    def build_encoder(embedding_dim, hidden_dim, max_seq_len, vocabulary_size, n_gru_layers):
        encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_seq_len)(encoder_in)
        
        states1, states2 = [], []
        x = embedding
        for _ in range(n_gru_layers):
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

        hidden_state_expand = hidden_state[:, tf.newaxis]
        score_1 = Dense(units, use_bias=False)(hidden_state_expand)
        score_2 = Dense(units, use_bias=False)(encoder_out)
        added = Activation('tanh')(score_1 + score_2)
        alpha = Dense(1, activation='softmax', use_bias=False)(added)
        
        context_vector = K.sum(alpha * encoder_out, axis=1, keepdims=True)
        
        return Model([hidden_state, encoder_out], context_vector, name='attention')

    def build_decoder(embedding_dim, hidden_dim, max_seq_len, vocabulary_size, n_gru_layers, with_attention, attention_dim):
        decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
        decoder_states1_in = Input((n_gru_layers, hidden_dim), name='decoder_states1_in')
        decoder_states2_in = Input((n_gru_layers, hidden_dim), name='decoder_states2_in')
        # attention should be calcuated with two concatenated states, so the dimension will be `hidden_dim` * 2
        encoder_out = Input((max_seq_len, hidden_dim * 2), name='decoder_encoder_out')
        teacher_forcing_ratio = Input(tuple(), name='teacher_forcing_ratio')
        random = K.random_uniform((K.shape(decoder_in)[0], max_seq_len))

        if with_attention:
            attention = build_attention(attention_dim, hidden_dim * 2, max_seq_len)
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_seq_len)
        grus = [Bidirectional(GRU(hidden_dim, return_sequences=True, return_state=True), merge_mode='concat') for _ in range(n_gru_layers)]
        dense = Dense(vocabulary_size, activation='softmax')

        inputs, states1, states2 = decoder_in[:, 0:1], [decoder_states1_in[:, i] for i in range(n_gru_layers)], [decoder_states2_in[:, i] for i in range(n_gru_layers)]
        outputs = []
        for i in trange(max_seq_len):
            embed = embedding(inputs)

            if with_attention:
                states_concat = Concatenate()([states1[0], states2[0]])
                context_vector = attention([states_concat, encoder_out])
            else:
                context_vector = encoder_out[:, -1:]
            
            x = Concatenate()([embed, context_vector])

            for j in range(n_gru_layers):
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
        for j in range(n_gru_layers):
            x, state1, state2 = grus[j](x, initial_state=[decoder_states1_in[:, j], decoder_states2_in[:, j]])
            states1.append(state1[:, tf.newaxis])
            states2.append(state2[:, tf.newaxis])
        decoder_out = dense(x)
        states1 = Concatenate(axis=1)(states1)
        states2 = Concatenate(axis=1)(states2)
        
        decoder_infer = Model([decoder_in_one_word, decoder_states1_in, decoder_states2_in, encoder_out],
                                [decoder_out, states1, states2], name='decoder_infer')
        return decoder_train, decoder_infer

    print('\033[32;1mBuilding encoder...\033[0m')
    encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
    encoder = build_encoder(embedding_dim, hidden_dim, max_seq_len, vocabulary_size_en, n_gru_layers)
    encoder_out, states1, states2 = encoder(encoder_in)
    
    print('\033[32;1mBuilding decoder...\033[0m')
    decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
    teacher_forcing_ratio = Input(tuple(), dtype='float32', name='teacher_forcing_ratio')
    decoder_train, decoder_infer = build_decoder(embedding_dim, hidden_dim, max_seq_len,
        vocabulary_size_cn, n_gru_layers, with_attention, attention_dim)
    decoder_out = decoder_train([decoder_in, states1, states2, encoder_out, teacher_forcing_ratio])
    
    model = Model([encoder_in, decoder_in, teacher_forcing_ratio], decoder_out, name='seq2seq')
    return model, encoder, decoder_infer


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
    if return_distribution:
        target_seq = np.full((X.shape[0], max_seq_len, len(word2idx)), word2idx['<PAD>'], np.float32)
    else:
        target_seq = np.full((X.shape[0], max_seq_len), word2idx['<PAD>'], np.int32)
    seq_eos = np.zeros(X.shape[0], np.bool)
    previous_word = np.full(X.shape[0], word2idx['<BOS>'], dtype=np.int32)  # add <BOS> in the beginning
    for i in range(max_seq_len):
        decoder_out, states1, states2 = decoder.predict_on_batch([previous_word[:, np.newaxis], states1, states2, encoder_out])
        #TODO beam search

        previous_word = np.argmax(decoder_out[:, 0], axis=-1)
        if return_distribution:
            target_seq[:, i] = decoder_out[:, 0]
        else:
            target_seq[:, i] = previous_word.copy()
        seq_eos[previous_word == word2idx['<EOS>']] = True
        if np.all(seq_eos):
            break
    else:
        target_seq[np.logical_not(seq_eos), -1] = word2idx['<EOS>']
    return target_seq

def predict(encoder, decoder, X, word2idx, beam_search=False, batch_size=1024, return_distribution=False):
    sequences = []
    for i in trange(0, X.shape[0], batch_size):
        decoder_seq = decode_sequence(encoder, decoder, X[i:i+batch_size], max_seq_len, word2idx, beam_search=beam_search, return_distribution=return_distribution)
        sequences.append(decoder_seq)
    return np.concatenate(sequences, axis=0)

def evaluate(sess, encoder, decoder, X, Y, Y_raw, word2idx, idx2word, beam_search=False, batch_size=1024):
    sequences_distribution = predict(encoder, decoder, X, word2idx, beam_search=beam_search, batch_size=batch_size, return_distribution=True)
    sentences = utils.seq2sent(np.argmax(sequences_distribution, axis=-1), idx2word)
    
    y_true = Input((max_seq_len,), dtype='int32', name='y_true')
    y_pred = Input((max_seq_len, len(word2idx)), name='y_pred')
    loss = K.sum(sparse_categorical_crossentropy(y_true, y_pred))
    loss_fn = K.function([y_true, y_pred], [loss])
    
    loss_value = np.sum([loss_fn([Y[i:i + batch_size], sequences_distribution[i:i + batch_size]])[0]
                            for i in trange(0, X.shape[0], batch_size)]) / X.shape[0]
    return loss_value, utils.bleu_score(sentences, Y_raw)

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

    max_seq_len = 20 + 2  # contain <BOS> and <EOS>
    word2idx_en = utils.get_word2idx(os.path.join(data_dir, 'word2int_en.json'))
    word2idx_cn = utils.get_word2idx(os.path.join(data_dir, 'word2int_cn.json'))
    idx2word_en = utils.get_idx2word(os.path.join(data_dir, 'int2word_en.json'))
    idx2word_cn = utils.get_idx2word(os.path.join(data_dir, 'int2word_cn.json'))
    vocabulary_size_en, vocabulary_size_cn = len(word2idx_en), len(word2idx_cn)
    print(f'\033[32;1men vocabulary_size: {vocabulary_size_en}, cn vocabulary_size: {vocabulary_size_cn}\033[0m')

    embedding_dim, hidden_dim = 256, 128
    model, encoder, decoder = build_model(embedding_dim, hidden_dim, max_seq_len, vocabulary_size_en,
        vocabulary_size_cn, with_attention=attention_enabled)
    
    model.summary()
    
    if training:
        trainX, trainY_decoder_in, trainY, trainY_raw = utils.load_data(os.path.join(data_dir, 'training.txt'), word2idx_en, max_seq_len, label=True, word2idx_Y=word2idx_cn)
        validX, validY_decoder_in, validY, validY_raw = utils.load_data(os.path.join(data_dir, 'validation.txt'), word2idx_en, max_seq_len, label=True, word2idx_Y=word2idx_cn)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
        
        train_teacher_forcing_ratio = np.full(trainX.shape[0], teacher_forcing_ratio)
        valid_teacher_forcing_ratio = np.full(validX.shape[0], teacher_forcing_ratio)

        model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy')
        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 3, verbose=1, min_lr=1e-6)
        logger = CSVLogger(model_path+'.csv')
        tensorboard = TensorBoard(os.path.join('logs', os.path.basename(model_path[:model_path.rfind('.')])), write_graph=False, update_freq='epoch')
        model.fit([trainX, trainY_decoder_in, train_teacher_forcing_ratio], trainY, validation_data=([validX, validY_decoder_in, valid_teacher_forcing_ratio], validY), batch_size=256, epochs=50, callbacks=[checkpoint, reduce_lr, logger, tensorboard])
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

        print(f'\033[32;1mTraining score: {evaluate(sess, encoder, decoder, trainX, trainY, trainY_raw, word2idx_cn, idx2word_cn, beam_search=beam_search_enabled)}\033[0m')
        print(f'\033[32;1mValidaiton score: {evaluate(sess, encoder, decoder, validX, validY, validY_raw, word2idx_cn, idx2word_cn, beam_search=beam_search_enabled)}\033[0m')
