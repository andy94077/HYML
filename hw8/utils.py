import pickle
import json
import numpy as np
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

def get_word2idx(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_idx2word(path):
    with open(path, 'r') as f:
        idx2word = json.load(f)
    return {int(k): w for k, w in idx2word.items()}
    
def seq2sent(sequences, idx2word):
    '''Return sentences converted from sequences with the idx2word mapping.

    It will truncate the sentences after it meets '<EOS>'.

    Args:
        sequences: A list of sequence. i.e. [[i_0, i_1, ..., i_n, i('<EOS>'), i('<PAD>'), ...], ...].
        idx2word: A dict with indexs as key and words as values.
    '''
    sentences = []
    for sequence in sequences:
        sentence = []
        for i in sequence:
            word = idx2word[i]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences

def bleu_score(sentences, targets):
    '''Return the BELU@1 score.
    
    Args:
        sentences: A list of sentences. i.e. [[w_0, w_1, ..., w_n], ...]
        targets: A list of sentences, which has the same structure as `sentences`
    '''
    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp.extend([word for word in token])
        return tmp

    score = np.sum([sentence_bleu([cut_token(target)], cut_token(sentence), weights=(1, 0, 0, 0))
                    for sentence, target in zip(sentences, targets)])
    return score

def data_preprocessing(X, word2idx, max_seq_len, pad='default'):
    '''Convert word to index, and pad `X` to `max_seq_len`.

    Args:
        X: A list of lists of words.
        word2idx: A dict with words as keys and indexs as values.
        max_seq_len: The maximum sequence length.
        pad: The sequence length after padded. Default is the same as `max_seq_len`.
    Returns:
        A numpy array with converted and padded sequences.
    '''
    X = [[word2idx['<BOS>']] + [word2idx.get(w, word2idx['<UNK>']) for w in text[:max_seq_len - 2]] + [word2idx['<EOS>']] for text in X]
    return pad_sequences(X, max_seq_len if pad == 'default' else pad, padding='post', truncating='post', value=word2idx['<PAD>'])

def load_data(path, word2idx_X, max_seq_len, label=True, word2idx_Y=None):
    '''Return preprocessed training data from the file.

    Args:
        path: A file path of the corpus.
        word2idx_X: A dict with words as keys and indexs as values for training data.
        max_seq_len: The maximum sequence length.
        label: Whether the file contains labels.
        word2idx_Y: A dict with words as keys and indexs as values for training target.
            If None, it will be set as `word2idx_X`.
    
    Returns:
        If `label` is True, returns (`trainX`, `trainY[:, :-1]`, `trainY[:, 1:]`, `trainY_raw`), 
        where `trainX` is the training data sequences, 
        `trainY[:, :-1]` is the target data sequences having '<BOS>' in the beginning,
        `trainY[:, 1:]` is the target data sequences not having '<BOS>' in the beginning,
        and `trainY_raw` is the target data sentences.

        Otherwise, returns `trainX`.
    '''
    with open(path, 'r') as f:
        f_all = [ll.split('\t') for ll in f.readlines()] # [0]=en, [1]=cn
    # split the string to list of words
    trainX = [text_to_word_sequence(ll[0], filters='') for ll in f_all]
    trainX = data_preprocessing(trainX, word2idx_X, max_seq_len)
    if label:
        if word2idx_Y is None:
            word2idx_Y = word2idx_X
        # split the string to list of words
        trainY_raw = [text_to_word_sequence(ll[1], filters='') for ll in f_all]
        trainY = data_preprocessing(trainY_raw, word2idx_Y, max_seq_len, pad=max_seq_len + 1)
        # trainY[:, :-1] has <BOS> in the beginning, trainY[:, 1:] does not have <BOS> in the beginning
        # Both of trainY[:, :-1] and trainY[:, 1:] has length `max_seq_len`
        return trainX, trainY[:, :-1], trainY[:, 1:], trainY_raw
    return trainX
        
def train_test_split(X, Y, split_ratio=0.2, seed=880301):
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[0])
    return X[idx[:-int(X.shape[0] * split_ratio)]], X[idx[-int(X.shape[0] * split_ratio):]], Y[idx[:-int(Y.shape[0] * split_ratio)]], Y[idx[-int(Y.shape[0] * split_ratio):]]

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def generate_csv(sequences, idx2word, output_file):
    with open(output_file, 'w') as f:
        for sent in seq2sent(sequences, idx2word):
            print(' '.join(sent), file=f)

