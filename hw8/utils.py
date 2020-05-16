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
    
def seq2sent(sequence, idx2word):
    '''Return sentence converted from sequence with the idx2word mapping.

    It will truncate the sentence after it meets '<EOS>'.

    Args:
        sequences: A sequence. i.e. [i_0, i_1, ..., i_n, i('<EOS>'), i('<PAD>'), ...].
        idx2word: A dict with indexs as key and words as values.
    '''
    sentence = []
    for i in sequence:
        word = idx2word[i]
        if word == '<EOS>':
            break
        sentence.append(word)
    
    return sentence

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

def data_preprocessing(X, word2idx, max_seq_len, with_BOS=True):
    '''Convert word to index, and pad `X` to `max_seq_len`.

    Args:
        X: A list of lists of words.
        word2idx: A dict with words as keys and indexs as values.
        max_seq_len: The maximum sequence length.
        with_BOS: Whether the returned array contains <BOS> in the beginning.
    
    Returns:
        A numpy array with converted and padded sequences.
    '''
    X = [([word2idx['<BOS>']] if with_BOS else []) + [word2idx.get(w, word2idx['<UNK>']) for w in text] + [word2idx['<EOS>']] for text in X]
    return pad_sequences(X, max_seq_len, padding='post', truncating='post', value=word2idx['<PAD>'])

def load_data(path, word2idx, max_seq_len, label=True):
    '''Return preprocessed training data from the file.

    Args:
        path: A file path of the corpus.
        word2idx: A dict with words as keys and indexs as values.
        max_seq_len: The maximum sequence length.
        label: Whether the file contains labels.
    
    Returns:
        If `label` is True, returns (`trainX`, `trainY`, `trainY_raw`), 
        where `trainX` is the training data sequences, `trainY` is the target data sequences,
        and `trainY_raw` is the target data sentences.

        Otherwise, returns `trainX`.
    '''
    with open(path, 'r') as f:
        f_all = [ll.split('\t') for ll in f.readlines()] # [0]=en, [1]=cn
    # split the string to list of words
    trainX = [text_to_word_sequence(ll[0], filters='') for ll in f_all]
    trainX = data_preprocessing(trainX, word2idx, max_seq_len)
    if label:
        # split the string to list of words
        trainY_raw = [text_to_word_sequence(ll[1], filters='') for ll in f_all]
        trainY = data_preprocessing(trainY_raw, word2idx, max_seq_len, with_BOS=False)
        return trainX, trainY, trainY_raw
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
        for seq in sequences:
            print(' '.join(seq2sent(seq, idx2word)), file=f)

