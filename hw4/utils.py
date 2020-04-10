import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def data_preprocessing(X, word2idx, max_seq_len):
    '''
    Convert word to index, and pad 'X' to 'max_seq_len'.

    Parameters:
        X: list of lists of words
    '''
    X = [[word2idx.get(w, word2idx['<UNK>']) for w in text] for text in X]
    return pad_sequences(X, max_seq_len, padding='post', truncating='post', value=word2idx['<PAD>'])

def load_train_data(path, word2idx, max_seq_len, label=True):
    if label:
        with open(path, 'r') as f:
            f_all = [ll.split('+++$+++') for ll in f.readlines()]
        trainX = [text_to_word_sequence(ll[1]) for ll in f_all]  # split the string to list of words
        trainX = data_preprocessing(trainX, word2idx, max_seq_len)
        trainY = np.array([int(ll[0]) for ll in f_all])
        return trainX, trainY
    else:
        with open(path, 'r') as f:
            trainX = [text_to_word_sequence(ll) for ll in f.readlines()]  # split the string to list of words
        trainX = data_preprocessing(trainX, word2idx, max_seq_len)
        return trainX
        
def load_test_data(path, word2idx, max_seq_len):
    with open(path, 'r') as f:
        f_all = [ll[ll.find(',') + 1:] for ll in f.readlines()[1:]]
    test = [text_to_word_sequence(ll) for ll in f_all]
    return data_preprocessing(test, word2idx, max_seq_len)  # split the string to list of words

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

def generate_csv(y_pred, output_file):
    Y = np.round(y_pred).astype(np.int)
    df = pd.DataFrame(list(zip(range(Y.shape[0]), Y.ravel())), columns=['id', 'label'])
    df.to_csv(output_file, index=False)

