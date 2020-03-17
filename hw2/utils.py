import numpy as np
import pandas as pd
import pickle

def data_preprocessing(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[-1]):
            if X[i, 9, j] >= 0:
                continue
            if j == 0:
                X[i, 9, j] = X[i, 9, min(j + 1, X.shape[-1])]
            elif j == X.shape[-1] - 1:
                X[i, 9, j] = X[i, 9, max(j - 1, 0)]
            else:
                X[i, 9, j] = (X[i, 9, j - 1] + X[i, 9, j + 1]) / 2

def load_train_data(trainX_path, trainY_path, normalize=True):
    trainX = pd.read_csv(trainX_path).iloc[:, 1:].to_numpy().astype(np.float32)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainY = pd.read_csv(trainY_path).iloc[:, 1:2].to_numpy().astype(np.float32)
    if normalize:
        trainX = (trainX - mean) / (std + 1e-10)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), trainX], axis=1)
    return trainX, trainY, mean, std

def load_test_data(path, mean=None, std=None):
    test = pd.read_csv(path).iloc[:, 1:].to_numpy().astype(np.float32)
    if mean is not None and std is not None:
        test = (test - mean) / (std + 1e-10)
    test = np.concatenate([np.ones((test.shape[0], 1), np.float32), test], axis=1)
    return test

def train_test_split(X, Y, split_ratio=0.2, seed=880301):
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[0])
    return X[:-int(X.shape[0] * split_ratio)], X[-int(X.shape[0] * split_ratio):], Y[:-int(Y.shape[0] * split_ratio)], Y[-int(Y.shape[0] * split_ratio):]

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

