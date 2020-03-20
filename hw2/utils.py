import numpy as np
import pandas as pd
import pickle

def data_preprocessing(X):
    X = np.concatenate([X.to_numpy(), X['age'].to_numpy()[:, np.newaxis]**2], axis=1)
    #X = np.concatenate([X.to_numpy(), X['age'].to_numpy()[:, np.newaxis]**2, X['capital gains'].to_numpy()[:, np.newaxis] * X['capital losses'].to_numpy()[:, np.newaxis], X['dividends from stocks'].to_numpy()[:, np.newaxis]**2], axis=1)
    return X.astype(np.float32)

def load_train_data(trainX_path, trainY_path, normalize=True, preprocessing=False):
    trainX = pd.read_csv(trainX_path).iloc[:, 1:]#.to_numpy().astype(np.float32)
    if preprocessing:
        trainX = data_preprocessing(trainX)
    else:
        trainX = trainX.to_numpy().astype(np.float32)
    #trainX = np.concatenate([trainX, trainX[:, 0:1] ** 2], axis=1)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainY = pd.read_csv(trainY_path).iloc[:, 1:2].to_numpy().astype(np.float32)
    if normalize:
        trainX = (trainX - mean) / (std + 1e-10)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), trainX], axis=1)
    return trainX, trainY, mean, std

def load_test_data(path, mean=None, std=None, preprocessing=False):
    test = pd.read_csv(path).iloc[:, 1:]#.to_numpy().astype(np.float32)
    if preprocessing:
        test = data_preprocessing(test)
    else:
        test = test.to_numpy().astype(np.float32)
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

