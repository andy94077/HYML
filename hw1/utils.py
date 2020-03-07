import numpy as np
import pandas as pd

def load_train_data(hr):
    train = pd.read_csv('train.csv', encoding = 'big5').iloc[:, 3:]
    train[train == 'NR'] = 0
    train = train.to_numpy().reshape(12, 20, 18, 24).astype(np.float32) # month, day, measurement, time
    train = np.transpose(train, [0, 2, 1, 3]).reshape(12, 18, -1) # month, measurement, day * time
    train[:, 9] = np.abs(train[:, 9])
    trainX = np.concatenate([m[:, i:i+hr].reshape(1, -1) for m in train for i in range(m.shape[1] - hr)], axis=0)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), (trainX - mean) / (std + 1e-10)], axis=1)
    trainY = train[:, 9, hr:].reshape(-1, 1)
    return trainX, trainY, mean, std

def load_test_data(path, mean, std):
    test = pd.read_csv(path, header=None, encoding = 'big5').iloc[:, 2:]
    test[test == 'NR'] = 0
    test = test.to_numpy().reshape(-1, 18, 9).astype(np.float32)
    test[:, 9] = np.abs(test[:, 9])
    test = test.reshape(-1, 18 * 9)
    test = np.concatenate([np.ones((test.shape[0], 1), np.float32), (test - mean) / (std + 1e-10)], axis=1)
    return test

def train_test_split(X, Y, split_ratio=0.2):
    np.random.seed(880301)
    idx = np.random.permutation(X.shape[0])
    return X[:-int(X.shape[0] * split_ratio)], X[-int(X.shape[0] * split_ratio):], Y[:-int(Y.shape[0] * split_ratio)], Y[-int(Y.shape[0] * split_ratio):]
