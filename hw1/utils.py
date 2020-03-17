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

def load_train_data(path, hr, preprocessing=True):
    train = pd.read_csv(path, encoding = 'big5').iloc[:, 3:]
    train[train == 'NR'] = 0
    train = train.to_numpy().reshape(12, 20, 18, 24).astype(np.float32) # month, day, measurement, time
    train = np.transpose(train, [0, 2, 1, 3]).reshape(12, 18, -1) # month, measurement, day * time
    if preprocessing:
        data_preprocessing(train)
    trainX = np.concatenate([m[:, i:i+hr].reshape(1, -1) for m in train for i in range(m.shape[1] - hr)], axis=0)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), (trainX - mean) / (std + 1e-10)], axis=1)
    trainY = train[:, 9, hr:].reshape(-1, 1)
    return trainX, trainY, mean, std

def load_train_data_sin(path, hr):
    train = pd.read_csv(path, encoding = 'big5').iloc[:, 3:]
    train[train == 'NR'] = 0
    train = train.to_numpy().reshape(12, 20, 18, 24).astype(np.float32) # month, day, measurement, time
    train = np.transpose(train, [0, 2, 1, 3]).reshape(12, 18, -1) # month, measurement, day * time
    data_preprocessing(train)

    trainY = train[:, 9, hr:].reshape(-1, 1)

    t14, t15 = train[:, 14:15].copy(), train[:, 15:16].copy()
    train[:, 14] = np.sin(t14[:, 0] * np.pi / 180)
    train[:, 15] = np.sin(t15[:, 0] * np.pi / 180)
    train = np.concatenate([train, np.cos(t14 * np.pi / 180), np.cos(t15 * np.pi / 180)], axis=1)
    
    trainX = np.concatenate([m[:, i:i+hr].reshape(1, -1) for m in train for i in range(m.shape[1] - hr)], axis=0)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), (trainX - mean) / (std + 1e-10)], axis=1)
    return trainX, trainY, mean, std

def load_test_data(path, mean, std, preprocessing=True):
    test = pd.read_csv(path, header=None, encoding = 'big5').iloc[:, 2:]
    test[test == 'NR'] = 0
    test = test.to_numpy().reshape(-1, 18, 9).astype(np.float32)
    if preprocessing:
        data_preprocessing(test)
    test = test.reshape(-1, 18 * 9)
    test = np.concatenate([np.ones((test.shape[0], 1), np.float32), (test - mean) / (std + 1e-10)], axis=1)
    return test

def load_test_data_sin(path, mean, std):
    test = pd.read_csv(path, header=None, encoding = 'big5').iloc[:, 2:]
    test[test == 'NR'] = 0
    test = test.to_numpy().reshape(-1, 18, 9).astype(np.float32)
    data_preprocessing(test)
    
    t14, t15 = test[:, 14:15].copy(), test[:, 15:16].copy()
    test[:, 14] = np.sin(t14[:, 0] * np.pi / 180)
    test[:, 15] = np.sin(t15[:, 0] * np.pi / 180)
    test = np.concatenate([test, np.cos(t14 * np.pi / 180), np.cos(t15 * np.pi / 180)], axis=1)
    
    test = test.reshape(-1, test.shape[-2] * 9)
    test = np.concatenate([np.ones((test.shape[0], 1), np.float32), (test - mean) / (std + 1e-10)], axis=1)
    return test

def train_test_split(X, Y, split_ratio=0.2):
    np.random.seed(880301)
    idx = np.random.permutation(X.shape[0])
    return X[:-int(X.shape[0] * split_ratio)], X[-int(X.shape[0] * split_ratio):], Y[:-int(Y.shape[0] * split_ratio)], Y[-int(Y.shape[0] * split_ratio):]

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def generate_csv(y_pred, output_file):
    Y = np.clip(np.round(y_pred), 0, np.inf)
    df = pd.DataFrame(list(zip([f'id_{i}' for i in range(Y.shape[0])], Y.ravel())), columns=['id', 'value'])
    df.to_csv(output_file, index=False)

