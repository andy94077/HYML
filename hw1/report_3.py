import argparse
import os, sys
import numpy as np
import pandas as pd

import utils
from Adam import Adam

def load_train_data_only_pm25(path, hr):
    train = pd.read_csv(path, encoding = 'big5').iloc[:, 3:]
    train[train == 'NR'] = 0
    train = train.to_numpy().reshape(12, 20, 18, 24).astype(np.float32) # month, day, measurement, time
    train = np.transpose(train, [0, 2, 1, 3]).reshape(12, 18, -1) # month, measurement, day * time
    utils.data_preprocessing(train)
    trainX = np.concatenate([m[9, i:i+hr].reshape(1, -1) for m in train for i in range(m.shape[1] - hr)], axis=0)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), (trainX - mean) / (std + 1e-10)], axis=1)
    trainY = train[:, 9, hr:].reshape(-1, 1)
    return trainX, trainY, mean, std

def load_test_data_only_pm25(path, mean, std):
    test = pd.read_csv(path, header=None, encoding = 'big5').iloc[:, 2:]
    test[test == 'NR'] = 0
    test = test.to_numpy().reshape(-1, 18, 9).astype(np.float32)
    utils.data_preprocessing(test)
    test = test[:, 9]
    print(test.shape)
    test = np.concatenate([np.ones((test.shape[0], 1), np.float32), (test - mean) / (std + 1e-10)], axis=1)
    return test

def rmse(X, Y, w):
    return np.sqrt(np.mean((X @ w - Y) ** 2))

def compute_gradient(XTX, XTY, w):
    return 2 * (XTX @ w - XTY)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-v', '--split-ratio', type=float, default=0.0, help='split ratio for validation set. (default: %(default)s)')
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.training_file
    training = not args.no_training
    split_ratio = args.split_ratio
    hr = 9
    lr = 1e-3

    if training:
        trainX, trainY, mean, std = load_train_data_only_pm25(train_file, hr)
        if split_ratio > 0:
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, 0.1)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')
        np.save(model_path[:model_path.rfind('.npy')] + '_mean.npy', mean)
        np.save(model_path[:model_path.rfind('.npy')] + '_std.npy', std)

        optimizer = Adam(compute_gradient, lr)

        XTX = trainX.T @ trainX
        XTY = trainX.T @ trainY
        epochs = 40000
        w = np.random.normal(0, 0.05, size=(trainX.shape[1], 1)).astype(np.float32)
        for epoch in range(epochs):
            optimizer.update(XTX, XTY, w)
            if epoch % 100 == 0:
                if split_ratio > 0:
                    print(f'epoch {epoch + 100:04}, loss: {rmse(trainX, trainY, w):.5}, valid_loss: {rmse(validX, validY, w):.5}')
                else:
                    print(f'epoch {epoch + 100:04}, loss: {rmse(trainX, trainY, w):.5}')
        a = w[1:].reshape(-1, hr)
        for i in a:
            print(('%.3f '*hr) % tuple(i))
        np.save(model_path, w)
    else:
        w = np.load(model_path)
        mean, std = np.load(model_path[:model_path.rfind('.npy')] + '_mean.npy'), np.load(model_path[:model_path.rfind('.npy')] + '_std.npy')

    if not training:
        trainX, trainY, mean, std = load_train_data_only_pm25(train_file, hr)
        if split_ratio > 0:
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, 0.1)
    print(f'Training loss: {rmse(trainX, trainY, w)}')
    if split_ratio > 0:
        print(f'Validation loss: {rmse(validX, validY, w)}')


