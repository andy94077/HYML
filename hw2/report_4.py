import argparse
import sys, os
import numpy as np

import utils
from Adam import Adam

def sigmoid(s):
    return np.clip(1 / (1 + np.exp(-s)), 1e-8, 1 - 1e-8)

def f(X, w):
    return sigmoid(X @ w)

def loss(X, Y, w):
    y_pred = f(X, w)
    return -np.mean(Y * np.log(y_pred + 1e-10) + (1 - Y) * np.log(1 - y_pred + 1e-10))

def gradient(X, Y, w):
    return X.T @ (f(X, w) - Y) / X.shape[0]

def accuracy(X, Y, w):
    return np.mean((np.sign(X @ w).ravel() + 1) / 2 == Y.ravel())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training-file', nargs=2, help='trainX and trainY')
    args = parser.parse_args()

    train_file = args.training_file

    trainX, trainY, mean, std = utils.load_train_data(train_file[0], train_file[1], normalize=False)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
    optimizer = Adam(gradient, 1e-4)

    epochs = 150
    batch_size = 64
    w = np.zeros((trainX.shape[1], 1), np.float32)
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(trainX.shape[0])
        shuffleX, shuffleY = trainX[idx], trainY[idx]
        for i in range(0, idx.shape[0], batch_size):
            batchX, batchY = shuffleX[i:i + batch_size], shuffleY[i:i + batch_size]
            optimizer.update(batchX, batchY, w)

    print(f'loss: {loss(trainX, trainY, w):.5}, acc: {accuracy(trainX, trainY, w):.4}, valid_loss: {loss(validX, validY, w):.5}, valid_acc: {accuracy(validX, validY, w):.4}')



