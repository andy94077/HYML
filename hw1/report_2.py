import argparse
import os, sys
import numpy as np
import pandas as pd

import utils
from Adam import Adam

def rmse(X, Y, w):
    return np.sqrt(np.mean((X @ w - Y) ** 2))

def compute_gradient(XTX, XTY, w):
    return 2 * (XTX @ w - XTY)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training-file')
    parser.add_argument('--hr', type=int, default=9, help='hours taken to predict. (default: %(default)s)')
    args = parser.parse_args()

    train_file = args.training_file
    lr = 1e-3
    hr = args.hr

    trainX, trainY, mean, std = utils.load_train_data(train_file, hr, preprocessing=False)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, 0.1)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')

    optimizer = Adam(compute_gradient, lr)

    XTX = trainX.T @ trainX
    XTY = trainX.T @ trainY
    epochs = 40000
    w = np.random.normal(0, 0.05, size=(trainX.shape[1], 1)).astype(np.float32)
    for epoch in range(epochs):
        optimizer.update(XTX, XTY, w)
        if epoch % 100 == 0:
            print(f'epoch {epoch + 100:04}, loss: {rmse(trainX, trainY, w):.5}, valid_loss: {rmse(validX, validY, w):.5}')

    a = w[1:].reshape(-1, hr)
    for i in a:
        print(('%.3f '*hr) % tuple(i))

    print(f'Training loss: {rmse(trainX, trainY, w)}')
    print(f'Validation loss: {rmse(validX, validY, w)}')


