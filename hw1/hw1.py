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
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, help='testing file and the predicted file')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. (default: %(default)s)')
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.training_file
    training = not args.no_training
    test = args.test
    lr = args.lr

    if training:
        trainX, trainY, mean, std = utils.load_train_data(train_file, 9)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')
        np.save('mean.npy', mean)
        np.save('std.npy', std)

        optimizer = Adam(compute_gradient, lr)

        XTX = trainX.T @ trainX
        XTY = trainX.T @ trainY
        epochs = 40000
        w = np.random.normal(0, 0.05, size=(trainX.shape[1], 1)).astype(np.float32)
        for epoch in range(epochs):
            optimizer.update(XTX, XTY, w)
            if epoch % 100 == 0:
                print(f'epoch {epoch + 100:04}, loss: {rmse(trainX, trainY, w):.5}')
        a = w[1:].reshape(-1, 9)
        for i in a:
            print(('%.3f '*9) % tuple(i))
        np.save(model_path, w)
    else:
        w = np.load(model_path)
        mean, std = np.load('mean.npy'), np.load('std.npy')

    if test:
        testX = utils.load_test_data(test[0], mean, std)
        utils.generate_csv(testX @ w, test[1])
    else:
        if not training:
            trainX, trainY, mean, std = utils.load_train_data(train_file, 9)
        print(f'Training loss: {rmse(trainX, trainY, w)}')


