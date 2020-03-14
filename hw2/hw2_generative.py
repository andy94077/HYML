import argparse
import sys, os
import numpy as np

import utils
from Adam import Adam

def sigmoid(s):
    return np.clip(1 / (1 + np.exp(-s)), 1e-8, 1 - 1e-8)

def f(X, w, b):
    return np.array(sigmoid(X * w + b))

def loss(X, Y, w, b):
    y_pred = f(X, w, b)
    return -np.mean(Y * np.log(y_pred + 1e-10) + (1 - Y) * np.log(1 - y_pred + 1e-10))

def accuracy(X, Y, w, b):
    return np.mean(np.round(f(X, w, b)).ravel() == Y.ravel())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file', nargs=2, help='trainX and trainY')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, help='testing file and the predicted file')
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.training_file
    training = not args.no_training
    test = args.test

    if training:
        trainX, trainY, _, _ = utils.load_train_data(train_file[0], train_file[1], normalize=False)
        trainX = np.matrix(trainX[:, 1:])  # remove bias coefficient
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')
        mu0, mu1 = np.mean(trainX[(trainY == 0).ravel()], axis=0).T, np.mean(trainX[(trainY == 1).ravel()], axis=0).T
        cov = np.matrix(np.sum(trainY) / trainY.shape[0] * np.cov(trainX[(trainY == 0).ravel()].T) + (1 - np.sum(trainY) / trainY.shape[0]) * np.cov(trainX[(trainY == 1).ravel()].T))

        u, s, v = np.linalg.svd(cov, full_matrices=False)
        cov_I = np.matmul(v.T * 1 / s, u.T)
        w = cov_I.T * (mu1 - mu0)
        b = -0.5 * mu1.T * cov_I * mu1 + 0.5 * mu0.T * cov_I * mu0 + np.log(np.sum(trainY)/ (trainY.shape[0] - np.sum(trainY)))
        np.save(model_path, [w, b])
    else:
        w, b = np.load(model_path, allow_pickle=True)

    if test:
        testX = utils.load_test_data(test[0])
        testX = np.matrix(testX[:, 1:])
        utils.generate_csv(f(testX, w, b), test[1])
    else:
        if not training:
            trainX, trainY, _, _ = utils.load_train_data(train_file[0], train_file[1], normalize=False)
            trainX = np.matrix(trainX[:, 1:])  # remove bias coefficient
        print(f'loss: {loss(trainX, trainY, w, b):.5}, acc: {accuracy(trainX, trainY, w, b):.4}')




