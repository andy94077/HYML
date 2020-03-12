import argparse
import sys, os
import numpy as np

import utils
from Adam import Adam

def sigmoid(s):
    return np.clip(1 / (1 + np.exp(-s)), 1e-8, 1 - 1e-8)

def loss(X, Y, w):
    return np.mean(np.log(1 + np.exp(-(X @ w) * Y)))

def gradient(X, Y, w):
    return np.mean(sigmoid(-(X @ w) * Y) * (-Y) * X, axis=0).reshape(-1, 1)

def accuracy(X, Y, w):
    return np.mean(np.sign(X @ w) == Y.ravel())

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
        trainX, trainY, mean, std = utils.load_train_data(train_file[0], train_file[1], Y_to_sign=True)
        trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
        #print(validY[:100].ravel())
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        np.save(model_path[:model_path.rfind('.npy')] + '_mean.npy', mean)
        np.save(model_path[:model_path.rfind('.npy')] + '_std.npy', std)

        optimizer = Adam(gradient, 0.1)

        epochs = 400
        batch_size = 32
        w = np.random.normal(0, 0.05, size=(trainX.shape[1], 1)).astype(np.float32)
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(trainX.shape[0])
            shuffleX, shuffleY = trainX[idx], trainY[idx]
            for i in range(0, idx.shape[0], batch_size):
                batchX, batchY = shuffleX[i:i + batch_size], shuffleY[i:i + batch_size]
                optimizer.update(batchX, batchY, w)
                if i % (batch_size * 10) == 0:
                    print(f'epoch {epoch:04}, {i:05}/{trainX.shape[0]}, loss: {loss(batchX, batchY, w):.5}, acc: {accuracy(batchX, batchY, w):.4}', end='\r')
            print(f'epoch {epoch:04}, loss: {loss(trainX, trainY, w):.5}, acc: {accuracy(trainX, trainY, w):.4}, valid_loss: {loss(validX, validY, w):.5}, valid_acc: {accuracy(validX, validY, w):.4}')
        np.save(model_path, w)
    else:
        w = np.load(model_path)
        mean, std = np.load(model_path[:model_path.rfind('.npy')] + '_mean.npy'), np.load(model_path[:model_path.rfind('.npy')] + '_std.npy')

    if test:
        testX = utils.load_test_data(test[0], mean, std)
        utils.generate_csv(sigmoid(testX @ w), test[1])
    else:
        if not training:
            trainX, trainY, mean, std = utils.load_train_data(train_file, Y_to_sign=True)
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, 0.1)
        print(f'loss: {loss(trainX, trainY, w):.5}, acc: {accuracy(trainX, trainY, w):.4}, valid_loss: {loss(validX, validY, w):.5}, valid_acc: {accuracy(validX, validY, w):.4}')



