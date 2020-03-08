import argparse
import os, sys
import numpy as np
import pandas as pd

import utils

def rmse(X, Y, w):
    return np.sqrt(np.mean(np.square(X * w - Y)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, help='testing file and the predicted file')
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.training_file
    training = not args.no_training
    test = args.test

    if training:
        trainX, trainY, mean, std = utils.load_train_data(train_file, 9)
        trainX, trainY = np.matrix(trainX), np.matrix(trainY)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')
        np.save('mean_best.npy', mean)
        np.save('std_best.npy', std)


        w = (trainX.T * trainX).I * (trainX.T * trainY)
        #a = np.array(w)[1:].reshape(-1, 9)
        #for i in a:
        #    print(('%.3f '*9) % tuple(i))
        #print(w.shape)
        np.save(model_path, w)
    else:
        w = np.load(model_path)
        mean, std = np.load('mean_best.npy'), np.load('std_best.npy')

    if test:
        testX = np.matrix(utils.load_test_data(test[0], mean, std))
        utils.generate_csv(np.array(testX * w), test[1])
    else:
        if not training:
            trainX, trainY, mean, std = utils.load_train_data(train_file, 9)
            trainX, trainY = np.matrix(trainX), np.matrix(trainY)
        print(f'Training loss: {rmse(trainX, trainY, w)}')

