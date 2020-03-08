import os, sys, argparse
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import utils

def rmse(y_pred, Y):
    return np.sqrt(np.mean((y_pred.ravel() - Y.ravel()) ** 2))

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
        trainX, trainY, mean, std = utils.load_train_data_sin(train_file, 9, normalize=False)
        trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, 0.1)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        np.save('mean_best.npy', mean)
        np.save('std_best.npy', std)

        model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=64, max_depth=3, random_state=88031)#, n_iter_no_change=10, tol=1e-4)
        model.fit(trainX, trainY.ravel())
        utils.save_model(model_path, model)
        a = model.feature_importances_[1:].reshape(-1, 9)
        for i in a:
            print(('%.3f '*9) % tuple(i))
    else:
        model = utils.load_model(model_path)
        mean = np.load('mean_best.npy')
        std = np.load('std_best.npy')

    if test:
        testX = utils.load_test_data_sin(test[0], mean, std)
        utils.generate_csv(model.predict(testX), test[1])
    else:
        if not training:
            trainX, trainY, mean, std = utils.load_train_data(train_file, 9)
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
        print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        print(f'\033[32;1mTraining score: {rmse(model.predict(trainX), trainY)}\033[0m')
        print(f'\033[32;1mValidaiton score: {rmse(model.predict(validX), validY)}\033[0m')
        
