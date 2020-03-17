import os, sys, argparse
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import utils

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

    trainX, trainY, mean, std = utils.load_train_data(train_file[0], train_file[1], normalize=False)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, 0.1)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
    if training:
        model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=3, random_state=880301)#, n_iter_no_change=10, tol=1e-4)
        model.fit(trainX, trainY.ravel())
        utils.save_model(model_path, model)
        #a = model.feature_importances_[1:].reshape(-1, 9)
        #for i in a:
        #    print(('%.3f '*9) % tuple(i))
    else:
        model = utils.load_model(model_path)

    if test:
        testX = utils.load_test_data(test[0], mean, std)
        utils.generate_csv(model.predict(testX), test[1])
    else:
        print(f'\033[32;1mTraining score: {model.score(trainX, trainY)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.score(validX, validY)}\033[0m')
        
