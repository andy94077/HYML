import os, sys, argparse
import pickle
import numpy as np
from multiprocessing.pool import Pool

from RandomForest import RandomForest
import utils

if __name__ == "__main__":
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

	trainX, trainY, mean, std = utils.load_train_data(train_file[0], train_file[1])
	trainY = (trainY * 2 - 1).astype(np.int32).ravel()
	trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
	print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')

	if training:
		T = 32
		clf = RandomForest(T).fit(trainX, trainY, max_height=9)
		utils.save_model(model_path, clf)
	else:
		clf = utils.load_model(model_path)

	if test:
		testX = utils.load_test_data(test[0], mean, std)
		utils.generate_csv((model.predict(testX) + 1) / 2, test[1])
	else:
		print(f'\033[32;1mTraining score: {clf.score(trainX, trainY)}\033[0m')
		print(f'\033[32;1mValidaiton score: {clf.score(validX, validY)}\033[0m')
		
