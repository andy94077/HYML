import argparse
import sys, os
import numpy as np

import utils

def get_g(X, Y, u):
	'''@return (tuple, float), (g, epsilon)'''
	min_g = (-1, 0, -np.inf)
	min_err_sum = np.sum(u)
	for i in range(X.shape[1]):
		Xi = np.unique(X[:, i])

		thetas = np.concatenate(([-np.inf], (Xi[:-1] + Xi[1:]) / 2))
		for s in [-1, 1]:
			err_sum_list = (g_func((s, i, thetas[:, np.newaxis]), X) != Y) @ u
			err_i = np.argmin(err_sum_list)
			err_sum = err_sum_list[err_i]
			if err_sum < min_err_sum:
				min_err_sum = err_sum
				min_g = (s, i, thetas[err_i])
	return min_g, min_err_sum / np.sum(u)

def g_func(tup,X):
	s, i, theta = tup
	return s * np.sign(X[:, i] - theta)

def G_func(G, X):
	return np.sign(np.sum(g_func((G[:, 0], G[:, 1].astype(int), G[:, 2]), X), axis=1)).ravel()

def accuracy(X, Y, G):
	return np.mean(G_func(G, X).ravel() == Y.ravel())

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path')
	parser.add_argument('-t', '--training-file', nargs=2, help='trainX and trainY')
	parser.add_argument('-T', '--no-training', action='store_true')
	parser.add_argument('-s', '--test', nargs=2, help='testing file and the predicted file')
	parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
	args = parser.parse_args()

	model_path = args.model_path
	train_file = args.training_file
	training = not args.no_training
	test = args.test
	ensemble = args.ensemble

	trainX, trainY, mean, std = utils.load_train_data(train_file[0], train_file[1])
	trainY = (trainY * 2 - 1).astype(np.int32).ravel()
	trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)

	if training:
		T = 300
		G = np.zeros((T, 3)) #[s, i, theta]
		u = np.full(trainX.shape[0], 1 / trainX.shape[0])
		for t in range(T):
			g, epsilon = get_g(trainX, trainY, u)

			G[t] = np.array([g[0] * np.log(1 / epsilon - 1) / 2, g[1], g[2]])
			
			wrong_i = g_func(g, trainX) != trainY
			correct_i = np.logical_not(wrong_i)
			u[wrong_i] *= np.sqrt(1 / epsilon - 1)
			u[correct_i] /= np.sqrt(1 / epsilon - 1)

			if t % 10 == 0:
				print(f't: {t+10:03}/{T:03}, acc: {accuracy(trainX, trainY, G):.4}, valid_acc: {accuracy(validX, validY, G):.4}')
		np.save(model_path, G)
	else:
		G = np.load(model_path)
	
	if test:
		testX = utils.load_test_data(test[0], mean, std)
		pred = (G_func(G, testX) + 1) / 2
		if ensemble:
			np.save(test[1], pred)
		else:
			utils.generate_csv(pred, test[1])
	else:
		print(f'acc: {accuracy(trainX, trainY, G):.4}, valid_acc: {accuracy(validX, validY, G):.4}')
