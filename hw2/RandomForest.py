import os, sys, argparse
import numpy as np
from multiprocessing.pool import Pool

from DecisionTree import DecisionTree
import utils

def bagging(seed, X, Y, rate):
	np.random.seed(seed)
	choice = np.random.choice(np.arange(X.shape[0]), size=int(rate * X.shape[0]))
	return X[choice], Y[choice]

def generate_tree(tup):
	'''
	@parameters:
		tup: (X, Y, max_height)
	@return: DecisionTree: tree
	'''
	seed, X, Y, max_height = tup
	bagX, bagY = bagging(seed, X, Y, 0.6)
	
	return DecisionTree().fit(bagX, bagY, max_height)

def f(tup):
	t, X = tup
	return t.predict(X)

class RandomForest():
	def __init__(self, tree_n):
		self.tree_n = tree_n
		self.trees = []

	def fit(self, X, Y, max_height=-1):
		np.random.seed(880301)
		poo = Pool(os.cpu_count() // 2)
		self.trees = poo.map(generate_tree, [(seed, X, Y, max_height) for seed in np.random.randint(0, 2147483647, self.tree_n)])

		return self

	def predict(self, X):
		poo = Pool(os.cpu_count())
		return np.sign(np.sum(poo.map(f, [(t, X) for t in self.trees]), axis=0))

	def score(self, X, Y):
		return np.mean(self.predict(X).ravel() == Y.ravel())

