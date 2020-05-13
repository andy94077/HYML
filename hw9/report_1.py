import os, sys, argparse
import importlib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
from clustering import Clustering, GeneralClustering

def plot_result(embedding, file_path, Y=None, title=None):
	if Y is None:
		plt.scatter(embedding[:, 0], embedding[:, 1])
	else:
		for c in np.unique(Y):
			plt.scatter(embedding[Y == c, 0], embedding[Y == c, 1], alpha=0.7, label=str(c))
		plt.legend(fancybox=True)
	if title is not None:
		plt.title(title)
	plt.savefig(file_path, bbox_inches='tight')
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('latents_file', nargs='+')
	parser.add_argument('labels_file')
	parser.add_argument('output_image')
	parser.add_argument('-t', '--transform-function', default='improved_transform')
	parser.add_argument('--title')
	args = parser.parse_args()

	latents_path = args.latents_file
	Y_path = args.labels_file
	output_img = args.output_image
	transform_function = args.transform_function
	title = args.title

	latents = np.concatenate([np.load(path) for path in latents_path], axis=0)
	latents = latents.reshape(latents.shape[0], -1)
	print(f'\033[32;1mlatents: {latents.shape}\033[0m')

	if transform_function not in globals():
		globals()[transform_function] = getattr(importlib.import_module(transform_function[:transform_function.rfind('.')]), transform_function.split('.')[-1])
	_, embedding = GeneralClustering(globals()[transform_function]()).fit_transform(latents)

	Y = np.load(Y_path)
	embedding = embedding[-Y.shape[0]:]
	plot_result(embedding, output_img, Y, title)
