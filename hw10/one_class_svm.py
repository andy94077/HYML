import os, sys, argparse
import importlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_model_path')
    parser.add_argument('latents_file', nargs='+')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('--seed', type=int, default=int(1e9) + 7)
    args = parser.parse_args()

    model_path = args.clustering_model_path
    latents_path = args.latents_file
    training = not args.no_training
    test = args.test
    ensemble = args.ensemble
    seed = args.seed
    input_shape = (32, 32)

    latents = np.concatenate([np.load(path) for path in latents_path], axis=0)
    latents = latents.reshape(latents.shape[0], -1)
    print(f'\033[32;1mlatents: {latents.shape}\033[0m')
    np.random.seed(880301)
    if training:
        model = OneClassSVM().fit(latents)
        utils.save_model(model_path, model)
    else:
        print('\033[32;1mLoading Model\033[0m')

    model = utils.load_model(model_path)
    if test:
        pred = model.score_samples(latents)

        if ensemble:
            np.save(test, pred)
        else:
            utils.generate_csv(pred, test)
    else:
        pred = model.score_samples(latents)
        print(f'\033[32;1mValidation score: {np.mean(pred)}\033[0m')
