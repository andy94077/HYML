import os, sys, argparse
import importlib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

import utils
from clustering import GeneralClustering

def calc_dist(model, transformedX, pred):
    return np.sum((model.transforms[-1].cluster_centers_[pred] - transformedX) ** 2, axis=1)

def baseline_transform(seed):
    return [KMeans(n_clusters=10, n_jobs=-1, random_state=seed)]


def improved_transform(seed=int(1e9) + 7, n_clusters=10):
    return [PCA(2, whiten=True, random_state=seed),
            KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=seed)
    ]

def improved_transform2(seed=int(1e9) + 7, n_clusters=10):
    return [PCA(64, whiten=True, random_state=seed),
            KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=seed)
    ]

def improved_transform3(seed=int(1e9) + 7):
    return [KernelPCA(64, 'rbf', n_jobs=-1, random_state=seed),
            KMeans(n_clusters=10, n_jobs=-1, random_state=seed)
    ]

def improved_transform4(seed=int(1e9) + 7):
    return [PCA(64, whiten=True, random_state=seed),
            #TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=8, n_jobs=-1, random_state=seed)
    ]

def improved_transform5(seed=int(1e9) + 7):
    return [PCA(64, whiten=True, random_state=seed),
            TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=8, n_jobs=-1, random_state=seed)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_model_path')
    parser.add_argument('latents_file', nargs='+')
    parser.add_argument('-t', '--transform-function', default='improved_transform')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-k', '--n-clusters', type=int, help='number of clusters')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('--seed', type=int, default=int(1e9) + 7)
    args = parser.parse_args()

    model_path = args.clustering_model_path
    latents_path = args.latents_file
    transform_function = args.transform_function
    test = args.test
    n_clusters = args.n_clusters
    ensemble = args.ensemble
    seed = args.seed
    input_shape = (32, 32)

    latents = np.concatenate([np.load(path) for path in latents_path], axis=0)
    latents = latents.reshape(latents.shape[0], -1)
    print(f'\033[32;1mlatents: {latents.shape}\033[0m')
    
    np.random.seed(880301)
    if transform_function not in globals():
        globals()[transform_function] = getattr(importlib.import_module(transform_function[:transform_function.rfind('.')]), transform_function.split('.')[-1])
    if n_clusters is not None:
        model, transformedX, pred = GeneralClustering(globals()[transform_function](seed, n_clusters)).fit_transform(latents)
    else:
        model, transformedX, pred = GeneralClustering(globals()[transform_function](seed)).fit_transform(latents)

    utils.save_model(model_path, model)

    dist = calc_dist(model, transformedX, pred)
    if test:
        if ensemble:
            np.save(test, dist)
        else:
            utils.generate_csv(dist, test)
    else:
        print(f'\033[32;1mValidation score: {np.mean(dist)}\033[0m')
