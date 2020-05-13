import os, sys, argparse
import importlib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

import utils
from clustering import Clustering, GeneralClustering

def predict(clf, latents, invert=False):
    Y = clf.predict(latents)
    return 1 - Y if invert else Y

def baseline_transform():
    return [KernelPCA(n_components=200, kernel='rbf', n_jobs=-1, random_state=0), 
            TSNE(n_components=2, n_jobs=-1, random_state=0), 
            MiniBatchKMeans(n_clusters=2, random_state=0)
    ]


def improved_transform():
    seed = int(1e9) + 7
    return [KernelPCA(1024, 'rbf', n_jobs=-1, random_state=seed),
            KernelPCA(64, 'rbf', n_jobs=-1, random_state=seed),
            TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=2, n_jobs=-1, random_state=seed)
    ]

def improved_transform2():
    seed = int(1e9) + 7
    return [KernelPCA(1024, 'rbf', n_jobs=-1, random_state=seed),
            PCA(64, whiten=True, random_state=seed),
            TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=2, n_jobs=-1, random_state=seed)
    ]

def improved_transform3():
    seed = int(1e9) + 7
    return [KernelPCA(64, 'rbf', n_jobs=-1, random_state=seed),
            TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=2, n_jobs=-1, random_state=seed)
    ]

def improved_transform4():
    seed = int(1e9) + 7
    return [PCA(64, whiten=True, random_state=seed),
            TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=2, n_jobs=-1, random_state=seed)
    ]

def improved_transform5():
    seed = int(1e9) + 7
    return [PCA(16, whiten=True, random_state=seed),
            TSNE(n_components=2, n_jobs=-1, random_state=seed),
            KMeans(n_clusters=2, n_jobs=-1, random_state=seed)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_model_path')
    parser.add_argument('latents_file', nargs='+')
    parser.add_argument('-y', '--labels-file')
    parser.add_argument('-t', '--transform-function', default='improved_transform')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('-i', '--invert', action='store_true')
    args = parser.parse_args()

    model_path = args.clustering_model_path
    latents_path = args.latents_file
    Y_path = args.labels_file
    transform_function = args.transform_function
    training = not args.no_training
    test = args.test
    ensemble = args.ensemble
    invert = args.invert
    input_shape = (32, 32)

    latents = np.concatenate([np.load(path) for path in latents_path], axis=0)
    latents = latents.reshape(latents.shape[0], -1)
    if training:
        print(f'\033[32;1mlatents: {latents.shape}\033[0m')

        if transform_function not in globals():
            globals()[transform_function] = getattr(importlib.import_module(transform_function[:transform_function.rfind('.')]), transform_function.split('.')[-1])
        model, pred = GeneralClustering(globals()[transform_function]()).fit_predict(latents)

        utils.save_model(model_path, model)
    else:
        print('\033[32;1mLoading Model\033[0m')
        model = utils.load_model(model_path)

    if test:
        if not training:
            pred = predict(model, latents, invert=invert)

        if ensemble:
            np.save(test, pred)
        else:
            utils.generate_csv(pred, test)
    else:
        if not training:
            pred = predict(model, latents, invert=invert)
        Y = np.load(Y_path)
        pred = pred[-Y.shape[0]:]
        print(f'\033[32;1mValidation score: {max(np.mean(pred == Y), np.mean((1 - pred) == Y))}\033[0m')
