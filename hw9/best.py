import os, sys, argparse
import importlib
import numpy as np
from sklearn.cluster import KMeans

import utils
from clustering import Clustering

def predict(clf, latents, invert=False):
    Y = clf.predict(latents)
    return 1 - Y if invert else Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_model_path')
    parser.add_argument('latents_file')
    parser.add_argument('-c', '--model-class', default='Clustering')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('-i', '--invert', action='store_true')
    args = parser.parse_args()

    model_path = args.kmeans_model_path
    latents_path = args.latents_file
    model_class = args.model_class
    training = not args.no_training
    test = args.test
    ensemble = args.ensemble
    invert = args.invert
    input_shape = (32, 32)

    latents = utils.load_data(latents_path, normalize=not use_latents, preprocessing=False)
    latents = latents.reshape(latents.shape[0], -1)
    if training:
        print(f'\033[32;1mlatents: {latents.shape}\033[0m')

        if model_class not in globals():
            globals()[model_class] = getattr(importlib.import_module(model_class[:model_class.rfind('.')]), model_class.split('.')[-1])
        model, pred = globals()[model_class]().fit_predict(latents)

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
