import os, sys, argparse
import importlib
import numpy as np
from sklearn.cluster import KMeans
from keras import backend as K
import tensorflow as tf

import utils
from autoencoder import build_autoencoder
from clustering import Clustering

def predict(encoder, clf, X, invert=False):
    latents = encoder.predict(X, batch_size=256).reshape(X.shape[0], -1)
    Y = clf.predict(latents)
    return 1 - Y if invert else Y

def evaluate(encoder, clf, X, Y):
    Y_pred = predict(encoder, clf, X)[-Y.shape[0]:]
    return np.max(np.mean([Y_pred == Y, (1 - Y_pred) == Y], axis=1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kmeans_model_path')
    parser.add_argument('autoencoder_model_path')
    parser.add_argument('-t', '--training-file', help='training data')
    parser.add_argument('-v', '--validation-file', nargs=2, help='validation data and validation labels')
    parser.add_argument('-f', '--model-function', default='build_autoencoder', help='build function of the autoencoder')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, type=str, help='predicted file')
    parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
    parser.add_argument('-i', '--invert', action='store_true')
    parser.add_argument('-g', '--gpu', default='3', help='available gpu device')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_path = args.kmeans_model_path
    autoencoder_path = args.autoencoder_model_path
    trainX_path = args.training_file
    validX_path, validY_path = args.validation_file if args.validation_file else [None, None]
    function = args.model_function
    training = not args.no_training
    test = args.test
    ensemble = args.ensemble
    invert = args.invert
    input_shape = (32, 32)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, autoencoder = globals()[function](input_shape + (3,))  # ignore the decoder model
    encoder.summary()
    autoencoder.load_weights(autoencoder_path)

    if training:
        trainX = utils.load_data(trainX_path, normalize=True, preprocessing=False)
        if validX_path is not None and validY_path is not None:
            validX, validY = utils.load_data(validX_path, validY_path, normalize=True, preprocessing=False)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        else:
            print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')

        latents = encoder.predict(trainX, batch_size=256).reshape(trainX.shape[0], -1)
        if validX_path is not None and validY_path is not None:
            latent_validX = encoder.predict(validX, batch_size=256).reshape(validX.shape[0], -1)
            latents = np.concatenate([latents, latent_validX], axis=0)
        model = Clustering().fit(latents)

        utils.save_model(model_path, model)
    else:
        print('\033[32;1mLoading Model\033[0m')
        model = utils.load_model(model_path)

    if test:
        testX = utils.load_data(test[0], normalize=True, preprocessing=False)
        pred = predict(encoder, model, testX, invert=invert)
        if ensemble:
            np.save(test, pred)
        else:
            utils.generate_csv(pred, test[1])
    else:
        if not training:
            trainX = utils.load_data(trainX_path, normalize=True, preprocessing=False)
            validX, validY = utils.load_data(validX_path, validY_path, normalize=True, preprocessing=False)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, validY: {validY.shape}\033[0m')
        X = np.concatenate([trainX, validX], axis=0)
        print(f'\033[32;1mValidaiton score: {evaluate(encoder, model, X, validY)}\033[0m')
        
