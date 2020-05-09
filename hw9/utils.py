import os
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm

def data_preprocessing(X):
    return X ** (1 / 1.5)

def load_data(X_path, Y_path=None, normalize=True, preprocessing=False):
    trainX = np.load(X_path)
    trainX = trainX.astype(np.float32)
    if normalize:
        trainX -= 127.5
        trainX /= 128
    if preprocessing:
        trainX = data_preprocessing(trainX)
    
    if Y_path is not None:
        trainY = np.load(Y_path)
        return trainX, trainY
    return trainX

def train_test_split(X, Y, split_ratio=0.2, seed=880301):
    np.random.seed(seed)
    split_idx = [np.random.permutation(np.arange(Y.shape[0])[Y[:, i] == 1]) for i in range(Y.shape[1])]
    train_idx = np.concatenate([idx[:-int(idx.shape[0]*split_ratio)] for idx in split_idx], axis=0)
    valid_idx = np.concatenate([idx[-int(idx.shape[0]*split_ratio):] for idx in split_idx], axis=0)
    return X[train_idx], X[valid_idx], Y[train_idx], Y[valid_idx]

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def generate_csv(y_pred, output_file):
    Y = np.round(y_pred).astype(np.int)
    df = pd.DataFrame(list(zip(range(Y.shape[0]), Y.ravel())), columns=['id', 'label'])
    df.to_csv(output_file, index=False)

