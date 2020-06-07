import os
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm

def train_test_split(X, Y=None, split_ratio=0.2, seed=880301):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n = int(len(X) * split_ratio)
    return (X[idx[:-n]], X[idx[-n:]], Y[idx[:-n]], Y[idx[-n:]]) if Y is not None else X[idx[:-n]], X[idx[-n:]]

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def generate_csv(y_pred, output_file):
    Y = np.round(y_pred, 5)
    df = pd.DataFrame(list(zip(range(1, Y.shape[0] + 1), Y.ravel())), columns=['id', 'anomaly'])
    df.to_csv(output_file, index=False)

