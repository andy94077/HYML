import os
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm
from keras.utils import to_categorical

def read_image(data_dir, input_shape):
    img_names = sorted(os.listdir(data_dir))
    X = np.array([cv2.resize(cv2.imread(os.path.join(data_dir, n)), input_shape) for n in tqdm(img_names)])
    return X

def load_data(data_dir, img_shape, normalize=True):
    if os.path.exists(os.path.join(data_dir, f'trainX{img_shape[0]}.npy')):
        trainX = np.load(os.path.join(data_dir, f'trainX{img_shape[0]}.npy'))
    else:
        trainX = read_image(os.path.join(data_dir, 'training'), img_shape)
        np.save(os.path.join(data_dir, f'trainX{img_shape[0]}.npy'), trainX)
    trainX = trainX.astype(np.float32)
    if normalize:
        trainX = (trainX - 128) / 128
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
    Y = np.argmax(y_pred, axis=1).astype(np.int)
    df = pd.DataFrame(list(zip(range(Y.shape[0]), Y.ravel())), columns=['Id', 'Category'])
    df.to_csv(output_file, index=False)

