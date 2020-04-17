import os
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm
from keras.utils import to_categorical

def read_image(data_dir, input_shape, with_label=True):
    img_names = sorted(os.listdir(data_dir))
    X = np.array([cv2.resize(cv2.imread(os.path.join(data_dir, n)), input_shape) for n in tqdm(img_names)])
    if with_label:
        Y = np.array([int(n.split('_')[0]) for n in img_names])
        return X, Y
    return X


def data_preprocessing(X):
    X **= (1 / 1.5)

def load_train_data(data_dir, img_shape, normalize=True, preprocessing=False):
    if os.path.exists(os.path.join(data_dir, f'trainX{img_shape[0]}.npy')) and os.path.exists(os.path.join(data_dir, f'trainY{img_shape[0]}.npy')):
        trainX = np.load(os.path.join(data_dir, f'trainX{img_shape[0]}.npy'))
        trainY = np.load(os.path.join(data_dir, f'trainY{img_shape[0]}.npy'))
    else:
        trainX, trainY = read_image(os.path.join(data_dir, 'training'), img_shape)
        try:
            np.save(os.path.join(data_dir, f'trainX{img_shape[0]}.npy'), trainX)
            np.save(os.path.join(data_dir, f'trainY{img_shape[0]}.npy'), trainY)
        except:
            pass
    trainX = trainX.astype(np.float32)
    if normalize:
        trainX /= 255#(trainX - 127.5) / 128
    if preprocessing:
        data_preprocessing(trainX)
    trainY = to_categorical(trainY)
    return trainX, trainY

def load_test_data(data_dir, img_shape, normalize=True, preprocessing=False):
    test = read_image(os.path.join(data_dir, 'testing'), img_shape, with_label=False)
    test = test.astype(np.float32)
    if normalize:
        test /= 255#(test - 127.5) / 128
    if preprocessing:
        data_preprocessing(test)
    return test

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

