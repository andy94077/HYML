import os
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm
import torch

def load_train_filename_and_labels(data_dir):
    '''
    Combine images in 'data_dir/training' directory and 'data_dir/validation' directory.
    Args:
        data_dir: The base directory of the dataset. Must include 'training' and 'validation' sub-directory.
    Returns:
        (List of absolute path of images, List of labels of images)
    '''
    val_filename = os.listdir(os.path.join(data_dir, 'validation'))
    train_filename = sorted(os.listdir(os.path.join(data_dir, 'training')) + [name[:-3] + '_.jpg' for name in val_filename])
    filenames = [os.path.join(data_dir, 'training', n) if n[-5] != '_' else os.path.join(data_dir, 'validation', n[:-5] + 'jpg') for n in train_filename]
    labels = [int(os.path.basename(name).split('_')[0]) for name in filenames]
    return filenames, labels

def load_test_filename(data_dir):
    '''
    Args:
        data_dir: the base directory of the dataset. Must include 'testing' sub-directory.
    Returns:
        List of absolute path of images.
    '''
    filename = [os.path.join(data_dir, 'testing', name) for name in os.listdir(os.path.join(data_dir, 'testing'))]
    return filename

def train_test_split(X, Y, split_ratio=0.2, seed=880301):
    np.random.seed(seed)
    Y = np.array(Y)
    split_idx = [np.random.permutation(np.arange(Y.shape[0])[Y == i]) for i in range(np.max(Y))]
    train_idx = np.concatenate([idx[:-int(idx.shape[0]*split_ratio)] for idx in split_idx], axis=0)
    valid_idx = np.concatenate([idx[-int(idx.shape[0]*split_ratio):] for idx in split_idx], axis=0)
    return [X[i] for i in train_idx], [X[i] for i in valid_idx], Y[train_idx], Y[valid_idx]

def generate_csv(y_pred, output_file):
    Y = np.argmax(y_pred, axis=1).astype(np.int)
    df = pd.DataFrame(list(zip(range(Y.shape[0]), Y.ravel())), columns=['Id', 'Category'])
    df.to_csv(output_file, index=False)

