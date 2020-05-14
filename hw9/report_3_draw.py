import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf

import utils
from report_3 import build_autoencoder3
from clustering import GeneralClustering
from main import improved_transform4

parser = argparse.ArgumentParser()
parser.add_argument('csv_path')
parser.add_argument('model_path')
parser.add_argument('training_file', help='training data')
parser.add_argument('validation_file', help='validation data')
parser.add_argument('validation_label', help='validation label')
parser.add_argument('-g', '--gpu', default='3', help='available gpu device')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

csv_path = args.csv_path
model_path = args.model_path
trainX_path = args.training_file
validX_path = args.validation_file
validY_path = args.validation_label
input_shape = (32, 32)

df = pd.read_csv(csv_path)
trainX = utils.load_data(trainX_path, normalize=True)
validX, validY = utils.load_data(validX_path, validY_path, normalize=True)
X = np.concatenate([trainX, validX], axis=0)

fig, axs = plt.subplots(2, 1, figsize=(6, 6))
axs[0].plot(range(1, 100, 10), df['loss'][::10])
axs[0].set_xticks(range(1, 100, 10))
axs[0].set_xlabel('epoch')
axs[0].set_title('Reconstruction error (MSE)')

encoder, _, model = build_autoencoder3(input_shape + (3,))  # ignore the decoder model

acc = []
for i in range(1, 100, 10):
	model.load_weights(f'{model_path[:model_path.rfind(".h5")]}_epoch_{i}.h5')
	latents = encoder.predict(X, batch_size=256)
	latents = latents.reshape(latents.shape[0], -1)

	_, pred = GeneralClustering(improved_transform4()).fit_predict(latents)
	pred = pred[-validY.shape[0]:]
	acc.append(max([np.mean(pred == validY), np.mean((1 - pred) == validY)]))

axs[1].plot(range(1, 100, 10), acc)
axs[1].set_xticks(range(1, 100, 10))
axs[1].set_xlabel('epoch')
axs[1].set_title('Accuracy (val)')
plt.tight_layout()
fig.savefig('report_3.jpg', bbox_inches='tight')