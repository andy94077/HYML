import os, sys, argparse
import importlib
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, Flatten, Activation, GlobalAveragePooling2D, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
import cv2

import utils

def train(model_dir, gan, generator, discriminator, X, batch_size, epochs, training_ratio=3, generate_imgs_frequency=None, generate_csv=False, seed=880301):
    root = model_dir
    img_dir = os.path.join(model_dir, 'imgs')
    model_dir = os.path.join(root, 'models')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if generate_csv:
        csv = open(os.path.join(root, 'logs.csv'), 'w')
        print('epoch,step,G_loss,D_loss', file=csv)

    np.random.seed(seed)
    step = 1
    training_ratio_list = training_ratio
    if not isinstance(training_ratio, (list, tuple)):
        training_ratio_list = [training_ratio] * epochs
    for epoch, training_ratio in zip(range(epochs), training_ratio_list):
        # pad the size of X to make it be the multiple of the batch size
        idx = np.concatenate([np.random.permutation((X.shape[0] + batch_size - 1) // batch_size * batch_size) % X.shape[0] for _ in range(training_ratio)])
        discriminator_trained_times = 0
        for i in range(0, idx.shape[0], batch_size):
            realX = X[idx[i:i+batch_size]]
            noise = np.random.uniform(size=(realX.shape[0],) + gan.input.shape[1:])
            # print(noise.shape)
            fakeX = generator.predict_on_batch(noise)
            # print(fakeX.min(), fakeX.max(), end=' ')

            real_loss = discriminator.train_on_batch(realX, np.ones(realX.shape[0]))
            fake_loss = discriminator.train_on_batch(fakeX, np.zeros(fakeX.shape[0]))
            if discriminator_trained_times < training_ratio - 1:
                discriminator_trained_times += 1
                continue
            discriminator_trained_times = 0
            discriminator_loss = (real_loss + fake_loss) / 2

            noise = np.random.uniform(size=(realX.shape[0],) + gan.input.shape[1:])
            generator_loss = gan.train_on_batch(noise, np.ones(noise.shape[0]))
            print(f'[{min((i+batch_size) // training_ratio, X.shape[0])}/{X.shape[0]}], epoch: {epoch+1:0{len(str(epochs))}}/{epochs}, G_loss: {generator_loss:.5f}, D_loss: {discriminator_loss:.5f}', end='\r')
            if generate_imgs_frequency is not None and step % generate_imgs_frequency == 0:
                gan.save_weights(os.path.join(model_dir, f'{epoch+1:0{len(str(epochs))}}_{step:05}.h5'))
                generate_grid_img(os.path.join(img_dir, f'{epoch+1:0{len(str(epochs))}}_{step:05}.jpg'), generator)
                if generate_csv:
                    print(f'{epoch+1},{step},{generator_loss},{discriminator_loss}', file=csv)
            step += 1
        print('')

        if generate_imgs_frequency is None:
            gan.save_weights(os.path.join(model_dir, f'{epoch+1:0{len(str(epochs))}}.h5'))
            generate_grid_img(os.path.join(img_dir, f'{epoch+1:0{len(str(epochs))}}.jpg'), generator)
            if generate_csv:
                print(f'{epoch+1},{step},{generator_loss},{discriminator_loss}', file=csv)

def generate_grid_img(img_path, generator, grid_size=(2, 8), seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.uniform(size=(grid_size[0] * grid_size[1],) + generator.input.shape[1:])
    imgs = generator.predict_on_batch(noise)
    imgs = imgs.reshape(grid_size+imgs.shape[1:])
    grid = []
    for i in range(imgs.shape[0]):
        grid.append(np.concatenate(imgs[i], axis=1))
    grid = np.concatenate(grid, axis=0)
    cv2.imwrite(img_path, (grid * 128 + 128).astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_dir')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-r', '--seed', type=int, default=880301, help='random seed')
    parser.add_argument('-g', '--gpu', default='3', help='available gpu device')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    data_dir = args.data_dir
    model_dir = args.model_dir
    training = not args.no_training
    test = args.test
    function = args.model_function
    seed = args.seed
    input_shape = (64, 64)
    in_dim = 100

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model, generator, discriminator = globals()[function](input_shape + (3,), in_dim)
    discriminator.summary()
    model.summary()
    
    if training:
        trainX = utils.load_data(data_dir, input_shape)
        print(f'\033[32;1mtrainX: {trainX.shape}\033[0m')

        batch_size = 128
        train(model_dir, model, generator, discriminator, trainX, batch_size=batch_size, epochs=60, generate_imgs_frequency=200, training_ratio=[3] * 30 + [2] * 40 + [1] * 40, generate_csv=True, seed=seed)
    else:
        print('\033[32;1mLoading Model\033[0m')
        model.load_weights(model_dir)
    
    if test:
        generate_grid_img(test, generator, seed=seed)
