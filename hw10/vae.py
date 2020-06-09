import os, argparse
import importlib
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Conv2DTranspose, GlobalAveragePooling2D, Reshape, Add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
import cv2

import utils

def calc_dist(model, X):
    pred = model.predict(X, batch_size=256)
    return np.mean((X.reshape(X.shape[0], -1) - pred.reshape(pred.shape[0], -1))** 2, axis=1)

def build_vae(input_shape):
    '''
    Arguments:
        input_shape(tuple): the shape of input images (H, W, C)
    
    Returns:
        encoder, decoder, autoencoder models
    '''
    def build_encoder(input_shape, latent_dim):
        inputs = Input(input_shape)
        x = Conv2D(256, 4, strides=2, padding='same', activation='relu')(inputs)
        x = Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(64, 4, strides=2, padding='same', activation='relu')(x)
        # x = Conv2D(64, 4, strides=2, padding='same', activation='relu')(x)
        x = Flatten()(x)
        mean = Dense(latent_dim)(x)
        logvar = Dense(latent_dim)(x)

        epsilon = K.random_normal(K.shape(mean))
        z = mean + K.exp(0.5 * logvar) * epsilon
        encoder = Model(inputs, [z, mean, logvar], name='encoder')
        return encoder

    def build_decoder(latent_dim):
        decoder = Sequential([
            Dense(4* 4* 64, activation='relu', input_shape=(latent_dim,)),
            Reshape((4, 4, 64)),
            # Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
        ], name='decoder')
        return decoder

    latent_dim = 512
    encoder = build_encoder(input_shape, latent_dim)
    # encoder.summary()
    decoder = build_decoder(latent_dim)
    # decoder.summary()

    inputs = Input(input_shape)
    z, mean, logvar = encoder(inputs)
    decoder_out = decoder(z)
    autoencoder = Model(inputs, decoder_out)

    bce_loss = K.sum(binary_crossentropy(inputs, decoder_out), axis=[1, 2])
    kl_loss = -0.5 * K.sum(1 + logvar - K.square(mean) - K.exp(logvar), axis=-1)
    vae_loss = K.mean(bce_loss + kl_loss)
    autoencoder.add_loss(vae_loss)

    autoencoder.add_metric(tf.reduce_mean(bce_loss), name='bce_sum', aggregation='mean')
    autoencoder.add_metric(tf.reduce_mean(bce_loss) / input_shape[0] / input_shape[1], name='bce', aggregation='mean')
    autoencoder.add_metric(tf.reduce_mean(kl_loss), name='KL', aggregation='mean')
    autoencoder.compile(Adam(1e-3, decay=5e-4))

    return encoder, decoder, autoencoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--training-file', help='Training data')
    parser.add_argument('-f', '--model-function', default='build_vae')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', nargs=2, type=str, help='Testing file and predicted file. If the file extension is .npy, output the latents file, otherwise output the csv.')
    parser.add_argument('-i', '--output-image', nargs=2, help='Output original image and reconstructed image')
    parser.add_argument('-g', '--gpu', default='3', help='Available gpu device')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_path = args.model_path
    trainX_path = args.training_file
    function = args.model_function
    training = not args.no_training
    test = args.test
    output_image = args.output_image
    input_shape = (32, 32)

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    encoder, _, model = globals()[function](input_shape + (3,))  # ignore the decoder model
    model.summary()

    if training:
        trainX = (np.load(trainX_path) + 1) / 2
        trainX, validX = utils.train_test_split(trainX, split_ratio=0.1)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')

        batch_size = 512
        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 10, verbose=1, min_lr=1e-5)
        #logger = CSVLogger(model_path+'.csv')
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, update_freq='epoch')
        model.fit(trainX, validation_data=(validX, None), batch_size=batch_size, epochs=100, verbose=2, callbacks=[checkpoint, reduce_lr])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        testX = (np.load(test[0]) + 1) / 2
        if test[1][-4:] == '.npy':
            pred = encoder.predict(testX, batch_size=512)
            np.save(test[1], pred[0] if isinstance(pred, (list, tuple)) else pred)
        else:
            pred = model.predict(testX)
            y_true = Input(testX.shape[1:])
            y_pred = Input(pred.shape[1:])
            func = K.function([y_true, y_pred], [K.sum(binary_crossentropy(y_true, y_pred), axis=[1, 2])])
            # utils.generate_csv(func([testX, pred])[0], test[1])
            utils.generate_csv(np.sum((pred - testX)** 2, axis=(1, 2, 3)), test[1])
    elif output_image:
        testX = (np.load(output_image[0]) + 1) / 2

        pred = model.predict(testX[:10])
        testX = np.concatenate(testX[:10] * 255, axis=1).astype(np.uint8)
        pred = np.concatenate(pred * 255, axis=1).astype(np.uint8)
        cv2.imwrite(output_image[1], np.concatenate([testX, pred], axis=0))
    else:
        if not training:
            trainX = (np.load(trainX_path) + 1) / 2
            trainX, validX = utils.train_test_split(trainX, split_ratio=0.1)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')
        print(f'\033[32;1mTraining loss: {model.evaluate(trainX, batch_size=512, verbose=0)}\033[0m')
        print(f'\033[32;1mValidation loss: {model.evaluate(validX, batch_size=512, verbose=0)}\033[0m')

