import os, sys, argparse
import importlib
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from lime import lime_image

import utils

def build_model(input_shape, output_dim):
    model = Sequential([
        Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, 3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(output_dim, activation='softmax')
        ])
    return model

def normalize(X):
    return (X - np.min(X, axis=(-1, -2, -3), keepdims=True)) / (np.max(X, axis=(-1, -2, -3), keepdims=True) - np.min(X, axis=(-1, -2, -3), keepdims=True) + 1e-8)

def plot_saliency_map(fig_name, model, X):
    K.set_learning_phase(0)
    X = X.copy()
    saliency_map = []
    Y = np.argmax(model.predict_on_batch(X), axis=-1)
    for i, x in enumerate(X):
        grad_tensor = K.gradients(model.output[:, Y[i]], model.input)[0]
        get_grad = K.function([model.input], [grad_tensor])

        grad = np.abs(get_grad([[x]])[0][0])
        saliency_map.append(normalize(grad))
    saliency_map = np.array(saliency_map)
    saliency_map = np.clip((saliency_map) * 255, 0, 255).astype(np.uint8)

    X **= 1.5  # restore gamma correction
    X = np.clip(X, 0, 255).astype(np.uint8)

    ## plot figures
    fig, axs = plt.subplots(2, X.shape[0], figsize=(15, 8))
    for row, target in enumerate([X, saliency_map]):
        for col, img in enumerate(target):
            axs[row][col].imshow(img[..., ::-1]) # BGR to RGB
    fig.suptitle('saliency map', fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return saliency_map

def plot_filter_activation(model, X, layer_name_list, filter_idx_list, train_iter=150):
    def get_max_activation_result(model, layer_output, filter_idx, iter_n):
        loss = K.mean(layer_output[:, :, :, filter_idx])
        grads = K.gradients(loss, model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iter_func = K.function([model.input], [loss, grads])
        
        max_activation_result = np.random.normal(size=(1,) + model.input.shape[1:]) * 25 + 128  # shape=(1, 128, 128, 3)
        for t in range(iter_n):
            print(f'{t + 1:0{len(str(iter_n))}}', end='\r')
            loss_value, grads_value = iter_func([max_activation_result])
            max_activation_result += 1 * grads_value

        return np.clip(max_activation_result[0], 0, 255).astype(np.uint8)

    K.set_learning_phase(0)
    model.trainable = False
    if not isinstance(train_iter, list):
        train_iter = [train_iter] * len(layer_name_list)
    for (layer, iter_n) in zip(layer_name_list, train_iter):
        ## initialization for subplots
        R, C = 1 + len(filter_idx_list), 2 + X.shape[0]
        fig, axs = plt.subplots(R, C, figsize=(R * 4, C * 4))

        ## remove the bounding box for the layer name
        for i in range(axs.shape[0]):
            axs[i, 0].axis('off')
        axs[0, 1].axis('off')

        # show original images and restore gamma correction
        for i, x in enumerate(X):
            axs[0, i + 2].imshow(np.clip(x[...,::-1]** 1.5, 0, 255).astype(np.uint8))  # BGR to RGB

        ## get the filter output for each image
        layer_output = model.get_layer(layer).output
        model2 = Model(model.input, layer_output)
        pred = model2.predict(X)
        pred = normalize(pred)

        for i, filter_idx in enumerate(filter_idx_list):
            max_activation_result = get_max_activation_result(model, layer_output, filter_idx, iter_n)

            axs[1 + i, 0].text(0.35, 0.35, f'{layer}\nfilter {filter_idx}', fontsize=24)
            axs[1 + i, 1].imshow(max_activation_result[...,::-1])  # BGR to RGB
            for j, y in enumerate(pred):
                axs[1 + i, 2 + j].imshow(y[:,:, filter_idx][...,::-1])  # BGR to RGB
        
        fig.suptitle(f'layer "{layer}" filter visualization', fontsize=28)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.savefig(f'{layer}.jpg', bbox_inches='tight')
    plt.close()

def plot_lime(fig_name, model, X, Y, label_name=None):
    def segmentation(X):
        return slic(X, n_segments=100, compactness=1, sigma=1)

    fig, axs = plt.subplots((X.shape[0] + 4) // 5, 5, figsize=(20, (X.shape[0] + 4) // 5 * 4))
    for i in range(X.shape[0], axs.shape[0] * axs.shape[1]):
        axs[i // 5, i % 5].axis('off')
    
    np.random.seed(880301)
    if label_name is None:
        label_name = range(X.shape[0])
    for i, (x, y, name) in enumerate(zip(X, Y, label_name)):                                                                                                                                             
        explainer = lime_image.LimeImageExplainer()                                                                                                                              
        explaination = explainer.explain_instance(image=x, classifier_fn=model.predict_on_batch, segmentation_fn=segmentation)
        lime_img = explaination.get_image_and_mask(label=y, positive_only=False, hide_rest=False, num_features=11, min_weight=0.05)[0]
        
        axs[i // 5, i % 5].imshow(normalize(lime_img)[...,::-1])
        axs[i // 5, i % 5].set_title(str(name))
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()

def plot_deep_dream(fig_name, model, X, layer_name_list, filter_idx_list, train_iter=150):
    def get_max_activation_result(model, x, layer_output, filter_idx, iter_n):
        loss = K.mean(layer_output[:, :, :, filter_idx])
        grads = K.gradients(loss, model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iter_func = K.function([model.input], [loss, grads])
        
        max_activation_result = x.copy()[np.newaxis]
        for t in range(iter_n):
            print(f'{t + 1:0{len(str(iter_n))}}', end='\r')
            loss_value, grads_value = iter_func([max_activation_result])
            max_activation_result += 1e-3 * grads_value

        return np.clip(max_activation_result[0], 0, 255).astype(np.uint8)

    K.set_learning_phase(0)
    model.trainable = False

    ## initialization for subplots
    R, C = 1 + len(layer_name_list) * len(filter_idx_list), 1 + X.shape[0]
    fig, axs = plt.subplots(R, C, figsize=(R * 4, C * 4))

    ## remove the bounding box for the layer name
    for i in range(axs.shape[0]):
        axs[i, 0].axis('off')

    # show original images and restore gamma correction
    for i, x in enumerate(X):
        axs[0, i + 1].imshow(np.clip(x[...,::-1]** 1.5, 0, 255).astype(np.uint8))  # BGR to RGB
    for i, (layer, iter_n) in enumerate(zip(layer_name_list, train_iter)):
        ## get the filter output for each image
        layer_output = model.get_layer(layer).output
        model2 = Model(model.input, layer_output)
        pred = model2.predict(X)
        pred = normalize(pred)

        for j, filter_idx in enumerate(filter_idx_list):
            axs[1 + i * len(filter_idx_list) + j, 0].text(0.35, 0.35, f'{layer}\nfilter {filter_idx}', fontsize=24)
            for k, x in enumerate(X):
                max_activation_result = get_max_activation_result(model, x, layer_output, filter_idx, iter_n)
                axs[1 + i * len(filter_idx_list) + j, 1 + k].imshow(max_activation_result[:,:, filter_idx][...,::-1])  # BGR to RGB
        
    fig.savefig(fig_name, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-f', '--model-function', default='build_model')
    parser.add_argument('-g', '--gpu', default='5')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    function = args.model_function
    input_shape = (128, 128)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    trainX, trainY = utils.load_train_data(data_dir, input_shape, normalize=False, preprocessing=True)
    print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])
    model = globals()[function](input_shape + (3,), 11)
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    print('\033[32;1mLoading Model\033[0m')
    model.load_weights(model_path)

    idx = [83, 4218, 4707, 8598]
    images, labels = trainX[idx], trainY[idx]
    plot_saliency_map('1.jpg', model, images)
    plot_filter_activation(model, images, ['conv2d_2'], list(range(5)))

    # idx = [[] for _ in range(trainY.shape[1])]
    # for i, ii in enumerate(np.argmax(trainY, axis=1)):
    #     idx[ii].append(i)
    # idx2 = [idx[0][3], idx[1][5], idx[2][3], idx[3][3], idx[4][3], idx[5][2], idx[6][2], idx[7][2], idx[8][2], idx[9][3], idx[10][3]]
    # plot_lime('3.jpg', model, trainX[idx2], range(len(idx2)),
    #             label_name=['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
    #                         'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit'])

    plot_deep_dream('4.jpg', model, images, ['conv2d_2', 'conv2d_8'], [0], [150, 1000])