import os, sys
import numpy as np
import pandas as pd

from Adam import Adam

def load_train_data(hr):
    train = pd.read_csv('train.csv', encoding = 'big5').iloc[:, 3:]
    train[train == 'NR'] = 0
    train = train.to_numpy().reshape(12, 20, 18, 24).astype(np.float32) # month, day, measurement, time
    train = np.transpose(train, [0, 2, 1, 3]).reshape(12, 18, -1) # month, measurement, day * time
    train[:, 9] = np.abs(train[:, 9])
    trainX = np.concatenate([m[:, i:i+hr].reshape(1, -1) for m in train for i in range(m.shape[1] - hr)], axis=0)
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainX = np.concatenate([np.ones((trainX.shape[0], 1), np.float32), (trainX - mean) / (std + 1e-10)], axis=1)
    trainY = train[:, 9, hr:].reshape(-1, 1)
    return trainX, trainY, mean, std

def load_test_data(path, mean, std):
    pass

def train_test_split(X, Y, split_ratio=0.2):
    np.random.seed(880301)
    idx = np.random.permutation(X.shape[0])
    return X[:-int(X.shape[0] * split_ratio)], X[-int(X.shape[0] * split_ratio):], Y[:-int(Y.shape[0] * split_ratio)], Y[-int(Y.shape[0] * split_ratio):]

def rmse(X, Y, w):
    return np.sqrt(np.mean(X @ w - Y) ** 2)

def compute_gradient(XTX, XTY, w):
    return 2 * (XTX @ w - XTY + 1e-3 * w)

if __name__ == '__main__':
    input_file, output_file = sys.argv[1:3]

    trainX, trainY, mean, std = load_train_data(9)
    trainX, validX, trainY, validY = train_test_split(trainX, trainY)

    if not os.path.exists('model.npy'):
        optimizer = Adam(compute_gradient)

        XTX = trainX.T @ trainX
        XTY = trainX.T @ trainY
        epoches = 2000
        best = np.inf
        w = np.random.normal(0, 0.05, size=(trainX.shape[1], 1))
        for epoch in range(epoches):
            optimizer.update(XTX, XTY, w)
            if epoch % 100 == 0:
                valid_loss = rmse(validX, validY, w)
                print(f'epoch {epoch:04}, loss: {rmse(trainX, trainY, w):.5}, valid_loss: {valid_loss:.5}')
                if best > valid_loss:
                    np.save('model.npy', w)
                    best = valid_loss
    else:
        w = np.load('model.npy')

    testX = load_test_data(input_file, mean, std)
    Y = np.clip(np.round(testX @ w), 0, np.inf)
    df = pd.DataFrame(list(zip([f'id_{i}' for i in range(Y.shape[0])], Y.ravel())), columns=['id', 'value'])
    df.to_csv(output_file, index=False)

