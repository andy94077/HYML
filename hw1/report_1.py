import argparse
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
from Adam import Adam

def rmse(X, Y, w):
    return np.sqrt(np.mean((X @ w - Y) ** 2))

def compute_gradient(XTX, XTY, w):
    return 2 * (XTX @ w - XTY)

if __name__ == '__main__':
    trainX, trainY, mean, std = utils.load_train_data(sys.argv[1], 9)

    XTX = trainX.T @ trainX
    XTY = trainX.T @ trainY

    for lr in [1., 1e-1, 1e-2, 1e-3]:
        loss = []
        optimizer = Adam(compute_gradient, lr)

        epoches = 40000
        w = np.random.normal(0, 0.05, size=(trainX.shape[1], 1)).astype(np.float32)
        for epoch in range(epoches):
            optimizer.update(XTX, XTY, w)
            if epoch % 100 == 0:
                loss.append(rmse(trainX, trainY, w))

        plt.plot(list(range(0, epoches, 100)), loss, label=f'{lr:.1e}')

    plt.title('loss with different lr')
    plt.xlabel('iteration')
    plt.ylabel('rmse')
    plt.legend()
    plt.savefig('report_1.jpg', bbox_inches='tight')

