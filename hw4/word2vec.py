import os
import numpy as np
import pandas as pd
import argparse
import gensim

import utils

class Word2Vec():
    def __init__(self, embedding_dim=None):
        self.fitted = False
        self.embedding_dim = embedding_dim
        self.model = None

    def load(self, model_path):
        self.fitted = True
        self.model = gensim.models.Word2Vec.load(model_path)
        self.embedding_dim = self.model.vector_size
        return self

    def fit(self, X):
        self.model = gensim.models.Word2Vec(X, size=self.embedding_dim, window=5, min_count=5, workers=os.cpu_count() // 2, iter=10, sg=1)
        self.fitted = True
        return self

    def save(self, model_path):
        if not self.fitted:
            raise NotImplementedError
        self.model.save(model_path)

    def get_embedding(self):
        embedding = [self.model[w] for w in self.model.wv.vocab]
        pad_vector = [0.] * self.embedding_dim
        pad_vector[0] = 1e-5
        unk_vector = [0.] * self.embedding_dim
        unk_vector[1] = 1e-5
        embedding.extend([pad_vector, unk_vector])
        return np.array(embedding, np.float32)

    def get_word2idx(self):
        word2idx = {w:i for i, w in enumerate(self.model.wv.vocab)}
        word2idx['<PAD>'] = len(word2idx)
        word2idx['<UNK>'] = len(word2idx)
        return word2idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('train_data_with_label')
    parser.add_argument('train_data_without_label')
    parser.add_argument('test_data')
    args = parser.parse_args()

    trainX, _ = utils.load_train_data(args.train_data_with_label, preprocessing=False)
    trainX_no_label = utils.load_train_data(args.train_data_without_label, label=False, preprocessing=False)

    print("loading testing data ...")
    testX = utils.load_test_data(args.test_data, preprocessing=False)

    model = Word2Vec(300).fit(trainX + trainX_no_label + testX)
    
    print("saving model ...")
    model.save(args.model_path)
