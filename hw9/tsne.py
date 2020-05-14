import os, argparse
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def TSNE_and_draw(vec):
    tsne = TSNE(n_jobs=os.cpu_count() // 2, perplexity=100)
    embedding = tsne.fit_transform(vec)
    return embedding 

def plot_result(embedding, file_path, Y=None):
    if Y is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=0.5)
    else:
        for c in np.unique(Y):
            plt.scatter(embedding[Y == c, 0], embedding[Y == c, 1], s=2, alpha=0.7, label=str(c))
        plt.legend(fancybox=True)
    plt.savefig(file_path) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vector_data')
    parser.add_argument('output_image')
    parser.add_argument('-y', '--vector-labels')
    args = parser.parse_args()

    X, Y = np.load(args.vector_data), (np.load(args.vector_labels) if args.vector_labels else None)
    X = X.reshape(X.shape[0], -1)

    embedding = TSNE_and_draw(X)
    plot_result(embedding, args.output_image, Y) 

