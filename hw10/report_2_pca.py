import os, sys, argparse
import importlib
import numpy as np
from sklearn.decomposition import PCA

import utils

def calc_dist(model, latents):
    return np.mean((model.inverse_transform(model.transform(latents)) - latents) ** 2, axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('latents_file', nargs='+')
    parser.add_argument('output_file', help='predicted file')
    args = parser.parse_args()

    latents_path = args.latents_file
    output_file = args.output_file

    latents = np.concatenate([np.load(path) for path in latents_path], axis=0)
    latents = latents.reshape(latents.shape[0], -1)
    print(f'\033[32;1mlatents: {latents.shape}\033[0m')

    model = PCA(2, random_state=880301).fit(latents)

    dist = calc_dist(model, latents)
    utils.generate_csv(dist, output_file)
