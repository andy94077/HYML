import os
import argparse
import numpy as np

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble_result')
    parser.add_argument('sources', nargs='+', help='npy files to be ensembled')
    args = parser.parse_args()

    ensemble_result = args.ensemble_result
    sources = args.sources

    f = np.load(sources[0]).astype(np.float32)
    for name in sources[1:]:
        f += np.load(name)
    f /= len(sources)
    utils.generate_csv(f, ensemble_result)
