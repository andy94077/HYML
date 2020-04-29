import os, argparse
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import Adverdataset


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir')
	parser.add_argument('output_dir2')
	args = parser.parse_args()

	output_dir = args.output_dir
	output_dir2 = args.output_dir2
	if not os.path.exists(output_dir2):
		os.makedirs(output_dir2)

	for name in os.listdir(output_dir):
		Image.open(os.path.join(output_dir, name)).filter(ImageFilter.GaussianBlur(2)).save(os.path.join(output_dir2, name))
