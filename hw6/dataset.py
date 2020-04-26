import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Adverdataset(Dataset):
	def __init__(self, data_dir, transforms):
		df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
		self.X = [Image.open(os.path.join(data_dir, 'images', f'{name:0{len(str(df.shape[0]))}}.png')).resize((224, 224)) for name in df['ImgId']]
		self.label = torch.from_numpy(df['TrueLabel'].to_numpy()).long()
		self.transforms = transforms

	def __getitem__(self, idx):
		return self.transforms(self.X[idx]), self.label[idx]
	
	def __len__(self):
		return len(self.X)

