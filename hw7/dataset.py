import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from tqdm import tqdm

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='', filenames=None, labels=None, transforms=None):
        self.transform = transforms
        self.labels = torch.LongTensor(labels) if labels is not None and not isinstance(labels, torch.LongTensor) else labels

        if data_dir != '':
            filenames = sorted(os.listdir(data_dir))
            self.data = [Image.fromarray(cv2.imread(os.path.join(data_dir, name))[...,::-1]) for name in tqdm(filenames)]
            self.labels = torch.LongTensor([int(os.path.basename(name).split('_')[0]) for name in filenames])
        elif filenames is not None:
            self.data = [Image.fromarray(cv2.imread(name)[...,::-1]) for name in tqdm(filenames)]
        else:
            raise AssertionError('data_dir and filenames can not be both empty')
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.data)

