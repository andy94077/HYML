import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='', filenames=None, labels=None, transforms=None):
        self.transform = transforms
        if data_dir != '':
            filenames = sorted(os.listdir(data_dir))
            self.data = [Image.open(os.path.join(data_dir, name)) for name in filenames]            
        elif filenames:
            self.data = [Image.open(name) for name in filenames]
        else:
            raise AssertionError('data_dir and filenames can not be both empty')
        
        self.labels = torch.LongTensor(labels) if not labels and not isinstance(labels, torch.LongTensor) else labels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.data)

