
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch

class CelebA(Dataset):

    def __init__(self, opt, data_div):

        if data_div == 0:
            split = 'train'
        elif data_div == 1:
            split = 'valid'
        elif data_div == 2:
            split ='test'

        self.datasets = datasets.CelebA(opt.dataroot,
                              transform = transforms.Compose([
                                   transforms.ToTensor()
                               ]),
                              split = split,
                              download = opt.download)

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, idx):

        return self.datasets.__getitem__(idx)
