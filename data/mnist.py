from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch

class Mnist4x4grid(Dataset):
    root = 'CUB_200_2011/images'

    def __init__(self, opt):
        self.dataset = datasets.MNIST(opt.dataroot, train = opt.is_train, download = opt.download)
        self.data, self.targets = self.dataset.data, self.dataset.targets
        self.num_images = 2

    def __len__(self):
        return self.dataset.__len__() // self.num_images

    def __getitem__(self, idx):

        im1, target1 = self.data[idx], int(self.targets[idx])
        im2, target2 = self.data[self.__len__() + idx], int(self.targets[self.__len__() + idx])

        w,h = im1.shape
        im = torch.zeros((64,64), dtype=torch.float)

        pos = np.random.choice([0,1,2,3], self.num_images, replace=False) # choose tiles to place
        pos = np.asarray([[0,0],[0,1],[1,0],[1,1]])[pos]

        im[pos[0,0]*w:(pos[0,0]+1)*w, pos[0,1]*h:(pos[0,1]+1)*h] = im1.type(torch.float)
        im[pos[1,0]*w:(pos[1,0]+1)*w, pos[1,1]*h:(pos[1,1]+1)*h] = im2.type(torch.float)

        target = ''
        for item in sorted([target1,target2]):
            target += str(item)

        return im.unsqueeze(0), int(target)
