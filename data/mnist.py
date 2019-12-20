from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch

class Mnist4x4grid(Dataset):

    def __init__(self, opt):

        self.datasets = []

        self.datasets.append(datasets.MNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = opt.is_train,
                                      download = opt.download))

        self.datasets.append(datasets.MNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = opt.is_train,
                                      download = opt.download))

        self.num_images = opt.N

    def __len__(self):
        return min([self.datasets[i].__len__() for i in range(self.num_images)])

    def __getitem__(self, idx):

        im = torch.zeros((1, 64,64), dtype=torch.float)
        target = ''
        for i in range(self.num_images):
            im1, target1 = self.datasets[i].__getitem__(idx)

            c, w,h = im1.shape

            x = np.random.randint(0,64-w)
            y = np.random.randint(0,64-h)

            im[:,x:x+w, y:y+h] = im1.type(torch.float)
            target += str(target1)

        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)

        return im, int(target)
