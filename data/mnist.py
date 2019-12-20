from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch

class Mnist4x4grid(Dataset):

    def __init__(self, opt):
        self.dataset = datasets.MNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = opt.is_train,
                                      download = opt.download,)
        self.data, self.targets = self.dataset.data, self.dataset.targets

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):

        im1, target1 = self.dataset.__getitem__(idx)

        w,h = im1.shape
        im = torch.zeros((64,64), dtype=torch.float)

        x = np.random.randint(0,64-w)
        y = np.random.randint(0,64-h)

        im[:,x:x+w, y:y+h] = im1.type(torch.float)

        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)

        return im.unsqueeze(0), target1
