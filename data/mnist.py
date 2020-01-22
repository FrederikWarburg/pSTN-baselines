from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch

class MnistSideBySide(Dataset):

    def __init__(self, opt, train_div):

        self.datasets = []

        # False (test) or True (train,val)
        trainingset = train_div > 0

        self.datasets.append(datasets.MNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = trainingset,
                                      download = opt.download))

        self.datasets.append(datasets.KMNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = trainingset,
                                      download = opt.download))

        self.num_images = opt.digits

    def __len__(self):
        return min([self.datasets[i].__len__() for i in range(self.num_images)])

    def __getitem__(self, idx):

        im = torch.zeros((1, 64,64), dtype=torch.float)
        target = ''
        for i in range(self.num_images):
            y = np.random.randint(0,32)
            im1, target1 = self.datasets[i].__getitem__((idx)*(i+1)%self.datasets[i].__len__())

            c, w,h = im1.shape

            x = i*w + i*8 #np.random.randint(0,32-w)

            im[:,y:y+h,x:x+w] = im1.type(torch.float)
            target += str(target1)

        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)

        return im, int(target)



class MnistRandomPlacement(Dataset):

    def __init__(self, opt, train_div):

        self.datasets = []

        # False (test) or True (train,val)
        trainingset = train_div > 0

        self.datasets.append(datasets.MNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = trainingset,
                                      download = opt.download))

        self.datasets.append(datasets.KMNIST(opt.dataroot,
                                      transform = transforms.Compose([
                                           transforms.ToTensor()
                                       ]),
                                      train = trainingset,
                                      download = opt.download))

        self.num_images = opt.digits

    def __len__(self):
        return min([self.datasets[i].__len__() for i in range(self.num_images)])

    def __getitem__(self, idx):

        im = torch.zeros((1, 96, 96), dtype=torch.float)
        target = ''

        used_positions = []
        for i in range(self.num_images):
            while True:
                x = np.random.randint(0, 96 - 32)
                if len(used_positions) == 0 or abs(used_positions[0][0] - x) > 32:
                    break
            while True:
                y = np.random.randint(0, 96 - 32)
                if len(used_positions) == 0 or abs(used_positions[0][1] - y) > 32:
                    break

            im1, target1 = self.datasets[i].__getitem__((idx)*(i+1)%self.datasets[i].__len__())

            c, w,h = im1.shape

            im[:,y:y+h,x:x+w] = im1.type(torch.float)
            target += str(target1)

        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)

        return im, int(target)
