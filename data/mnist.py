from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
import numpy as np
import torch
from torch import distributions
from utils import transformers
import torch.nn.functional as F


def transform_image_affine(x, opt):
    gaussian = distributions.normal.Normal(0, 1)  # split up the multivariate Gaussian into 1d Gaussians
    epsilon = gaussian.sample(sample_shape=torch.Size([4]))
    random_params = epsilon * opt.sigma_p
    random_params[0, 1] += 1
    affine_transformer = transformers.affine_transformation()
    theta = affine_transformer.make_affine_parameters(random_params)
    x = x.unsqueeze(0)
    grid = F.affine_grid(theta, x.size())  # makes the flow field on a grid
    x_transformed = F.grid_sample(x, grid)  # interpolates x on the grid
    return x_transformed.squeeze(0)


def make_MNIST_subset(opt, mode):
    train_trafo_no_DA = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_trafo = train_trafo_no_DA
    train_trafo_DA = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,)),
         lambda x: transform_image_affine(x, opt)])

    test_transformation = test_trafo
    train_transformation = train_trafo_DA if opt.DA else train_trafo_no_DA

    if mode == 'test':
        dataset = datasets.MNIST(
            root='data', train=False, transform=test_transformation)
    else:
        train_indices = np.load(
                'indices/MNIST%s_train_indices_fold_%s.npy' %(opt.subset, opt.fold))
        validation_indices = np.load('indices/MNIST_validation_indices.npy')
        full_training_data = datasets.MNIST(
                root='data', train=True, download=True, transform=train_transformation)
        if mode == 'train':
            dataset = Subset(full_training_data, train_indices)
        if mode == 'valid':
            dataset = Subset(full_training_data, validation_indices)
    return dataset


class MnistSideBySide(Dataset):

    def __init__(self, opt, train_div):

        self.datasets = []

        # False (test) or True (train,val)
        trainingset = train_div in ['train', 'val']

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
        self.cropsize = opt.crop_size

        # False (test) or True (train,val)
        trainingset = train_div in ['train', 'val']

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

        used_positions, target = [], ''
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

        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.cropsize),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)

        return im, int(target)

