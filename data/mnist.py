import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
from utils import transformers


def transform_image_affine(x, opt):
    gaussian = distributions.normal.Normal(0, 1)  # split up the multivariate Gaussian into 1d Gaussians
    epsilon = gaussian.sample(sample_shape=torch.Size([4]))
    random_params = epsilon * opt.sigma_p
    random_params[1] += 1
    transformer = transformers.AffineTransformer()
    theta = transformer.make_affine_matrix(*random_params)
    x = x.unsqueeze(0)
    grid = F.affine_grid(theta, x.size())  # makes the flow field on a grid
    x_transformed = F.grid_sample(x, grid)  # interpolates x on the grid
    return x_transformed.squeeze(0)


def get_trafo(opt):
    train_trafo_no_DA = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_trafo = train_trafo_no_DA

    if opt.data_augmentation == 'None':
        train_trafo = train_trafo_no_DA

    elif opt.data_augmentation == 'standard':
        train_trafo = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,)),
             lambda x: transform_image_affine(x, opt)])
    elif opt.data_augmentation == 'RandAugment':
        pass
    elif opt.data_augmentation == 'AffineRandAugment':
        pass
    else:
        print('Please pass valid DA method!')
    return train_trafo, test_trafo


def make_mnist_subset(opt, mode):
    train_trafo, test_trafo = get_trafo(opt)

    if mode == 'train':
        train_indices = np.load(
            '%s/subset_indices/MNIST%s_train_indices_fold_%s.npy' % (opt.dataroot, opt.subset, opt.fold))
        full_training_data = datasets.MNIST(
            root=opt.dataroot, train=True, download=True, transform=train_trafo)
        dataset = Subset(full_training_data, train_indices)

    if mode == 'val':
        full_training_data_no_trafo = datasets.MNIST(
            root=opt.dataroot, train=True, download=True, transform=test_trafo)
        validation_indices = np.load('%s/subset_indices/%s_validation_indices.npy' % (opt.dataroot, opt.dataset))
        dataset = Subset(full_training_data_no_trafo, validation_indices)

    if mode == 'test':
        dataset = datasets.MNIST(
            root=opt.dataroot, train=False, transform=test_trafo)

    return dataset


class MnistXKmnist(Dataset):

    def __init__(self, opt, mode):
        self.datasets = []
        self.mode = mode
        if self.mode == 'test':
            print('creating test set')
            self.samples = []
        
        # False (test) or True (train,val)
        trainingset = mode in ['train', 'val']

        transform = [transforms.Normalize((0.1307,), (0.3081,))]
        if mode in ['train'] and opt.data_augmentation:
            print("data augmentation", mode, opt.data_augmentation)
            transform.append(lambda x: transform_image_affine(x, opt))

        self.transform = transforms.Compose(transform)
        self.datasets.append(datasets.MNIST(opt.dataroot,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ]),
                                            train=trainingset,
                                            download=opt.download))

        self.datasets.append(datasets.KMNIST(opt.dataroot,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()
                                             ]),
                                             train=trainingset,
                                             download=opt.download))

        self.num_images = opt.digits

    def __len__(self):
        return min([self.datasets[i].__len__() for i in range(self.num_images)])

    def __getitem__(self, idx):
        im = torch.zeros((1, 64, 64), dtype=torch.float)
        target = ''
        for i in range(self.num_images):
            y = np.random.randint(0, 32)
            im1, target1 = self.datasets[i].__getitem__((idx) * (i + 1) % self.datasets[i].__len__())

            c, w, h = im1.shape

            x = np.random.randint(0, 32)

            im[:, y:y + h, x:x + w] = im1.type(torch.float)
            target += str(target1)
            
            if self.mode == 'test':
                self.samples.append((x, y))

        im = self.transform(im)

        return im, int(target)


class MnistRandomPlacement(Dataset):

    def __init__(self, opt, mode):

        self.datasets = []
        self.cropsize = opt.crop_size

        # False (test) or True (train,val)
        trainingset = mode in ['train', 'val']

        self.datasets.append(datasets.MNIST(opt.dataroot,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ]),
                                            train=trainingset,
                                            download=opt.download))

        self.datasets.append(datasets.KMNIST(opt.dataroot,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()
                                             ]),
                                             train=trainingset,
                                             download=opt.download))

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

            im1, target1 = self.datasets[i].__getitem__((idx) * (i + 1) % self.datasets[i].__len__())

            c, w, h = im1.shape

            im[:, y:y + h, x:x + w] = im1.type(torch.float)
            # print('created image', im.shape, 'x:', x, 'y:', y)

            target += str(target1)

        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(self.cropsize), transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)

        return im, int(target)
