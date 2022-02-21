from audioop import add
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
from utils.transformers import make_affine_matrix
import math
# from RandAugment import RandAugment
# this is now native in torch, use that implementation if we want to use it again

def transform_image_affine(x, opt):
    gaussian = distributions.normal.Normal(0, 1)  # split up the multivariate Gaussian into 1d Gaussians
    epsilon = gaussian.sample(sample_shape=torch.Size([4]))
    random_params = epsilon * opt.sigma_p
    random_params[1] += 1
    theta = make_affine_matrix(*random_params)
    x = x.unsqueeze(0)
    grid = F.affine_grid(theta, x.size())  # makes the flow field on a grid
    x_transformed = F.grid_sample(x, grid)  # interpolates x on the grid
    return x_transformed.squeeze(0)


def get_trafo(opt):
    if opt.normalize: 
        train_trafo_no_DA = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        train_trafo_no_DA = transforms.ToTensor()
    test_trafo = train_trafo_no_DA

    if opt.data_augmentation == 'None':
        train_trafo = train_trafo_no_DA

    elif opt.data_augmentation == 'standard':
        if opt.normalize: 
            train_trafo = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                lambda x: transform_image_affine(x, opt)])
        else: 
            train_trafo = transforms.Compose(
                            [transforms.ToTensor(),
                            lambda x: transform_image_affine(x, opt)])
    # elif opt.data_augmentation == 'RandAugment':
    #     train_trafo = train_trafo_no_DA
    #     train_trafo.transforms.insert(0, RandAugment(opt.rand_augment_N, opt.rand_augment_M, 'full'))

    # elif opt.data_augmentation == 'AffineRandAugment':
    #     train_trafo = train_trafo_no_DA
    #     train_trafo.transforms.insert(0, RandAugment(opt.rand_augment_N, opt.rand_augment_M, 'affine'))

    else:
        print('Please pass valid DA method!')
    return train_trafo, test_trafo


def make_fashion_mnist(opt, mode):
    train_trafo, _ = get_trafo(opt)
    if mode in ['train', 'val']:
        train = True
    else:
        train = False
    dataset = datasets.FashionMNIST(
        root=opt.dataroot, train=train, download=True, transform=train_trafo)

    if mode == 'test':
        return dataset
        
    else: 
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))
        if mode =='train':
            return train_set
        if mode == 'val':
            return val_set



def make_mnist_subset(opt, mode):
    train_trafo, test_trafo = get_trafo(opt)

    if mode == 'train':
        full_training_data = datasets.MNIST(
            root=opt.dataroot, train=True, download=True, transform=train_trafo)
        dataset = full_training_data
        if opt.subset is not None: 
            train_indices = np.load(
                '%s/subset_indices/MNIST%s_train_indices_fold_%s.npy' % (opt.dataroot, opt.subset, opt.fold))
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
    # hardcode num_images=1 for now 
    def __init__(self, opt, mode):
        self.cropsize = opt.crop_size
        self.num_images = opt.digits
        self.add_kmnist_noise = opt.add_kmnist_noise
        self.mode = mode

        if opt.dataset == 'random_placement_mnist':
            self.dataset = make_mnist_subset(opt, mode)

        if opt.dataset == 'random_placement_fashion_mnist':
            self.dataset =  make_fashion_mnist(opt, mode)    

        # make training set smaller if opt.subset is set
        if opt.subset is not None and mode == 'train': 
            train_indices = torch.randint(low=0, high=self.dataset.__len__(), size=(int(opt.subset), )) 
            self.dataset = Subset(self.dataset, train_indices)


        if self.add_kmnist_noise:
            self.noise_dataset = datasets.KMNIST(opt.dataroot,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()
                                             ]),
                                             train=True,
                                             download=opt.download)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):

        im = torch.zeros((1, 96, 96), dtype=torch.float)

        used_positions = []

        while True:
            x = np.random.randint(0, 96 - 32)
            if len(used_positions) == 0 or abs(used_positions[0][0] - x) > 32:
                break
        while True:
            y = np.random.randint(0, 96 - 32)
            if len(used_positions) == 0 or abs(used_positions[0][1] - y) > 32:
                break
        im1, target = self.dataset.__getitem__(idx)
        c, w, h = im1.shape
        im[:, y:y + h, x:x + w] = im1.type(torch.float)
        ground_truth_trafo = torch.tensor([x, y], dtype=torch.float)

        if self.add_kmnist_noise:
            # add noise image in 50% of training cases and every second val/test image:
            if self.mode == 'train':
                add_noise = (torch.rand(size=[1]) > 0.5)
            else: 
                add_noise = (idx % 2 == 0)
            
            if add_noise:
                rnd_ix = np.random.randint(low=0, high=self.noise_dataset.__len__(), size=None)
                im2, _ = self.noise_dataset.__getitem__(rnd_ix)
                noise_x = x
                noise_y = y
                while abs(noise_x - x) + abs(noise_y - y) < 30:
                    noise_x = np.random.randint(0, 96 - 32)
                    noise_y = np.random.randint(0, 96 - 32)
                c, w, h = im2.shape
                im[:, noise_y:noise_y + h, noise_x:noise_x + w] = im2.type(torch.float)

        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(self.cropsize), transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        im = transform(im)
        return im, [target, ground_truth_trafo] # also return around truth x and y



class MnistRandomRotation(Dataset):
    # hardcode num_images=1 for now 
    def __init__(self, opt, mode):
        self.normalize = opt.normalize
        self.mode = mode

        if opt.dataset == 'random_placement_mnist':
            self.dataset = make_mnist_subset(opt, mode)

        if opt.dataset == 'random_placement_fashion_mnist':
            self.dataset =  make_fashion_mnist(opt, mode)    


    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        im, target = self.dataset.__getitem__(idx)

        # add rotation 'noise' in 20% of training cases and every second val/test image:
        if self.mode == 'train':
            add_noise = True # (torch.rand(size=[1]) > 0.8)
        else: 
            add_noise = True # (idx % 2 == 0)
        
        angle = torch.tensor([0.])
        if add_noise:
            angle = - torch.tensor(math.pi) + 2 * torch.tensor(math.pi) * torch.rand(size=[1])

            def transform_image_affine(x):
                random_params = torch.tensor([angle, 1., 0, 0])
                theta = make_affine_matrix(*random_params)
                x = x.unsqueeze(0)
                grid = F.affine_grid(theta, x.size())  # makes the flow field on a grid
                x_transformed = F.grid_sample(x, grid)  # interpolates x on the grid
                return x_transformed.squeeze(0)

            im = transform_image_affine(im)
        
        if self.normalize:
            trafo = transforms.Normalize((0.1307,), (0.3081,))
            im = trafo(im)

        return im, [target, angle] # also return ground truth angle