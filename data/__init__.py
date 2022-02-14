import torch.utils.data

from data.celebA import CelebA
from data.cub_200_2011 import Cub2011
from data.gtsrb import GTSRB
from data.mnist import MnistXKmnist, make_mnist_subset, MnistRandomPlacement, MnistRandomRotation
from data.timeseries import make_timeseries_dataset


def create_dataset(opt, mode):  # mode in ['train', 'val', 'test']
    """loads dataset class"""
    if opt.dataset.lower() == 'cub':
        dataset = Cub2011(opt, mode)
    elif opt.dataset.lower() == 'gtsrb':
        dataset = GTSRB(opt, mode)
    elif opt.dataset.lower() == 'celeba':
        dataset = CelebA(opt, mode)
    elif opt.dataset.lower() == 'mnistxkmnist':
        dataset = MnistXKmnist(opt, mode)
    elif opt.dataset.lower() == 'random_placement_mnist':
        dataset = MnistRandomPlacement(opt, mode)
    elif opt.dataset.lower() == 'random_rotation_mnist':
        dataset = MnistRandomRotation(opt, mode)
    elif opt.dataset.lower().startswith('mnist'):
        dataset = make_mnist_subset(opt, mode)
    elif opt.dataset in opt.TIMESERIESDATASETS:
        dataset = make_timeseries_dataset(opt, mode)
    return dataset
