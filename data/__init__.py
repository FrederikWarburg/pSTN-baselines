from data.celebA import CelebA
from data.mnist import MnistXKmnist, make_mnist_subset, MnistRandomPlacement, MnistRandomRotation, make_fashion_mnist
from data.timeseries import make_timeseries_dataset
from data.mtsd import MTSD


def create_dataset(opt, mode):  # mode in ['train', 'val', 'test']
    """loads dataset class"""
    if opt.dataset.lower() == 'celeba':
        dataset = CelebA(opt, mode)
    elif opt.dataset.lower() == 'mnistxkmnist':
        dataset = MnistXKmnist(opt, mode)
    elif opt.dataset.lower() in ['random_placement_mnist', 'random_placement_fashion_mnist']:
        dataset = MnistRandomPlacement(opt, mode)
    elif opt.dataset.lower() in ['random_rotation_mnist', 'random_rotation_fashion_mnist']:
        dataset = MnistRandomRotation(opt, mode)
    elif opt.dataset.lower().startswith('mnist'):
        dataset = make_mnist_subset(opt, mode)
    elif opt.dataset.lower().startswith('mtsd'):
        dataset = MTSD(opt, mode)
    elif opt.dataset == 'fashion_mnist':
        dataset = make_fashion_mnist(opt, mode)
    elif opt.dataset in opt.TIMESERIESDATASETS:
        dataset = make_timeseries_dataset(opt, mode)
    return dataset
