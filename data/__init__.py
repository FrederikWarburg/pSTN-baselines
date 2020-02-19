import torch.utils.data

from data.celebA import CelebA
from data.cub_200_2011 import Cub2011
from data.gtsrb import GTSRB
from data.mnist import MnistSideBySide, MnistRandomPlacement, make_MNIST_subset

def CreateDataset(opt, mode):  # mode in ['train', 'val', 'test']
    """loads dataset class"""
    if opt.dataset.lower() == 'cub':
        dataset = Cub2011(opt, mode)
    elif opt.dataset.lower() == 'gtsrb':
        dataset = GTSRB(opt, mode)
    elif opt.dataset.lower() == 'celeba':
        dataset = CelebA(opt, mode)
    elif opt.dataset.lower() == 'mnist_easy':
        dataset = MnistSideBySide(opt, mode)
    elif opt.dataset.lower() == 'mnist_hard':
        dataset = MnistRandomPlacement(opt, mode)
    elif opt.dataset.lower().startswith('mnist'):
        dataset = make_mnist_subset(opt, mode)

    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt, mode):
        self.opt = opt

        self.dataset = CreateDataset(opt, mode)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.no_shuffle,
            num_workers=int(opt.num_threads),
            pin_memory=True)

        dataset_size = len(self.dataset)
        if mode == 'train': print('#training network on = %d images' % dataset_size)
        if mode == 'val': print('#validationg network on = %d images' % dataset_size)
        if mode == 'test': print('#testing network on = %d images' % dataset_size)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
