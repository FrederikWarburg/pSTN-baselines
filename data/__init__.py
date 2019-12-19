import torch.utils.data
from data.cub_200_2011 import Cub2011
from data.mnist import Mnist4x4grid

def CreateDataset(opt):
    """loads dataset class"""
    if opt.dataset.lower() == 'cub':
        dataset = Cub2011(opt)
    elif opt.dataset.lower() == 'mnist':
        dataset = Mnist4x4grid(opt)

    return dataset

class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt

        self.dataset = CreateDataset(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.no_shuffle,
            num_workers=int(opt.num_threads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
