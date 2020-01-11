import torch.utils.data
from data.cub_200_2011 import Cub2011
from data.mnist import Mnist4x4grid

def CreateDataset(opt, train, val, test):
    """loads dataset class"""

    # data_div = 0 if test, 1 if val and 2 if train.
    if test:
        data_div = 0
    elif val:
        data_div = 1
    else:
        data_div = 2

    if opt.dataset.lower() == 'cub':
        dataset = Cub2011(opt, data_div)
    elif opt.dataset.lower() == 'mnist':
        dataset = Mnist4x4grid(opt, data_div)

    return dataset

class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt, train=False, val=False, test=False):
        self.opt = opt

        self.dataset = CreateDataset(opt, train, val, test)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.no_shuffle,
            num_workers=int(opt.num_threads))

        dataset_size = len(self.dataset)
        if train: print('#training network on = %d images' % dataset_size)
        if val: print('#validationg network on = %d images' % dataset_size)
        if test: print('#testing network on = %d images' % dataset_size)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
