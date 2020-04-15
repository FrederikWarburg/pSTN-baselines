from __future__ import print_function

import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.dataset.lower() == ['celeba','cub']:
            from .cubclassifier import CubClassifier as Classifier
        elif opt.dataset.lower() in ['mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier as Classifier
        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import MnistClassifier as Classifier
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeseriesclassifier import TimeseriesClassifier as Classifier

        self.model = Classifier(opt)

    def forward(self, x):

        return self.model(x)
