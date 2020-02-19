from __future__ import print_function

import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier as classifier
        elif opt.dataset.lower() in ['celeba','mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier as classifier
        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import CNNClassifier as classifier
        elif opt.dataset.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesClassifier as classifier

        self.model = classifier(opt)

    def forward(self, x):

        return self.model(x)
