from __future__ import print_function
import torch.nn as nn
import torch


class STN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.num_param = opt.num_param
        self.N = opt.N

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

    def init_localizer(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cublocalizer import CubSTN
            self.stn = CubSTN(opt)
        elif opt.dataset.lower() == 'celeba':
            from .celebalocalizer import CelebaSTN
            self.stn = CelebaSTN(opt)
        elif opt.dataset.lower().startswith('mnist'):  # multiple different subsets
            from .mnistlocalizer import MnistSTN
            self.stn = MnistSTN(opt)

    def init_classifier(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier
            self.classifier = CubClassifier(opt)
        elif opt.dataset.lower() == 'celeba':
            from .celebaclassifier import CelebaClassifier
            self.classifier = CelebaClassifier(opt)
        elif opt.dataset.lower().startswith('mnist'):  # multiple different subsets
            from .mnistclassifier import MnistClassifier
            self.classifier = MnistClassifier(opt)

    def forward(self, x):

        x, _, _ = self.stn(x)

        x = self.classifier(x)

        return x
