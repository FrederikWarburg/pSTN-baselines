from __future__ import print_function

import torch.nn as nn
import torch


class STN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.num_param = opt.num_param
        self.N = opt.N

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

        # we initialize the model weights and bias of the regressors
        self.init_model_weights(opt)

    def init_localizer(self, opt):
        if opt.dataset.lower() in ['cub']:
            from .cublocalizer import CubSTN as STN
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebalocalizer import CelebaSTN as STN
        elif opt.dataset.lower() in ['mnist']:
            from .mnistlocalizer import MnistSTN as STN
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeserieslocalizer import TimeseriesSTN as STN

        self.stn = STN(opt)

    def init_classifier(self, opt):
        if opt.dataset.lower() in ['cub']:
            from .cubclassifier import CubClassifier as Classifier
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier as Classifier
        elif opt.dataset.lower() in ['mnist']:
            from .mnistclassifier import MnistClassifier as Classifier
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeseriesclassifier import TimeseriesClassifier as Classifier

        self.classifier = Classifier(opt)

    def init_model_weights(self, opt):
        self.stn.fc_loc[-1].weight.data.zero_()

        # Initialize the weights/bias with identity transformation
        if opt.transformer_type == 'affine':
            if self.num_param == 2:
                # We initialize bounding boxes with tiling
                bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
                self.stn.fc_loc[-1].bias.data.copy_(bias[:self.N].view(-1))
            elif self.num_param == 4:
                self.stn.fc_loc[-1].bias.data.copy_(torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 6:
                self.stn.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                      0, 1, 0] * self.N, dtype=torch.float))
        elif opt.transformer_type == 'diffeomorphic':
            # initialize param's as identity, default ok for variance in this case
            self.stn.fc_loc[-1].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.stn.theta_dim))

    def forward(self, x):

        # zoom in on relevant areas with stn
        x, theta, _ = self.stn(x)

        # make classification based on these areas
        x = self.classifier(x)

        return x, theta
