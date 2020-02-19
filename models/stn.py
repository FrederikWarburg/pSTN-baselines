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

        self.init_model_weights()

        self.init_transformer(opt)

    def init_localizer(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cublocalizer import CubSTN
            self.stn = CubSTN(opt)
        elif opt.dataset.lower() == 'celeba':
            from .celebalocalizer import CelebaSTN
            self.stn = CelebaSTN(opt)
        elif opt.dataset.lower() == 'mnist':
            from .mnistlocalizer import MnistSTN
            self.stn = MnistSTN(opt)
        elif opt.dataset.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesSTN
            self.stn = TimeseriesSTN(opt)

    def init_classifier(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier
            self.classifier = CubClassifier(opt)
        elif opt.dataset.lower() == 'celeba':
            from .celebaclassifier import CelebaClassifier
            self.classifier = CelebaClassifier(opt)

        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import MnistClassifier
            self.classifier = MnistClassifier(opt)
        elif opt.dataset.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesClassifier
            self.classifier = TimeseriesClassifier(opt)

    def init_model_weights(self):

        # Initialize the weights/bias with identity transformation
        self.stn.fc_loc[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
            self.stn.fc_loc[2].bias.data.copy_(bias[:self.N].view(-1))
        elif self.num_param == 4:
            self.stn.fc_loc[2].bias.data.copy_(torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float))
        elif self.num_param == 6:
            self.stn.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                            0, 1, 0] * self.N, dtype=torch.float))

    def init_transformer(self, opt):

        if opt.transformer_type == 'affine':
            from utils.transformers import AffineTransformer
            self.stn.transformer = AffineTransformer()
        elif opt.transformer_type == 'diffeomorphic':
            from utils.transformers import DiffeomorphicTransformer
            self.stn.transformer = DiffeomorphicTransformer(opt)

    def forward(self, x):

        x, _, _ = self.stn(x)

        x = self.classifier(x)

        return x
