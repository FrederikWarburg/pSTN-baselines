from __future__ import print_function
from builtins import breakpoint
from matplotlib.pyplot import thetagrids

import torch.nn as nn
import torch
from utils.transformers import init_transformer
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.num_param = opt.num_param
        self.N = opt.N
        self.transformer, self.theta_dim = init_transformer(opt)
        self.opt = opt

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

        # we initialize the model weights and bias of the regressors
        self.init_model_weights(opt)

    def init_localizer(self, opt):
        raise NotImplementedError

    def init_classifier(self, opt):
        raise NotImplementedError

    def init_model_weights(self, opt):
        self.fc_loc[-1].weight.data.zero_()

        # Initialize the weights/bias with identity transformation
        if opt.transformer_type == 'affine':
            if self.num_param == 2:
                if self.N == 1:
                    bias = torch.tensor([0, 0], dtype=torch.float) 
                else:
                    # We initialize bounding boxes with tiling
                    bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
                self.fc_loc[-1].bias.data.copy_(bias[:self.N].view(-1))
            elif self.num_param == 3:
                self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 4:
                self.fc_loc[-1].bias.data.copy_(torch.tensor([0, 0.5, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 5:
                self.fc_loc[-1].bias.data.copy_(torch.tensor([0, 1, 1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 6:
                self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0,
                                                              0, 1, 0] * self.N, dtype=torch.float))
        elif opt.transformer_type == 'diffeomorphic':
            # initialize param's as identity, default ok for variance in this case
            self.fc_loc[-1].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim))

    def forward(self, x, x_high_res):
        # zoom in on relevant areas with stn
        x, theta = self.forward_localizer(x, x_high_res)
        # make classification based on these areas
        x = self.forward_classifier(x)
        return x, theta

    def forward_localizer(self, x, x_high_res):
        if x_high_res is None: # in some cases (mnist, time series, ...) we only use the low res data
            x_high_res = x
        batch_size, c, w, h = x.shape
        theta = self.compute_theta(x)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_upsample = theta.view(batch_size * self.N, self.theta_dim)
        # a hack for random placement
        if self.opt.dataset == 'random_placement_fashion_mnist' and self.opt.freeze_classifier:
            w, h = 28, 28
        x = self.transformer(x_high_res, theta_upsample, small_image_shape=(w, h))
        return x, theta

    def compute_theta(self, x):
        batch_size = x.shape[0]
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        return theta

    def forward_classifier(self, x):
        return self.classifier(x)

