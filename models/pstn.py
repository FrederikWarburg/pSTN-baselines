from __future__ import print_function

import torch
import torch.nn as nn


class PSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.num_classes = opt.num_classes
        self.num_param = opt.num_param
        self.N = opt.N

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

        # we initialize the model weights and bias of the regressors
        self.init_model_weights(opt)

    def init_localizer(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cublocalizer import CubPSTN as PSTN
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebalocalizer import CelebaPSTN as PSTN
        elif opt.dataset.lower() == 'mnist':
            from .mnistlocalizer import MnistPSTN as PSTN
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeserieslocalizer import TimeseriesPSTN as PSTN

        self.pstn = PSTN(opt)

    def init_classifier(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier as Classifier
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier as Classifier
        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import MnistClassifier as Classifier
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeseriesclassifier import TimeseriesClassifier as Classifier

        self.classifier = Classifier(opt)

    def init_model_weights(self, opt):
        self.pstn.fc_loc_mu[-1].weight.data.zero_()

        # Initialize the weights/bias with identity transformation
        if opt.transformer_type == 'affine':

            # initialize mean network
            if self.num_param == 2:
                # We initialize bounding boxes with tiling
                bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
                self.pstn.fc_loc_mu[-1].bias.data.copy_(bias[:self.N].view(-1))
            elif self.num_param == 4:
                self.pstn.fc_loc_mu[-1].bias.data.copy_(torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 6:
                self.pstn.fc_loc_mu[-1].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                      0, 1, 0] * self.N, dtype=torch.float))

            # initialize variance network
            self.pstn.fc_loc_std[-2].weight.data.zero_()
            self.pstn.fc_loc_std[-2].bias.data.copy_(
                torch.tensor([-2], dtype=torch.float).repeat(self.num_param * self.N))

        elif opt.transformer_type == 'diffeomorphic':
            # initialize param's as identity, default ok for variance in this case
            self.pstn.fc_loc_mu[-1].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.pstn.theta_dim))
            self.pstn.fc_loc_std[-2].weight.data.zero_()

            if opt.dataset.lower() in opt.TIMESERIESDATASETS:
                self.pstn.fc_loc_std[-2].bias.data.copy_(
                     torch.tensor([-2], dtype=torch.float).repeat(self.pstn.theta_dim))

    def forward(self, x):
        # get input shape
        batch_size = x.shape[0]
        # get output for pstn module
        x, theta, _ = self.pstn(x)
        # make classification based on pstn output
        x = self.classifier(x)
        # format according to number of samples
        x = torch.stack(x.split([batch_size] * self.pstn.S))
        x = x.view(self.pstn.S, batch_size * self.num_classes)

        if self.training:
            # unpack theta
            mu, sigma = theta

            # calculate mean across samples
            x = x.mean(dim=0)
            x = x.view(batch_size, self.num_classes)

            # during training we want to return the mean as well as mu and sigma as the elbo uses all for optimization
            return (x, mu, sigma)
        else:
            x = torch.log(torch.tensor(1 / self.pstn.S)) + torch.logsumexp(x, dim=0)
            x = x.view(batch_size, self.num_classes)

        return x
