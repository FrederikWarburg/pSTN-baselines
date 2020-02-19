from __future__ import print_function

import torch
import torch.nn as nn

class PSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.num_classes = opt.num_classes
        self.num_param = opt.num_param
        self.N = opt.N

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

        self.init_model_weights(opt)

        self.init_transformer(opt)

    def init_localizer(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cublocalizer import CubPSTN as pstn
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebalocalizer import CelebaPSTN as pstn
        elif opt.dataset.lower() == 'mnist':
            from .mnistlocalizer import MnistPSTN as pstn
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeserieslocalizer import TimeseriesPSTN as pstn

        self.pstn = pstn(opt)

    def init_classifier(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier as classifier
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier as classifier
        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import MnistClassifier as classifier
        elif opt.dataset in opt.TIMESERIESDATASETS:
            from .timeseriesclassifier import TimeseriesClassifier as classifier

        self.classifier = classifier(opt)

    def init_model_weights(self, opt):
        self.stn.fc_loc_mu[-1].weight.data.zero_()

        # Initialize the weights/bias with identity transformation
        if opt.transformer_type == 'affine':
            if self.num_param == 2:
                # Tiling
                bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
                self.pstn.fc_loc_mu[-1].bias.data.copy_(bias[:self.N].view(-1))
            elif self.num_param == 4:
                self.stn.fc_loc_mu[-1].bias.data.copy_(torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 6:
                self.stn.fc_loc_mu[-1].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                      0, 1, 0] * self.N, dtype=torch.float))

            # initialize variance network
            self.stn.fc_loc_std[-2].weight.data.zero_()
            self.stn.fc_loc_std[-2].bias.data.copy_(
                torch.tensor([-2], dtype=torch.float).repeat(self.num_param * self.N))

        elif opt.transformer_type == 'diffeomorphic':
            # initialize param's as identity, default ok for variance in this case
            self.stn.fc_loc_mu[-1].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.self.theta_dim))
            self.stn.fc_loc_std[-2].weight.data.zero_()

            if opt.dataset.lower() in opt.TIMESERIESDATASETS:
                self.stn.fc_loc_std[-2].bias.data.copy_(
                     torch.tensor([-2], dtype=torch.float).repeat(self.self.theta_dim))

    def init_transformer(self, opt):
        if opt.transformer_type == 'affine':
            from utils.transformers import AffineTransformer
            self.stn.transformer = AffineTransformer()
        elif opt.transformer_type == 'diffeomorphic':
            from utils.transformers import DiffeomorphicTransformer
            self.stn.transformer = DiffeomorphicTransformer(opt)

    def forward(self, x):

        batch_size, c, w, h = x.shape

        x, theta, _ = self.pstn(x)

        x = self.classifier(x)

        x = torch.stack(x.split([batch_size] * self.pstn.S))
        x = x.view(self.pstn.S, batch_size * self.num_classes)

        if self.training:
            mu, sigma = theta

            x = x.mean(dim=0)
            x = x.view(batch_size, self.num_classes)
            return (x, mu, sigma)
        else:
            x = torch.log(torch.tensor(1 / self.pstn.S)) + torch.logsumexp(x, dim=0)
            x = x.view(batch_size, self.num_classes)

        return x
