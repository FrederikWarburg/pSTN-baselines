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

        self.init_model_weights()

        self.init_transformer(opt)

    def init_localizer(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cublocalizer import CubPSTN
            self.pstn = CubPSTN(opt)
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebalocalizer import CelebaPSTN
            self.pstn = CelebaPSTN(opt)
        elif opt.dataset.lower() == 'mnist':
            from .mnistlocalizer import MnistPSTN
            self.pstn = MnistPSTN(opt)
        elif opt.dataset.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesPSTN
            self.pstn = TimeseriesPSTN(opt)

    def init_classifier(self, opt):
        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier
            self.classifier = CubClassifier(opt)
        elif opt.dataset.lower() in ['celeba', 'mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier
            self.classifier = CelebaClassifier(opt)
        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import MnistClassifier
            self.classifier = MnistClassifier(opt)
        elif opt.dataset.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesClassifier
            self.classifier = TimeseriesClassifier(opt)

    def init_model_weights(self):

        self.pstn.fc_loc_std[2].weight.data.zero_()
        self.pstn.fc_loc_std[2].bias.data.copy_(
            torch.tensor([-2], dtype=torch.float).repeat(self.num_param * self.N))

        # Initialize the weights/bias with identity transformation
        self.pstn.fc_loc_mu[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
            self.pstn.fc_loc_mu[2].bias.data.copy_(bias[:self.N].view(-1))
        elif self.num_param == 4:
            self.pstn.fc_loc_mu[2].bias.data.copy_(torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float))
        elif self.num_param == 6:
            self.pstn.fc_loc_mu[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                            0, 1, 0] * self.N, dtype=torch.float))

        """
        if opt.transformer_type == 'affine':
            # initialize param's as identity
            self.fc_loc_mean[0].weight.data.zero_()
            self.fc_loc_mean[0].bias.data.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.float))
            self.fc_loc_std[0].weight.data.zero_()
            self.fc_loc_std[0].bias.data.copy_(torch.tensor([-2, -2, -2, -2], dtype=torch.float))
            # initialize transformer
            self.transfomer = affine_transformation()

        elif opt.transformer_type == 'diffeomorphic':
            # initialize param's as identity, default ok for variance in this case
            self.fc_loc_mean[2].weight.data.zero_()
            self.fc_loc_mean[2].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim)).to(self.device)
            # initialize transformer
            self.transfomer = diffeomorphic_transformation(opt)
            
        # Regressor for the affine matrix
        if opt.transformer_type == 'affine':
            self.fc_loc = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.theta_dim))
            # initialize param's as identity
            self.fc_loc[0].weight.data.zero_()
            self.fc_loc[0].bias.data.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.float))
            self.transfomer = affine_transformation()

        # Regressor for the diffeomorphic param's
        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                          self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.num_param))
            # initialize param's as identity, default ok for variance in this case
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim)).to(self.device)
            # initialize transformer
            self.transfomer = diffeomorphic_transformation(opt)
            
        # initialize param's
        self.fc_loc_mean[0].weight.data.zero_()
        self.fc_loc_mean[0].bias.data.copy_(
            torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim)).to(self.device)
        self.fc_loc_std[0].weight.data.zero_()
        self.fc_loc_std[0].bias.data.copy_(
            torch.tensor([-2], dtype=torch.float).repeat(self.theta_dim)).to(self.device)

        """

    def init_transformer(self, opt):

        if opt.transformer_type == 'affine':
            from utils.transformers import AffineTransformer
            self.pstn.transformer = AffineTransformer()
        elif opt.transformer_type == 'diffeomorphic':
            from utils.transformers import DiffeomorphicTransformer
            self.pstn.transformer = DiffeomorphicTransformer(opt)

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
