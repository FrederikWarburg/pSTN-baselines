import torch
import torch.nn as nn
from torch import distributions

from utils.transformers import DiffeomorphicTransformer, AffineTransformer

parameter_dict_P_STN = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 14 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 38,
    'localizer_filters1': 8,
    'localizer_filters2': 14,
    'color_channels': 1
}

parameter_dict_STN = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 18 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 50,
    'localizer_filters1': 12,
    'localizer_filters2': 18,
    'color_channels': 1
}


class MnistPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # self.N = opt.N
        self.S = opt.test_samples
        # self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_p = opt.sigma_p
        self.channels = 1

        self.parameter_dict = parameter_dict_P_STN

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(
                self.parameter_dict['color_channels'], self.parameter_dict['localizer_filters1'],
                kernel_size=self.parameter_dict['loc_kernel_size']),
            nn.MaxPool2d(self.mnist_parameter_dict['max_pool_res'], stride=self.parameter_dict['max_pool_res']),
            nn.ReLU(True),
            nn.Conv2d(
                self.parameter_dict['localizer_filters1'], self.parameter_dict['localizer_filters2'],
                kernel_size=self.parameter_dict['loc_kernel_size']),
            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
            nn.ReLU(),
        )

        # Regressor for the affine matrix
        if opt.transformer_type == 'affine':
            self.fc_loc_mu = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.num_param))
        # Regressor for the diffeomorphic param's
        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc_mu = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                          self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.num_param)
            )

        if opt.transformer_type == 'affine':
            self.fc_loc_std = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus())
            # initialize transformer
            self.transfomer = AffineTransformer()

        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc_std = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                          self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(False),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus())
            # initialize transformer
            self.transfomer = DiffeomorphicTransformer(opt)

    def forward(self, x):
        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, w, h = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        theta_sigma = self.fc_loc_std(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.num_param)
        theta_sigma_upsample = theta_sigma.view(batch_size * self.N, self.num_param)
        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        theta_sigma_upsample = theta_sigma_upsample.repeat(self.S, 1)
        x, params = self.transformer(x, theta_mu_upsample, theta_sigma_upsample, self.sigma_p)
        gaussian = distributions.normal.Normal(0, 1)
        epsilon = gaussian.sample(sample_shape=x.shape).to(self.device)
        x = x + self.sigma_n * epsilon
        return x, (theta_mu, theta_sigma), params


class MnistSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.S = opt.test_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.channels = 1

        self.parameter_dict = parameter_dict_STN

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(
                self.parameter_dict['color_channels'], self.parameter_dict['localizer_filters1'],
                kernel_size=self.parameter_dict['loc_kernel_size']),
            nn.MaxPool2d(self.mnist_parameter_dict['max_pool_res'], stride=self.parameter_dict['max_pool_res']),
            nn.ReLU(True),
            nn.Conv2d(
                self.parameter_dict['localizer_filters1'], self.parameter_dict['localizer_filters2'],
                kernel_size=self.parameter_dict['loc_kernel_size']),
            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
            nn.ReLU(),
        )

        if opt.transformer_type == 'diffeomorphic':
            self.transfomer = DiffeomorphicTransformer(opt)
        else:
            self.transfomer = AffineTransformer()


    def forward(self, x):
        batch_size, c, w, h = x.shape
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_upsample = theta.view(batch_size * self.N, self.num_param)
        x, params = self.transformer(x, theta_upsample)

        return x, theta_upsample, params
