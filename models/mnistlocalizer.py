import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.transformers import make_affine_parameters, diffeomorphic_transformation, affine_transformation

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
        #self.N = opt.N
        self.S = opt.test_samples
        #self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_prior = opt.sigma
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
            # TODO: initialize weights
        # Regressor for the diffeomorphic param's
        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc_mu = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.num_param)
            )

        if opt.transformer_type == 'affine':
            self.fc_loc_sigma = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus())
        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc_sigma = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(False),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus())

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

    def forward(self, x):
        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, w, h = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        theta_sigma = self.fc_loc_sigma(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1,self.N,1,1,1).view(self.N*batch_size,c,w,h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.num_param)
        theta_sigma_upsample = theta_sigma.view(batch_size * self.N, self.num_param)
        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        theta_sigma_upsample = theta_sigma_upsample.repeat(self.S, 1)
        x, params = self.transformer(theta_mu_upsample, theta_sigma_upsample, self.sigma_prior)
        return x, (theta_mu, theta_sigma), params


class SimpleSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.S = opt.test_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_prior = opt.sigma
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
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.num_param))
            # initialize param's as identity, default ok for variance in this case
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim)).to(self.device)
            # initialize transformer
            self.transfomer = diffeomorphic_transformation(opt)

    def forward(self, x):
        batch_size, c, w, h = x.shape
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N*batch_size,c,w,h)
        theta_upsample = theta.view(batch_size * self.N, self.num_param)
        affine_params = make_affine_parameters(theta_upsample)
        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)

        return x, theta, affine_params
