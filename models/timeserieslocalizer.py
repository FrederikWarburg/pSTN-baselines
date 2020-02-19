import torch
import torch.nn as nn
from torch import distributions

from utils.transformers import DiffeomorphicTransformer


class TimeseriesPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.S = opt.test_samples
        self.transformer = DiffeomorphicTransformer(opt)
        self.theta_dim = self.transformer.T.get_theta_dim()
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_p = opt.sigma_p
        self.channels = 1

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ###
            nn.Conv1d(64,
                      164, kernel_size=5),
            nn.BatchNorm1d(164),
            nn.ReLU(),
            ###
            nn.Conv1d(164, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ###
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout()
        )

        # Regressor for the mean
        self.fc_loc_mean = nn.Sequential(
            nn.Linear(64, self.theta_dim)  # HARD CODED FOR THE MEDIUM SIZE NETWORK FOR NOW
        )

        # Regressor for the variance
        self.fc_loc_std = nn.Sequential(
            nn.Linear(64, self.theta_dim),  # HARD CODED FOR THE MEDIUM SIZE NETWORK FOR NOW
            nn.Softplus()
        )


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


class TimeseriesSTN(TimeseriesPSTN):
    def __init__(self, opt):
        super().__init__(opt)
        self.fc_loc = nn.Sequential(
            nn.Linear(64, self.theta_dim)  # HARD CODED FOR THE MEDIUM SIZE NETWORK FOR NOW
        )


    def forward(self, x):
        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, w, h = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.num_param)
        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        x, params = self.transformer(x, theta_mu_upsample)
        return x, (theta_mu_upsample), params
