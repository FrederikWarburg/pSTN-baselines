import torch
import torch.nn as nn
from torch import distributions

from utils.transformers import init_transformer


class TimeseriesPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.S = opt.test_samples
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.alpha_p = opt.alpha_p
        self.beta_p = opt.beta_p
        self.channels = 1
        self.transformer, self.theta_dim = init_transformer(opt)

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
        self.fc_loc_mu = nn.Sequential(
            nn.Linear(64, self.theta_dim)
        )

        # Regressor for the variance
        self.fc_loc_beta = nn.Sequential(
            nn.Linear(64, self.theta_dim),
            nn.Softplus()
        )


    def forward(self, x):
        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, l = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        beta = self.fc_loc_beta(xs)

        x = x.repeat(1, self.N, 1).view(self.N * batch_size, c, l)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.theta_dim)
        beta_upsample = beta.view(batch_size * self.N, self.theta_dim)
        alpha_upsample = self.alpha_p * torch.ones_like(theta_mu_upsample)

        T_dist = distributions.studentT.StudentT(df= 2* alpha_upsample, loc=theta_mu_upsample, scale=torch.sqrt(beta_upsample / alpha_upsample))
        theta_samples = T_dist.rsample([self.S]) # shape: [self.S, batch_size, self.theta_dim]
        theta_samples = theta_samples.view([self.S * batch_size, self.theta_dim])
        
        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        x = x.view([self.S * batch_size, c, l])
        x = self.transformer(x, theta_samples)

        return x, theta_samples, (theta_mu, beta)


class TimeseriesSTN(TimeseriesPSTN):
    def __init__(self, opt):
        super().__init__(opt)
        self.fc_loc = nn.Sequential(
            nn.Linear(64, self.theta_dim)
        )

    def forward(self, x):
        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, l = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.repeat(1, self.N, 1).view(self.N * batch_size, c, l)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.theta_dim)
        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1)
        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        x = self.transformer(x, theta_mu_upsample)
        return x, theta_mu
