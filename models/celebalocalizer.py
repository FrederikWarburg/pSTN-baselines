import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer

class CelebaPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.N = opt.N
        self.S = opt.test_samples
        #self.dropout_rate = opt.dropout_rate
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_p = opt.sigma_p
        self.sigma_n = opt.sigma_n

        # number of channels
        self.channels = 1 if 'mnist' in opt.dataset.lower() else 3

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            #nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            #nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_mu = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param * self.N)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_std = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param * self.N),
            nn.Softplus()
        )

        # initialize transformer attribute
        self.transformer, self.theta_dim = init_transformer(opt)

    def forward(self, x):

        # get input dimensions
        batch_size, c, w, h = x.shape

        # number of samples depends on training or testing setting
        self.S = self.train_samples if self.training else self.test_samples

        # shared localizer
        xs = self.localization(x)

        # reshape [B, c' * w' * h']
        xs = xs.view(batch_size, -1)

        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        theta_sigma = self.fc_loc_std(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)

        # reshape theta_mu and theta_sigma to match the shape of x
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.num_param)
        theta_sigma_upsample = theta_sigma.view(batch_size * self.N, self.num_param)

        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        theta_sigma_upsample = theta_sigma_upsample.repeat(self.S, 1)

        # transform x for each sample
        x, theta_samples = self.transformer(x, theta_mu_upsample, theta_sigma_upsample)

        # add color space noise to all samples
        gaussian = distributions.normal.Normal(0, 1)
        epsilon = gaussian.sample(sample_shape=x.shape).to(x.device)
        x = x + self.sigma_n * epsilon

        return x, theta_samples, (theta_mu, theta_sigma)


class CelebaSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.N = opt.N
        self.num_param = opt.num_param
        #self.dropout_rate = opt.dropout_rate

        # number of channels
        self.channels = 1 if 'mnist' in opt.dataset.lower() else 3

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            #nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            #nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param * self.N)
        )

        # initializer transformer
        self.transformer, self.theta_dim = init_transformer(opt)

    def forward(self, x):

        # get input dimensions
        batch_size, c, w, h = x.shape

        # shared localizer
        xs = self.localization(x)

        # reshape [B, c' * w' * h']
        xs = xs.view(batch_size, -1)

        # estimate transformation with regressor
        theta = self.fc_loc(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)

        # reshape theta to match the shape of x
        theta_upsample = theta.view(batch_size * self.N, self.num_param)

        # transform x
        x, thetas = self.transformer(x, theta_upsample)

        return x, thetas
