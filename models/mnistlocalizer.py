import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils import make_affine_parameters, diffeomorphic_transformation, affine_transformation

class SimplePSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.S = opt.test_samples
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_prior = opt.sigma
        self.channels = 1 if 'mnist' in opt.dataset.lower() else 3

        # Spatial transformer localization-network
        self.localization = nn.Sequential()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_mu = nn.Sequential()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_sigma = nn.Sequential()

        # Initialize the weights/bias with identity transformation
        self.fc_loc_mu[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float)*0.5
            self.fc_loc_mu[2].bias.data.copy_(bias[:self.N].view(-1))
        elif self.num_param == 4:
            self.fc_loc[2].bias.data.copy_(torch.tensor([0,1,0,0]*self.N, dtype=torch.float))
        elif self.num_param == 6:
            self.fc_loc_mu[2].bias.data.copy_(torch.tensor([1,0,0,
                                                            0,1,0]*self.N, dtype=torch.float))

        if opt.transformer_type == 'affine':
            self.transfomer = affine_transformation
        elif opt.transformer_type == 'diffeomorphic':
            self.transfomer = diffeomorphic_transformation

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

        x, affine_params = self.transformer(theta_mu_upsample, theta_sigma_upsample, self.sigma_prior)

        return x, (theta_mu, theta_sigma), affine_params


class SimpleSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.num_param = opt.num_param
        self.channels = 1 if 'mnist' in opt.dataset.lower() else 3

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param*self.N)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float)*0.5
            self.fc_loc[2].bias.data.copy_(bias[:self.N].view(-1))
        elif self.num_param == 4:
            self.fc_loc[2].bias.data.copy_(torch.tensor([0,1,0,0]*self.N, dtype=torch.float))
        elif self.num_param == 6:
            self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,
                                                         0,1,0]*self.N, dtype=torch.float))



    def forward(self, x):

        batch_size, c, w, h = x.shape

        xs = self.localization(x)

        xs = xs.view(batch_size, -1)

        theta = self.fc_loc(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1,self.N,1,1,1).view(self.N*batch_size,c,w,h)
        theta_upsample = theta.view(batch_size * self.N, self.num_param)

        affine_params = make_affine_parameters(theta_upsample)

        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)

        return x, theta, affine_params
