import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils import make_affine_parameters

class SimplePSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.S = opt.test_samples
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_prior = opt.sigma

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
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
        self.fc_loc_mu = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param*self.N)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_sigma = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param*self.N),
            nn.Softplus()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc_mu[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float)*0.5
            self.fc_loc_mu[2].bias.data.copy_(bias[:self.N].view(-1))
        if self.num_param == 6:
            self.fc_loc_mu[2].bias.data.copy_(torch.tensor([1,0,0,
                                                            0,1,0]*self.N, dtype=torch.float))


    def forward(self, x):
        """
            :param x: input tensor [B, C, W, H]
            :return:
                x: N*S affine transformation of input tensor [B*N*S, C, W', H']
                theta_mu: the mean estimate of theta [B, N*num_params]
                theta_sigma: the variance estimate of theta [B, N*num_params]

            A quick comment on the upsampling of x, theta_mu, theta_sigma which is used to avoid for loops

            We will use the following notation
                tm_i_j_k = theta_mu for input image i, parameters j, and sample k
                ts_i_j_k = theta_sigma for input image i, parameters j, and sample k

            Example and dimensions before upsample (assuming B = 2, S = 2, N = 2):
                x = [im1, im2]                              x = [B, C, W, H]
                theta_mu = [tm1, tm2]                       theta_mu = [B, N*num_params]
                theta_sigma = [ts1, ts2]                    theta_sigma = [B, N*num_params]

            Example and dimensions after upsample (assuming B = 2, S = 2, N = 2):
                x = [im1, im2, im1, im2, im1, im2, im1, im2]                                            x = [B*N*S, C, W, H]
                theta_mu = [tm1_1_1, tm2_1_1, tm1_2_1, tm2_2_1, tm1_1_2, tm2_1_2, tm1_1_2, tm2_1_2]     theta_mu = [B*N*S, N*num_params]
                theta_sigma = [ts1_1_1, ts2_1_1, ts1_2_1, ts2_2_1, ts1_1_2, ts2_1_2, ts1_1_2, ts2_1_2]     theta_sigma = [B*N*S, N*num_params]
        """

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

        # make affine matrix
        affine_params = make_affine_parameters(theta_mu_upsample, theta_sigma_upsample, self.sigma_prior)

        # makes the flow field on a grid
        grid = F.affine_grid(affine_params, x.size())

        # interpolates x on the grid
        x = F.grid_sample(x, grid)

        return x, (theta_mu, theta_sigma), affine_params


class SimpleSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.num_param = opt.num_param

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
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
            bias = torch.tensor([[-1,-1],[1,1],[1,-1],[-1,1]], dtype=torch.float)*0.5 # Tiling
            self.fc_loc[2].bias.data.copy_(bias[:self.N].view(-1))
        if self.num_param == 6:
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
