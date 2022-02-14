from builtins import breakpoint
import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer


class BasePSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.train_samples = opt.train_samples
        self.S = opt.test_samples
        self.test_samples = opt.test_samples
        self.alpha_p = opt.alpha_p
        self.beta_p = opt.beta_p
        self.channels = 1
        self.transformer, self.theta_dim = init_transformer(opt)

        # Spatial transformer localization-network
        self.localization = None
        self.fc_loc_mu = None
        self.fc_loc_beta = None

    def forward(self, x):
        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, w, h = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        beta = self.fc_loc_beta(xs)

        # repeat x in the batch dim so we avoid for loop
        # (this doesn't do anything for N=1)
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.theta_dim) # mean is the same for all S: [bs * N, theta_dim]
        beta_upsample = beta.view(batch_size * self.N, self.theta_dim) # variance is also the same, difference comes in through sampling
        alpha_upsample = self.alpha_p * torch.ones_like(theta_mu_upsample) # upsample scalar alpha

        # make the T-dist object and sample it here? 
        # it's apparently ok to generate distribution anew in each forward pass (e.g. https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # maybe we could do this more efficiently because of the independence assumptions within theta? 
        T_dist = distributions.studentT.StudentT(df= 2* alpha_upsample, loc=theta_mu_upsample, scale=torch.sqrt(beta_upsample / alpha_upsample))
        theta_samples = T_dist.rsample([self.S]) # shape: [self.S, batch_size, self.theta_dim]
        theta_samples = theta_samples.view([self.S * batch_size, self.theta_dim])

        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        x = x.view([self.S * batch_size, c, w, h])
        x = self.transformer(x, theta_samples)

        # theta samples: [S, bs, nr_params]
        return x, theta_samples, (theta_mu, beta)