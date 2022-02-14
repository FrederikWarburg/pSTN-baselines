import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer


class BaseSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.test_samples = opt.test_samples
        self.channels = 1
        self.transformer, self.theta_dim = init_transformer(opt)

        self.localization = None
        self.fc_loc = None

    def forward(self, x):
        batch_size, c, w, h = x.shape
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        # input size = xs.shape[1]
        theta = self.fc_loc(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_upsample = theta.view(batch_size * self.N, self.theta_dim)
        x = self.transformer(x, theta_upsample)
        return x, theta
