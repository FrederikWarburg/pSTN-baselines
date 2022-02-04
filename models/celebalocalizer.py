import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer

from .base_stn import BaseSTN
from .base_pstn import BasePSTN


class CelebaPSTN(BasePSTN):
    def __init__(self, opt):
        super().__init__(opt)

        # number of channels
        self.channels = 3

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
            nn.Linear(100, self.theta_dim * self.N)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_beta = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.theta_dim * self.N),
            nn.Softplus()
        )


class CelebaSTN(BaseSTN):
    def __init__(self, opt):
        super().__init__(opt)

        # hyper parameters
        self.channels = 3

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
            nn.Linear(100, self.theta_dim * self.N)
        )