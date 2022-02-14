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
            nn.Conv2d(self.channels, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_mu =  nn.Sequential(
            nn.Linear(128*6*6, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.theta_dim * self.N)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_beta = nn.Sequential(
            nn.Linear(128*6*6, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.theta_dim * self.N),
            nn.Softplus()
        )


class CelebaSTN(BaseSTN):
    def __init__(self, opt):
        super().__init__(opt)

        # hyper parameters
        self.channels = 3

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*6*6, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.theta_dim * self.N)
        )