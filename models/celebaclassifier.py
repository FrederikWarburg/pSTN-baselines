from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CelebaClassifier(nn.Module):
    def __init__(self, opt):
        super(CelebaClassifier, self).__init__()

        # hyper parameters
        self.N = opt.N
        self.S = opt.test_samples
        self.dropout_rate = opt.dropout_rate
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.feature_size = 128 * 6 * 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channels = 3

        # initialize a classifier for each branch
        self.CNN = nn.Sequential(
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

        # create final layers
        self.fully_connected = nn.Sequential(
            nn.Linear(self.feature_size * self.N, 1024 * self.N),
            nn.Linear(1024 * self.N, opt.num_classes)
        )

    def classifier(self, x):
        bs, c, h, w = x.shape
        x = self.CNN(x)
        x_flat = x.view(bs, -1)
        pred = self.fully_connected(x_flat)
        return pred

    def forward(self, x):
        x = self.classifier(x)
        probs = F.log_softmax(x , dim=1)
        return probs