from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_feature_size


class CelebaClassifier(nn.Module):
    def __init__(self, opt):
        super(CelebaClassifier, self).__init__()

        # hyper parameters
        self.N = opt.N
        self.S = opt.test_samples
        self.dropout_rate = opt.dropout_rate
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.feature_size = 640
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channels = 3

        # initialize a classifier for each branch
        self.CNN = nn.Sequential(
            nn.Conv2d(self.channels, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=5),
            nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(self.feature_size, 200)
        self.fc2 = nn.Linear(200, self.num_classes)

    def classifier(self, x):
        # get input dimensions
        batch_size, C, W, H = x.shape
        # print('classifier img shape:', x.shape)
        x = self.CNN(x)
        x = x.view(batch_size, self.feature_size)
        x = F.relu(self.fc1(x))
        # use drop out during training
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.classifier(x)
        probs = F.log_softmax(x , dim=1)
        return probs