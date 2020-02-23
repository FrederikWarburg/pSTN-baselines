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
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.feature_size = 640

        # number of channels
        self.channels = 1 if 'mnist' in opt.dataset.lower() else 3

        # initialize a classifier for each branch
        self.model = nn.Module()
        for branch_ix in range(self.N):
            # create an instance of the classifier
            encoder = self.init_classifier_branch(opt)

            # add it to the model with an unique name
            self.model.add_module('branch_{}'.format(branch_ix), encoder)

        # create final layers
        self.fc1 = nn.Linear(self.feature_size * self.N, 200 * self.N)
        self.fc2 = nn.Linear(200 * self.N, opt.num_classes)

    def init_classifier_branch(self, opt):

        encoder = nn.Sequential(
            nn.Conv2d(self.channels, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        return encoder

    def forward(self, x):

        # get input dimensions
        batch_size, C, W, H = x.shape

        # number of samples depends on training or testing setting
        self.S = self.train_samples if self.training else self.test_samples

        # calculate original batch size
        batch_size = batch_size // (self.N * self.S)

        # split data into original batch size dimensions
        xs = torch.stack(x.split([self.N] * self.S * batch_size))

        # calculate the features for each transformed image and store in features.
        # this correspond to concatenating the N descriptors as to decribed in the paper for S samples
        features = torch.empty(batch_size * self.S, self.feature_size * self.N, requires_grad=False, device=x.device)
        for branch_ix in range(self.N):
            x = self.model._modules['branch_{}'.format(branch_ix)].forward(xs[:, branch_ix, :, :, :])
            features[:, branch_ix * self.feature_size:(branch_ix + 1) * self.feature_size] = x.view(batch_size * self.S,
                                                                                                    self.feature_size)
        # make a classification based on the concatenated features
        x = F.relu(self.fc1(features))

        # use drop out during training
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
