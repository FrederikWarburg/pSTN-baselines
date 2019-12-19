from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, opt):
        super(SimpleClassifier, self).__init__()

        self.N = opt.N
        self.feature_size = 640

        self.fc1 = nn.Linear(self.feature_size*self.N, 200*self.N)
        self.fc2 = nn.Linear(200*self.N, opt.num_classes)

        self.model = nn.Module()

        for branch_ix in range(self.N):
            encoder = self.init_classifier_branch(opt)
            self.model.add_module('branch_{}'.format(branch_ix), encoder)

    def init_classifier_branch(self, opt):

        encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
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

        # Perform the usual forward pass
        batch_size = x.shape[0] // self.N
        xs = x.split([batch_size]*self.N)

        features = torch.empty(batch_size, self.feature_size*self.N, requires_grad = False, device=x.device)
        for branch_ix in range(self.N):
            x = self.model._modules['branch_{}'.format(branch_ix)].forward(xs[branch_ix])
            features[:, branch_ix*self.feature_size:(branch_ix+1)*self.feature_size] = x.view(batch_size, self.feature_size)

        x = F.relu(self.fc1(features))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
