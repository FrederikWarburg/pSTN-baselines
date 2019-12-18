from __future__ import print_function
import torch.nn as nn

import torch.nn.functional as F
from utils.utils import make_affine_parameters
import torchvision.models as models
import torch

class PSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.num_classes = opt.num_classes
        self.num_samples = opt.samples
        self.num_param = 2 if opt.fix_scale_and_rot else 4

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

    def init_localizer(self, opt):

        # "Inception architecture with batch normalisation pretrained on ImageNet"
        inception = models.googlenet(pretrained = True)

        # "remove the last pooling layer to preserve the spatial information"
        layers = list(inception.children())[:-3]
        self.cnn = nn.Sequential(*layers)
        # add three weight layers

        if opt.is_train:
            count = 0
            for i, child in enumerate(self.cnn.children()):
                for param in child.parameters():
                    if count < opt.freeze_layers:
                        param.requires_grad = False

                    count += 1

        self.conv = nn.Conv2d(1024, 128, 1)

        # mean regressor
        self.mu_fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)
        self.mu_fc2 = nn.Linear(128, self.num_param*self.N)

        # variance regressor
        self.sigma_fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)
        self.sigma_fc2 = nn.Linear(128, self.num_param*self.N)

        # Initialize the weights/bias with identity transformation
        self.mu_fc2.weight.data.zero_()

        if self.num_param == 2:
            bias = torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=torch.float)*0.25
            self.mu_fc2.bias.data.copy_(bias[:self.N].view(-1))

        elif self.num_param == 4:
            self.mu_fc2.bias.data.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.float).repeat(self.N))

    def init_classifier(self, opt):
        from .inception import InceptionClassifier

        self.classifier = InceptionClassifier(opt)

    def pstn(self, x):
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

        batch_size, channels, height, width = x.size()

        # shared localizer
        xs = self.cnn(x)
        xs = self.conv(xs)
        xs = xs.view(batch_size, -1)

        # estimate mean with mean regressor
        mu_xs = F.relu(self.mu_fc1(xs))
        theta_mu = self.mu_fc2(mu_xs)

        # estimate sigma with variance regressor
        sigma_xs = F.relu(self.sigma_fc1(xs))
        theta_sigma = F.softplus(self.sigma_fc2(sigma_xs))

        # repeat x in the batch dim so we avoid for loop
        x = x.repeat(self.N*self.num_samples, 1, 1, 1)

        # initialized upsampled thetas
        theta_mu_upsample = torch.empty(batch_size*self.N, self.num_param, requires_grad=False, device=x.device)
        theta_sigma_upsample = torch.empty(batch_size*self.N, self.num_param, requires_grad=False, device=x.device)

        # split the shared theta into the N branches
        for i in range(self.N):
            theta_mu_upsample[i*batch_size:(i+1)*batch_size, :] = theta_mu[:, i*self.num_param: (i+1)*self.num_param]
            theta_sigma_upsample[i*batch_size:(i+1)*batch_size, :] = theta_sigma[:, i*self.num_param: (i+1)*self.num_param]

        # repeat for the number of samples
        theta_mu_upsample = theta_mu_upsample.repeat(self.num_samples, 1)
        theta_sigma_upsample = theta_sigma_upsample.repeat(self.num_samples, 1)

        # make affine matrix
        affine_params = make_affine_parameters(theta_mu_upsample, theta_sigma_upsample)

        # makes the flow field on a grid
        grid = F.affine_grid(affine_params, x.size())

        # interpolates x on the grid
        x = F.grid_sample(x, grid)

        return x, (theta_mu, theta_sigma), affine_params

    def forward(self, x):

        x, theta, _ = self.pstn(x)

        x = self.classifier(x)

        x = x.view(-1, self.num_samples, self.num_classes)

        if self.training:
            mu, sigma = theta
            x = (x.mean(dim=1), mu, sigma)
        else:
            x = torch.log(torch.tensor(1/self.num_samples)) + torch.logsumexp(x, dim=1)

        return x



