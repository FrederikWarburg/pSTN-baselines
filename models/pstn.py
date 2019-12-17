from __future__ import print_function
import torch.nn as nn

import torch.nn.functional as F
from utils.utils import make_affine_parameters
import torchvision.models as models
import torch

class pSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
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

        self.mu_fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)
        self.sigma_fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)

        self.mu_fc2 = nn.Linear(128, self.num_param*self.N)
        self.sigma_fc2 = nn.Linear(128, self.num_param*self.N)

        # Initialize the weights/bias with identity transformation
        self.mu_fc2.weight.data.zero_()
        #self.sigma_fc2.weight.data.zero_()
        if self.num_param == 2:
            #self.fc2.bias.data.normal_(0, 1).clamp_(min=-0.5,max=0.5)
            #self.fc2.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float).repeat(self.N))
            bias = torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=torch.float)*0.25#*opt.crop_size
            self.mu_fc2.bias.data.copy_(bias[:self.N].view(-1))

        elif self.num_param == 4:
            self.mu_fc2.bias.data.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.float).repeat(self.N))

    def init_classifier(self, opt):
        from .inception import InceptionClassifier

        self.classifier = InceptionClassifier(opt)

    def stn(self, x):

        batch_size, channels, height, width = x.size()
        xs = self.cnn(x)

        # estimate mean
        xs = self.conv(xs)
        xs = xs.view(batch_size, -1)

        mu_xs = F.relu(self.mu_fc1(xs))
        theta_mu = self.mu_fc2(mu_xs)

        sigma_xs = F.relue(self.sigma_fc1(xs))
        theta_sigma = self.sigma_fc2(sigma_xs)

        #TODO: CHekc this is correct
        # mu_theta =    [B*N*S, num_params]
        # sigma_theta = [B*N*S, num_params]
        # epsilon =     [B*N*S, num_params]

        x = x.repeat(self.N*self.num_samples, 1, 1, 1)
        theta_mu_upsample = torch.empty(batch_size*self.N*self.num_samples, self.num_param, requires_grad=False, device=x.device)
        theta_sigma_upsample = torch.empty(batch_size*self.N*self.num_samples, self.num_param, requires_grad=False, device=x.device)
        for i in range(self.N):
            theta_sigma_upsample[i*batch_size:(i+1)*batch_size, :] = theta_mu[:, i*self.num_param: (i+1)*self.num_param]
            theta_sigma_upsample[i*batch_size:(i+1)*batch_size, :] = theta_sigma[:, i*self.num_param: (i+1)*self.num_param]

        # make affine matrix
        affine_params = make_affine_parameters(theta_mu_upsample, theta_sigma_upsample)

        # makes the flow field on a grid
        grid = F.affine_grid(affine_params, x.size())

        # interpolates x on the grid
        x = F.grid_sample(x, grid)

        return x, theta_mu, theta_sigma

    def forward(self, x):

        x, mu, sigma = self.stn(x)

        x = self.classifier(x)

        if self.training:
            x = x.mean(dim=1)
            x = torch.cat([x, mu, sigma])
        else:
            x = torch.log(torch.tensor(1/self.num_samples)) + torch.logsumexp(x, dim=1)

        return x



