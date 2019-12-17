from __future__ import print_function
import torch.nn as nn

import torch.nn.functional as F
from models.inception import InceptionClassifier
from utils.utils import make_affine_parameters
import torchvision.models as models
import torch

class pSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.train = True
        self.N = opt.N
        self.S = opt.S
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

        # 1) "1 x 1 conv layer to reduce the number of feature channels from 1024 to 128"
        self.conv = nn.Conv2d(1024, 128, 1)
        # 2) "fully-connected layer with 128-D output"
        self.mu_fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)
        self.sigma_fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)
        self.relu = nn.ReLU()
        # 3) #fully connected layer with 2N-D output, where N is the number of transformers.
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

        mu_xs = self.mu_fc1(xs)
        mu_xs = self.relu(mu_xs)
        theta_mu = self.mu_fc2(mu_xs) # [b, num_params * N]

        sigma_xs = self.sigma_fc1(xs)
        sigma_xs = self.relu(sigma_xs)
        theta_sigma = self.sigma_fc2(sigma_xs) # [b, num_params * N]

        # [im1, im2, im3] => [im1, im2, im3, im1, im2, im3]
        # [theta1, theta2, theta3] => [theta1[:num_params],theta2[:num_params], theta3[:num_params],theta1[num_params:],theta2[num_params:],theta3[num_params:]
        x = x.repeat(self.N*self.S, 1, 1, 1)
        #TODO: CHekc this is correct


        # mu_theta =    [B*N, num_params]
        # sigma_theta = [B*N, num_params]
        # epsilon =     [B*N*S, num_params]

        theta_mu_upsample = torch.empty(batch_size*self.N*self.S, self.num_param, requires_grad=False, device=theta.device)
        theta_sigma_upsample = torch.empty(batch_size*self.N*self.S, self.num_param, requires_grad=False, device=theta.device)
        for i in range(self.N):
            theta_sigma_upsample[i*batch_size:(i+1)*batch_size, :] = theta_mu[:, i*self.num_param: (i+1)*self.num_param]
            theta_sigma_upsample[i*batch_size:(i+1)*batch_size, :] = theta_sigma[:, i*self.num_param: (i+1)*self.num_param]

        affine_params = make_affine_parameters(theta_mu_upsample, theta_sigma_upsample)

        grid = F.affine_grid(affine_params, x.size())  # makes the flow field on a grid

        x = F.grid_sample(x, grid)  # interpolates x on the grid

        return x, theta_mu_upsample, theta_sigma_upsample

    def forward(self, x):

        x, _, _ = self.stn(x)

        x = self.classifier(x)

        if train:
            aggregated_probs = x.mean(dim=1)
        else:
            aggregated_probs = torch.log(torch.tensor(1/model.S)) + torch.logsumexp(output_samples, dim=1)


        return x



