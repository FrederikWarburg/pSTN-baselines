from __future__ import print_function
import torch.nn as nn

import torch.nn.functional as F
from models.inception import InceptionClassifier
from utils.utils import make_affine_parameters
import torchvision.models as models
import torch

class STN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
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
        self.fc1 = nn.Linear(128*(opt.crop_size//32)**2, 128)
        self.relu = nn.ReLU()
        # 3) #fully connected layer with 2N-D output, where N is the number of transformers.
        self.fc2 = nn.Linear(128, self.num_param*self.N)

        # Initialize the weights/bias with identity transformation
        self.fc2.weight.data.zero_()
        if self.num_param == 2:
            self.fc2.bias.data.normal_(0, 1).clamp_(min=-0.5,max=0.5)
        elif self.num_param == 4:
            self.fc2.bias.data.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.float).repeat(self.N))

    def init_classifier(self, opt):
        from .inception import InceptionClassifier

        self.classifier = InceptionClassifier(opt)

    def stn(self, x):

        batch_size, channels, height, width = x.size()
        xs = self.cnn(x)

        # estimate mean
        xs = self.conv(xs)
        xs = xs.view(batch_size, -1)
        xs = self.fc1(xs)
        xs = self.relu(xs)
        theta = self.fc2(xs) # [b, num_params * N]

        # [im1, im2, im3] => [im1, im1, im2, im2, im3, im3]
        # [theta1, theta2, theta3] => [theta1[:num_params],theta1[num_params:], theta2[:num_params],theta2[num_params:],theta3[:num_params],theta3[num_params:]
        theta_split = torch.zeros((batch_size*self.N,self.num_param), device=theta.device) #[b * N, num_params]
        x_split = torch.zeros((batch_size*self.N, channels,height,width), device=theta.device) #[b*N, im]

        for b in range(batch_size):
            for i in range(self.N):
                theta_split[b*self.N + i] = theta[b, i*self.num_param:(i+1)*self.num_param]
                x_split[b*self.N + i] = x[b]

        affine_params = make_affine_parameters(theta_split)

        grid = F.affine_grid(affine_params, x_split.size())  # makes the flow field on a grid

        x_split = F.grid_sample(x_split, grid)  # interpolates x on the grid

        return x_split, theta_split

    def forward(self, x):

        x, _ = self.stn(x)
        print("theta", _)

        x = self.classifier(x)

        return x

    def forward_viz_stn(self, input):

        x_stn, theta = self.stn(input)

        return x_stn, theta
