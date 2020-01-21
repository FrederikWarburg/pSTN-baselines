import torch.nn as nn
import torch.nn.functional as F
from utils.utils import make_affine_parameters
import torchvision.models as models
import torch

FEATURE_SIZES = {'inception'    : 1024,
                 'inception_v3' : 2048,
                 'resnet50'     : 2048,
                 'resnet34'     : 512}

class InceptionPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.S = opt.test_samples
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_prior = opt.sigma
        self.feature_size = FEATURE_SIZES[opt.basenet.lower()]

        self.init_localizer(opt)
        self.init_mean_regressor(opt)
        self.init_std_regressor(opt)

        # Initialize the weights/bias with identity transformation
        self.fc_loc_mu[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float)*0.5
            self.fc_loc_mu[2].bias.data.copy_(bias[:self.N].view(-1))
        if self.num_param == 6:
            self.fc_loc_mu[2].bias.data.copy_(torch.tensor([1,0,0,
                                                            0,1,0]*self.N, dtype=torch.float))

    def init_localizer(self, opt):

        # "Inception architecture with batch normalisation pretrained on ImageNet"
        if opt.basenet.lower() == 'inception':
            basenet = models.googlenet(pretrained=True)
            # "remove the last layer (1000-way ILSVRC classifier)"
            layers = list(basenet.children())[:-3]

        elif opt.basenet.lower() == 'inception_v3':
            basenet = models.inception_v3(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet50':
            basenet = models.resnet50(pretrained = True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet34':
            basenet = models.resnet34(pretrained = True)
            layers = list(basenet.children())[:-2]

        layers.append(nn.Conv2d(self.feature_size, 128, 1))

        self.localization = nn.Sequential(*layers)

        if opt.is_train:
            count = 0
            for i, child in enumerate(self.localization.children()):
                for param in child.parameters():
                    if count < opt.freeze_layers:
                        param.requires_grad = False

                    count += 1

    def init_mean_regressor(self, opt):

        self.fc_loc_mu = nn.Sequential(
            nn.Linear(128*(opt.crop_size//32)**2, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_param*self.N)
        )

    def init_std_regressor(self, opt):

        self.fc_loc_sigma = nn.Sequential(
            nn.Linear(128*(opt.crop_size//32)**2, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_param*self.N),
            nn.Softplus()
        )

    def forward(self, x):

        self.S = self.train_samples if self.training else self.test_samples

        batch_size, c, w, h = x.shape

        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)

        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        theta_sigma = self.fc_loc_sigma(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1,self.N,1,1,1).view(self.N*batch_size,c,w,h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.num_param)
        theta_sigma_upsample = theta_sigma.view(batch_size * self.N, self.num_param)

        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)

        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        theta_sigma_upsample = theta_sigma_upsample.repeat(self.S, 1)

        # make affine matrix
        affine_params = make_affine_parameters(theta_mu_upsample, theta_sigma_upsample, self.sigma_prior)

        # makes the flow field on a grid
        grid = F.affine_grid(affine_params, x.size())

        # interpolates x on the grid
        x = F.grid_sample(x, grid)

        return x, (theta_mu, theta_sigma), affine_params

class InceptionSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.num_param = opt.num_param
        self.feature_size = FEATURE_SIZES[opt.basenet.lower()]

        self.init_localizer(opt)
        self.init_regressor(opt)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        if self.num_param == 2:
            # Tiling
            bias = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float)*0.5
            self.fc_loc[2].bias.data.copy_(bias[:self.N].view(-1))
        if self.num_param == 6:
            self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,
                                                            0,1,0]*self.N, dtype=torch.float))

    def init_localizer(self, opt):

        # "Inception architecture with batch normalisation pretrained on ImageNet"
        if opt.basenet.lower() == 'inception':
            basenet = models.googlenet(pretrained=True)
            # "remove the last pooling layer to preserve the spatial information"
            layers = list(basenet.children())[:-3]

        elif opt.basenet.lower() == 'inception_v3':
            basenet = models.inception_v3(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet50':
            basenet = models.resnet50(pretrained = True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet34':
            basenet = models.resnet34(pretrained = True)
            layers = list(basenet.children())[:-2]

        layers.append(nn.Conv2d(self.feature_size, 128, 1))

        self.localization = nn.Sequential(*layers)

        if opt.is_train:
            count = 0
            for i, child in enumerate(self.localization.children()):
                for param in child.parameters():
                    if count < opt.freeze_layers:
                        param.requires_grad = False

                    count += 1

    def init_regressor(self, opt):

        self.fc_loc = nn.Sequential(
            nn.Linear(128*(opt.crop_size//32)**2, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_param*self.N)
        )

    def forward(self, x):

        batch_size, c, w, h = x.shape

        xs = self.localization(x)

        xs = xs.view(batch_size, -1)

        theta = self.fc_loc(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1,self.N,1,1,1).view(self.N*batch_size,c,w,h)
        theta_upsample = theta.view(batch_size * self.N, self.num_param)

        affine_params = make_affine_parameters(theta_upsample)

        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)

        return x, theta, affine_params
