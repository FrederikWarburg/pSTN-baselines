import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import distributions

# possible base networks and their feature sizes
FEATURE_SIZES = {'inception': 1024,
                 'inception_v3': 2048,
                 'resnet50': 2048,
                 'resnet34': 512}


class CubPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.N = opt.N
        self.S = opt.test_samples
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param
        self.sigma_p = opt.sigma_p
        self.feature_size = FEATURE_SIZES[opt.basenet.lower()]

        # initialzie localizer, mean regressor and std regressor
        self.init_localizer(opt)
        self.init_mean_regressor(opt)
        self.init_std_regressor(opt)

        # initialize transformer
        self.transformer = None

    def init_localizer(self, opt):

        # Initialized base network
        if opt.basenet.lower() == 'inception':
            basenet = models.googlenet(pretrained=True)
            layers = list(basenet.children())[:-3]

        elif opt.basenet.lower() == 'inception_v3':
            basenet = models.inception_v3(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet50':
            basenet = models.resnet50(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet34':
            basenet = models.resnet34(pretrained=True)
            layers = list(basenet.children())[:-2]

        # add new 2D conv layer to retain spatial information
        layers.append(nn.Conv2d(self.feature_size, 128, 1))

        # add to localizer
        self.localization = nn.Sequential(*layers)

        # we add the option to freeze some of the layers to avoid overfitting
        count = 0
        for i, child in enumerate(self.localization.children()):
            for param in child.parameters():
                if count < opt.freeze_layers:
                    param.requires_grad = False

                count += 1

    def init_mean_regressor(self, opt):

        self.fc_loc_mu = nn.Sequential(
            nn.Linear(128 * (opt.crop_size // 32) ** 2, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_param * self.N)
        )

    def init_std_regressor(self, opt):

        self.fc_loc_std = nn.Sequential(
            nn.Linear(128 * (opt.crop_size // 32) ** 2, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_param * self.N),
            nn.Softplus()
        )

    def forward(self, x):

        # get input dimensions
        batch_size, c, w, h = x.shape

        # number of samples depends on training or testing setting
        self.S = self.train_samples if self.training else self.test_samples

        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)

        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        theta_sigma = self.fc_loc_std(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.num_param)
        theta_sigma_upsample = theta_sigma.view(batch_size * self.N, self.num_param)

        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)

        theta_mu_upsample = theta_mu_upsample.repeat(self.S, 1)
        theta_sigma_upsample = theta_sigma_upsample.repeat(self.S, 1)

        # transform x for each sample
        x, params = self.transformer(x, theta_mu_upsample, theta_sigma_upsample)

        # add color space noise to all samples
        gaussian = distributions.normal.Normal(0, 1)
        epsilon = gaussian.sample(sample_shape=x.shape).to(x.device)
        x = x + self.sigma_n * epsilon

        return x, (theta_mu, theta_sigma), params


class CubSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.N = opt.N
        self.num_param = opt.num_param
        self.feature_size = FEATURE_SIZES[opt.basenet.lower()]

        # initalize localizer and regressor
        self.init_localizer(opt)
        self.init_regressor(opt)

        # initalize transformer
        self.transformer = None

    def init_localizer(self, opt):

        # Initialized base network
        if opt.basenet.lower() == 'inception':
            basenet = models.googlenet(pretrained=True)
            layers = list(basenet.children())[:-3]

        elif opt.basenet.lower() == 'inception_v3':
            basenet = models.inception_v3(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet50':
            basenet = models.resnet50(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet34':
            basenet = models.resnet34(pretrained=True)
            layers = list(basenet.children())[:-2]

        # add new 2D conv layer to retain spatial information
        layers.append(nn.Conv2d(self.feature_size, 128, 1))

        # add to localizer
        self.localization = nn.Sequential(*layers)

        # we add the option to freeze some of the layers to avoid overfitting
        count = 0
        for i, child in enumerate(self.localization.children()):
            for param in child.parameters():
                if count < opt.freeze_layers:
                    param.requires_grad = False

                count += 1

    def init_regressor(self, opt):

        self.fc_loc = nn.Sequential(
            nn.Linear(128 * (opt.crop_size // 32) ** 2, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_param * self.N)
        )

    def forward(self, x):

        # get input dimensions
        batch_size, c, w, h = x.shape

        # shared localizer
        xs = self.localization(x)

        # reshape [B, c' * w' * h']
        xs = xs.view(batch_size, -1)

        # estimate transformation with regressor
        theta = self.fc_loc(xs)

        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)

        # reshape theta to match the shape of x
        theta_upsample = theta.view(batch_size * self.N, self.num_param)

        # transform x
        x, params = self.transformer(x, theta_upsample)

        return x, theta, params
