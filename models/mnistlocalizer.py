import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer
from .parameter_dicts import *

class MnistPSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.train_samples = opt.train_samples
        self.S = opt.test_samples
        self.test_samples = opt.test_samples
        self.alpha_p = opt.alpha_p
        self.beta_p = opt.beta_p
        self.channels = 1
        self.transformer, self.theta_dim = init_transformer(opt)
        self.parameter_dict = load_specifications(opt)
        self.opt = opt    
        self.reduce_samples = opt.reduce_samples


        # Spatial transformer localization-network
        if opt.modeltype == '2xlarge_loc':
            self.localization = nn.Sequential(
                            nn.Conv2d(
                                self.parameter_dict['color_channels'], self.parameter_dict['localizer_filters1'],
                                kernel_size=self.parameter_dict['loc_kernel_size'], padding=2),
                            nn.MaxPool2d(self.parameter_dict['max_pool_res'], stride=self.parameter_dict['max_pool_res']),
                            nn.ReLU(True),
                            nn.Conv2d(
                                self.parameter_dict['localizer_filters1'], self.parameter_dict['localizer_filters2'],
                                kernel_size=3, padding=1),
                            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
                            nn.ReLU(),
                            nn.Conv2d(
                                self.parameter_dict['localizer_filters2'], self.parameter_dict['localizer_filters3'],
                                kernel_size=3, padding=1),
                            nn.ReLU(True),
                            nn.Conv2d(
                                self.parameter_dict['localizer_filters3'], self.parameter_dict['localizer_filters4'],
                                kernel_size=3),
                            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
                            nn.ReLU(),
                        )
        else: 
            self.localization = nn.Sequential(
                nn.Conv2d(
                    self.parameter_dict['color_channels'], self.parameter_dict['localizer_filters1'],
                    kernel_size=self.parameter_dict['loc_kernel_size']),
                nn.MaxPool2d(self.parameter_dict['max_pool_res'], stride=self.parameter_dict['max_pool_res']),
                nn.ReLU(True),
                nn.Conv2d(
                    self.parameter_dict['localizer_filters1'], self.parameter_dict['localizer_filters2'],
                    kernel_size=self.parameter_dict['loc_kernel_size']),
                nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
                nn.ReLU(),
            )

        # Regressor for the affine matrix
        if opt.transformer_type == 'affine':
            self.fc_loc_mu = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.theta_dim))
        # # Regressor for the diffeomorphic param's
        if opt.transformer_type == 'diffeomorphic' or opt.modeltype == 'large_loc':
            self.fc_loc_mu = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                            self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim)
                )

        if opt.transformer_type == 'affine':
            self.fc_loc_beta = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus()) # beta needs to be positive, and also small so maybe a logscale parametrisation would be better

        if opt.transformer_type == 'diffeomorphic' or opt.modeltype == 'large_loc':
            self.fc_loc_beta = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                            self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(False),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus())

    def forward(self, x):
        # breakpoint()

        self.S = self.train_samples if self.training else self.test_samples
        batch_size, c, w, h = x.shape
        # shared localizer
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)

        # estimate mean and variance regressor
        theta_mu = self.fc_loc_mu(xs)
        beta = self.fc_loc_beta(xs)

        # repeat x in the batch dim so we avoid for loop
        # (this doesn't do anything for N=1)
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.theta_dim) # mean is the same for all S: [bs * N, theta_dim]
        beta_upsample = beta.view(batch_size * self.N, self.theta_dim) # variance is also the same, difference comes in through sampling
        alpha_upsample = self.alpha_p * torch.ones_like(theta_mu_upsample) # upsample scalar alpha

        # make the T-dist object and sample it here? 
        # it's apparently ok to generate distribution anew in each forward pass (e.g. https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # maybe we could do this more efficiently because of the independence assumptions within theta? 
        T_dist = distributions.studentT.StudentT(df= 2* alpha_upsample, loc=theta_mu_upsample, scale=torch.sqrt(beta_upsample / alpha_upsample))
        theta_samples = T_dist.rsample([self.S]) # shape: [self.S, batch_size, self.theta_dim]
        theta_samples = theta_samples.view([self.S * batch_size, self.theta_dim])

        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1, 1)
        x = x.view([self.S * batch_size, c, w, h])
        x = self.transformer(x, theta_samples)
        if self.opt.dataset == "random_placement_fashion_mnist" and self.opt.freeze_classifier:
            x = nn.functional.interpolate(x, size=(28, 28), mode='nearest')

        # theta samples: [S, bs, nr_params]
        # print('theta_samples:', theta_samples, '\ntheta_mu', theta_mu, '\nbeta', beta)
        # exit()
        return x, theta_samples, (theta_mu, beta)


class MnistSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N = opt.N
        self.test_samples = opt.test_samples
        self.channels = 1
        self.transformer, self.theta_dim = init_transformer(opt)
        self.parameter_dict  = load_specifications(opt)
        self.opt = opt

        # Spatial transformer localization-network
        if opt.modeltype == '2xlarge_loc':
            self.localization = nn.Sequential(
                            nn.Conv2d(
                                self.parameter_dict['color_channels'], self.parameter_dict['localizer_filters1'],
                                kernel_size=self.parameter_dict['loc_kernel_size'], padding=2),
                            nn.MaxPool2d(self.parameter_dict['max_pool_res'], stride=self.parameter_dict['max_pool_res']),
                            nn.ReLU(True),
                            nn.Conv2d(
                                self.parameter_dict['localizer_filters1'], self.parameter_dict['localizer_filters2'],
                                kernel_size=3, padding=1),
                            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
                            nn.ReLU(),
                            nn.Conv2d(
                                self.parameter_dict['localizer_filters2'], self.parameter_dict['localizer_filters3'],
                                kernel_size=3, padding=1),
                            nn.ReLU(True),
                            nn.Conv2d(
                                self.parameter_dict['localizer_filters3'], self.parameter_dict['localizer_filters4'],
                                kernel_size=3),
                            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
                            nn.ReLU(),
                        )
        else: 
            self.localization = nn.Sequential(
                nn.Conv2d(
                    self.parameter_dict['color_channels'], self.parameter_dict['localizer_filters1'],
                    kernel_size=self.parameter_dict['loc_kernel_size']),
                nn.MaxPool2d(self.parameter_dict['max_pool_res'], stride=self.parameter_dict['max_pool_res']),
                nn.ReLU(True),
                nn.Conv2d(
                    self.parameter_dict['localizer_filters1'], self.parameter_dict['localizer_filters2'],
                    kernel_size=self.parameter_dict['loc_kernel_size']),
                nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
                nn.ReLU(),
            )
           

        # Regressor for the affine matrix
        if opt.transformer_type == 'affine':
            self.fc_loc = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'], self.theta_dim))
        # Regressor for the diffeomorphic param's
        if opt.transformer_type == 'diffeomorphic'  or opt.modeltype == 'large_loc':
            self.fc_loc = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                            self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim)
            )


    def forward(self, x):
        batch_size, c, w, h = x.shape
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        # repeat x in the batch dim so we avoid for loop
        x = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, w, h)
        theta_upsample = theta.view(batch_size * self.N, self.theta_dim)
        x = self.transformer(x, theta_upsample)
        if self.opt.dataset == "random_placement_fashion_mnist" and self.opt.freeze_classifier:
            x = nn.functional.interpolate(x, size=(28, 28), mode='nearest')
        return x, theta


def load_specifications(opt):
    if opt.model.lower() in ['stn']:
        if 'random_rotation' in opt.dataset.lower():
            parameter_dict = parameter_dict_localiser_rotMNIST_STN
        elif opt.dataset == "random_placement_fashion_mnist":
            parameter_dict = parameter_dict_localiser_RandomPlacementMNIST_STN
        else:
            parameter_dict = parameter_dict_localiser_MNIST_STN

    elif opt.model.lower() == 'pstn':
        if 'random_rotation' in opt.dataset.lower():
            parameter_dict = parameter_dict_localiser_rotMNIST_P_STN
            if opt.modeltype == '2xlarge_loc':
                parameter_dict['resulting_size_localizer'] = 212 * 2 * 2
        elif opt.dataset == "random_placement_fashion_mnist":
            parameter_dict = parameter_dict_localiser_RandomPlacementMNIST_P_STN
        else:
            parameter_dict = parameter_dict_localiser_MNIST_P_STN

    else:
        print('Pass valid model!')
    return parameter_dict
