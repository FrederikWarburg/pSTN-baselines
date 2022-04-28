from __future__ import print_function
from numpy import identity

import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer


class PSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.num_classes = opt.num_classes
        self.num_param = opt.num_param
        self.N = opt.N
        self.transformer, self.theta_dim = init_transformer(opt)

        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.alpha_p = opt.alpha_p
        self.beta_p = opt.beta_p
        self.reduce_samples = opt.reduce_samples
        self.opt = opt

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

        # we initialize the model weights and bias of the regressors
        self.init_model_weights(opt)

    def init_localizer(self, opt):
        raise NotImplementedError

    def init_classifier(self, opt):
        raise NotImplementedError

    def init_model_weights(self, opt):
        self.fc_loc_mu[-1].weight.data.zero_()

        # Initialize the weights/bias with identity transformation
        if opt.transformer_type == 'affine':
            # initialize mean network
            if self.num_param == 2:
                if self.N == 1:
                    bias = torch.tensor([0, 0], dtype=torch.float) 
                else:
                    # We initialize bounding boxes with tiling
                    bias = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float) * 0.5
                self.fc_loc_mu[-1].bias.data.copy_(bias[:self.N].view(-1))
            elif self.num_param == 3:
                self.fc_loc_mu[-1].bias.data.copy_(torch.tensor([1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 4:
                self.fc_loc_mu[-1].bias.data.copy_(torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 5:
                self.fc_loc_mu[-1].bias.data.copy_(torch.tensor([0, 1, 1, 0, 0] * self.N, dtype=torch.float))
            elif self.num_param == 6:
                self.fc_loc_mu[-1].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                      0, 1, 0] * self.N, dtype=torch.float))
            # initialize beta network
            self.fc_loc_beta[-2].weight.data.zero_() # TODO: check that this is still a good init
            self.fc_loc_beta[-2].bias.data.copy_(
                torch.tensor([opt.var_init], dtype=torch.float).repeat(self.num_param * self.N))

        elif opt.transformer_type == 'diffeomorphic':
            # initialize param's as identity, default ok for variance in this case
            self.fc_loc_mu[-1].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim))
            self.fc_loc_beta[-2].bias.data.copy_(
                    torch.tensor([opt.var_init], dtype=torch.float).repeat(self.theta_dim))

    def make_DA_upsample_mask(self, is_oldie, is_attractive):
        if self.opt.upsample_oldies or (self.opt.upsample_attractive_oldies and not self.training):
            DA_upsample_mask = is_oldie
        elif self.opt.upsample_attractive_oldies: 
            DA_upsample_mask = torch.logical_and(is_oldie, is_attractive)
        else: 
            DA_upsample_mask = None
        return DA_upsample_mask


    def forward(self, x, x_high_res, is_attractive=None, is_oldie=None): # last three arguments for celebA experiments at training time
        # get input shape
        batch_size = x.shape[0]
        # determine upsample strategy from meta-data
        # breakpoint()
        DA_upsample_mask = self.make_DA_upsample_mask(is_oldie, is_attractive)

        # get output for pstn module
        x, theta_samples, theta_mu_beta = self.forward_localizer(x, x_high_res, DA_upsample_mask) 

        # make classification based on pstn output
        x = self.forward_classifier(x)
        # format according to number of samples
        x = torch.stack(x.split([batch_size] * self.S)) # shape: [S, batch_size, 2]
        
        # slow, naive DA upsampling fix for now, maybe vectorize later 
        # (only use first prediction when we're not upsampling)
        # we implement it this way bc of dropout, i.e. even if theta is identity we get different predictions
        if DA_upsample_mask is not None: 
            for i in range(batch_size):
                if torch.logical_not(DA_upsample_mask)[i]:
                    x[:, i, :] = x[0, i, :]
       
        x = x.view(self.S, batch_size * self.num_classes)

        if self.training:
            if self.reduce_samples == 'min': 
                # x [S, bs * classes]
                x = x.view(self.train_samples, batch_size, self.num_classes)
                x = x.permute(1,0,2)
                # x shape: [S, bs, nr_classes]
            else: 
                # calculate mean across samples
                x = x.mean(dim=0)
                x = x.view(batch_size, self.num_classes)
                # x shape: [bs, nr_classes]

        else:
            x = torch.log(torch.tensor(1.0 / float(self.S))) + torch.logsumexp(x, dim=0)
            x = x.view(batch_size, self.num_classes)

        return x, theta_samples, theta_mu_beta

    def forward_localizer(self, x, x_high_res, DA_upsample_mask, visualise=False): # visualise=True triggers train time behaviour for validation run
        if x_high_res is None: 
            x_high_res = x
        batch_size, c, h, w = x_high_res.shape
        _, _, small_h, small_w = x.shape
        self.S = self.train_samples if self.training else self.test_samples
        
        # figure out whether to do anything here (when in train only mode)
        apply_trafos = self.training or not self.opt.aug_training_only or visualise
        if not apply_trafos:
            x = x.repeat(self.S, 1, 1, 1)
            x = x.view([self.S * batch_size, c, small_h, small_w])
            return x, None, (None, None)
        
        theta_mu, beta = self.compute_theta_beta(x, DA_upsample_mask)

        # repeat x in the batch dim so we avoid for loop
        # (this doesn't do anything for N=1)
        x_high_res = x_high_res.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, h, w)
        theta_mu_upsample = theta_mu.view(batch_size * self.N, self.theta_dim) # mean is the same for all S: [bs * N, theta_dim]
        beta_upsample = beta.view(batch_size * self.N, self.theta_dim) # variance is also the same, difference comes in through sampling
        alpha_upsample = self.alpha_p * torch.ones_like(theta_mu_upsample) # upsample scalar alpha

        # make the T-dist object and sample it here? 
        # it's apparently ok to generate distribution anew in each forward pass (e.g. https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # maybe we could do this more efficiently because of the independence assumptions within theta? 
        T_dist = distributions.studentT.StudentT(df= 2* alpha_upsample, loc=theta_mu_upsample, scale=torch.sqrt(beta_upsample / alpha_upsample))
        theta_samples = T_dist.rsample([self.S]) # shape: [self.S, batch_size, self.theta_dim]
        if DA_upsample_mask is not None: 
            # fill in idenitity if we don't want to augment (see DA_upsample_mask)
            zero_out = torch.logical_not(DA_upsample_mask) 
            theta_samples[:, zero_out, :] = torch.tensor([0., 1., 0., 0.]).to(x.device) #.unsqueeze(0).repeat(self.S, 1)# those aren't used so they shouldn't contribute to KL

        theta_samples = theta_samples.view([self.S * batch_size, self.theta_dim])

        

        # repeat for the number of samples
        x_high_res = x_high_res.repeat(self.S, 1, 1, 1)
        x_high_res = x_high_res.view([self.S * batch_size, c, h, w])
        if self.opt.dataset == 'random_placement_fashion_mnist' and self.opt.freeze_classifier:
            small_h, small_w = 28, 28
        x = self.transformer(x_high_res, theta_samples, small_image_shape=(small_h, small_w))

        # theta samples: [S, bs, nr_params]
        return x, theta_samples, (theta_mu, beta)

    def compute_theta_beta(self, x, DA_upsample_mask):
        batch_size = x.shape[0]
        x = self.localization(x)
        x = x.view(batch_size, -1)
        
        if self.opt.identity_mean: 
            theta_mu = torch.tensor([0, 1, 0, 0] * self.N, dtype=torch.float).repeat(batch_size, 1).to(x.device)
        else:
            theta_mu = self.fc_loc_mu(x)
    
        beta = self.fc_loc_beta(x)
        if DA_upsample_mask is not None: 
            # "zero out" if we don't want to augment (see DA_upsample_mask)
            zero_out = torch.logical_not(DA_upsample_mask)
            beta[zero_out] = 5e-5 # those aren't used, but contribute to KL... try to use init values 

        return theta_mu, beta

    def forward_classifier(self, x):
        return self.classifier(x)