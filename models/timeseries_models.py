from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from utils import timeseries_io as io
from models.stn import STN
from models.pstn import PSTN
import torch

parameter_dict_timeseries_CNN = {
    'CNN_filters1': 164,
    'CNN_filters2': 256,
    'CNN_filters3': 256,
    'CNN_filters4': 256,
    'CNN_filters5': 256,
    'CNN_kernel_size1': 8,
    'CNN_kernel_size2': 5,
    'CNN_kernel_size3': 5,
    'CNN_kernel_size4': 3,
}

parameter_dict_timeseries_STN = {
    'CNN_filters1': 196,
    'CNN_filters2': 256,
    'CNN_filters3': 256,
    'CNN_filters4': 256,
    'CNN_filters5': 64,
    'CNN_kernel_size1': 8,
    'CNN_kernel_size2': 5,
    'CNN_kernel_size3': 5,
    'CNN_kernel_size4': 3,
}

# those are the same in this case
parameter_dict_timeseries_P_STN = parameter_dict_timeseries_STN


class TimeseriesPSTN(PSTN):
    def __init__(self, opt):
        super().__init__(opt)

    def init_localizer(self, opt):
        self.localization = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ###
            nn.Conv1d(64,
                      164, kernel_size=5),
            nn.BatchNorm1d(164),
            nn.ReLU(),
            ###
            nn.Conv1d(164, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ###
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout()
        )

        # Regressor for the mean
        self.fc_loc_mu = nn.Sequential(
            nn.Linear(64, self.theta_dim)
        )

        # Regressor for the variance
        self.fc_loc_beta = nn.Sequential(
            nn.Linear(64, self.theta_dim),
            nn.Softplus()
        )

    def init_classifier(self, opt):
        self.classifier = TimeseriesClassifier(opt)

    def forward_localizer(self, x, x_high_res):
        batch_size, c, l = x.shape
        self.S = self.train_samples if self.training else self.test_samples
        
        theta_mu, beta = self.compute_theta_beta(x)
        alpha_upsample = self.alpha_p * torch.ones_like(theta_mu) # upsample scalar alpha

        # make the T-dist object and sample it here? 
        # it's apparently ok to generate distribution anew in each forward pass (e.g. https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # maybe we could do this more efficiently because of the independence assumptions within theta? 
        T_dist = torch.distributions.studentT.StudentT(df= 2* alpha_upsample, loc=theta_mu, scale=torch.sqrt(beta / alpha_upsample))
        theta_samples = T_dist.rsample([self.S]) # shape: [self.S, batch_size, self.theta_dim]
        theta_samples = theta_samples.view([self.S * batch_size, self.theta_dim])

        # repeat for the number of samples
        x = x.repeat(self.S, 1, 1)
        x = x.view([self.S * batch_size, c, l])
        x = self.transformer(x, theta_samples)

        # theta samples: [S, bs, nr_params]
        return x, theta_samples, (theta_mu, beta)


class TimeseriesSTN(STN):
    def __init__(self, opt):
        super().__init__(opt)

    def init_localizer(self, opt):
        self.localization = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ###
            nn.Conv1d(64,
                      164, kernel_size=5),
            nn.BatchNorm1d(164),
            nn.ReLU(),
            ###
            nn.Conv1d(164, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ###
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout()
        )

        # Regressor for the mean
        self.fc_loc = nn.Sequential(
            nn.Linear(64, self.theta_dim)
        )

    def init_classifier(self, opt):
        self.classifier = TimeseriesClassifier(opt)

    # override the forward function since it's different for time series 
    def forward_localizer(self, x, x_high_res):
        theta = self.compute_theta(x)
        x = self.transformer(x, theta)
        return x, theta


class TimeseriesClassifier(nn.Module):
    def __init__(self, opt):
        super(TimeseriesClassifier, self).__init__()
        self.parameter_dict = self.load_specifications(opt)
        self.nr_classes = io.get_nr_classes_and_features(opt.dataset)[0]

        self.CNN = nn.Sequential(
            nn.Conv1d(1, self.parameter_dict['CNN_filters1'],
                      kernel_size=self.parameter_dict['CNN_kernel_size1']),
            nn.BatchNorm1d(self.parameter_dict['CNN_filters1']),
            nn.ReLU(),
            ###
            nn.Conv1d(self.parameter_dict['CNN_filters1'],
                      self.parameter_dict['CNN_filters2'], kernel_size=self.parameter_dict['CNN_kernel_size1']),
            nn.BatchNorm1d(self.parameter_dict['CNN_filters2']),
            nn.ReLU(),
            ###
            nn.Conv1d(self.parameter_dict['CNN_filters2'],
                      self.parameter_dict['CNN_filters3'], kernel_size=self.parameter_dict['CNN_kernel_size2']),
            nn.BatchNorm1d(self.parameter_dict['CNN_filters3']),
            nn.ReLU(),
            ###
            nn.Conv1d(self.parameter_dict['CNN_filters3'],
                      self.parameter_dict['CNN_filters4'], kernel_size=self.parameter_dict['CNN_kernel_size3']),
            nn.BatchNorm1d(self.parameter_dict['CNN_filters4']),
            nn.ReLU(),
            ###
            nn.Conv1d(self.parameter_dict['CNN_filters4'],
                      self.parameter_dict['CNN_filters5'], kernel_size=self.parameter_dict['CNN_kernel_size4']),
            nn.BatchNorm1d(self.parameter_dict['CNN_filters5']),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.25)
            # we experimented with this value in the original experiment, fix it to 0.5 for now
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.parameter_dict['CNN_filters5'], self.nr_classes)
        )

    # classifier network forward function
    def classifier(self, x):
        x = self.CNN(x)
        x_flat = x.view(-1, self.parameter_dict['CNN_filters5'])
        pred = self.fully_connected(x_flat)
        return pred

    def forward(self, x):
        # transform the input
        x_preds = self.classifier(x)
        # normalize outputs into probabilities
        x_probs = F.log_softmax(x_preds, dim=1)
        return x_probs

    def load_specifications(self, opt):
        if opt.model.lower() == 'cnn':
            parameter_dict = parameter_dict_timeseries_CNN
        elif opt.model.lower() == 'stn':
            parameter_dict = parameter_dict_timeseries_STN
        elif opt.model.lower() == 'pstn':
            parameter_dict = parameter_dict_timeseries_P_STN
        else:
            print('Pass valid model!')
        return parameter_dict
