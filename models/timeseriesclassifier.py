from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from utils import timeseries_io as io

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
                          self.parameter_dict['CNN_filters2'], kernel_size=self.parameter_dict['CNN_filters2']),
                nn.BatchNorm1d(self.parameter_dict['CNN_filters2']),
                nn.ReLU(),
                ###
                nn.Conv1d(self.parameter_dict['CNN_filters2'],
                          self.parameter_dict['CNN_filters3'], kernel_size=self.parameter_dict['CNN_filters3']),
                nn.BatchNorm1d(self.parameter_dict['CNN_filters3']),
                nn.ReLU(),
                ###
                nn.Conv1d(self.parameter_dict['CNN_filters3'],
                          self.parameter_dict['CNN_filters4'], kernel_size=self.parameter_dict['CNN_filters4']),
                nn.BatchNorm1d(self.parameter_dict['CNN_filters4']),
                nn.ReLU(),
                ###
                nn.Conv1d(self.parameter_dict['CNN_filters4'],
                          self.parameter_dict['CNN_filters5'], kernel_size=self.parameter_dict['CNN_filters5']),
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
        x = self.large_CNN(x)
        x_flat = x.view(-1, self.parameter_dict['CNN_filters5'])
        pred = self.large_fully_connected(x_flat)
        return pred

    def forward(self, x, epoch):
        # get batch size from data
        self.batch_size = x.shape[0]
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
        elif opt.model.lower() == 'p_stn':
            parameter_dict = parameter_dict_timeseries_P_STN
        else:
            print('Pass valid model!')
        return parameter_dict
