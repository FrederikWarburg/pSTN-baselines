import torch
import torch.nn as nn
from torch import distributions
from utils.transformers import init_transformer
from models.stn import STN
from models.pstn import PSTN
import torch.nn.functional as F


parameter_dict_P_STN = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 14 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 38,
    'localizer_filters1': 8,
    'localizer_filters2': 14,
    'color_channels': 1
}

parameter_dict_STN = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 18 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 50,
    'localizer_filters1': 12,
    'localizer_filters2': 18,
    'color_channels': 1
}

parameter_dict_classifier_MNIST_CNN = {
    'nr_target_classes': 10,
    'CNN_filters1': 12,
    'CNN_filters2': 24,
    'CNN_kernel_size': 5,
    'resulting_size_classifier': 24 * 4 * 4, # default is mnist; we override for random_placement_mnist below
    'hidden_layer_classifier': 52,
    'color_channels': 1
}

parameter_dict_classifier_MNIST_STN = {
    'nr_target_classes': 10,
    'CNN_filters1': 10,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'resulting_size_classifier': 320,
    'hidden_layer_classifier': 50,
    'color_channels': 1
}

parameter_dict_classifier_MNIST_P_STN = {
    'nr_target_classes': 10,
    'CNN_filters1': 10,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'loc_kernel_size': 5,
    'resulting_size_classifier': 320,
    'hidden_layer_classifier': 50,
    'color_channels': 1
}

class MnistPSTN(PSTN):
    def __init__(self, opt):
        self.parameter_dict = self.load_specifications(opt)
        super().__init__(opt)

        self.channels = 1

    def init_localizer(self, opt):
        # Spatial transformer localization-network
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
        # Regressor for the diffeomorphic param's
        elif opt.transformer_type == 'diffeomorphic':
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

        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc_beta = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                          self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(False),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim),
                # add activation function for positivity
                nn.Softplus())

    def init_classifier(self, opt):
        self.classifier = MnistClassifier(opt)

    def load_specifications(self, opt):
    
        parameter_dict = parameter_dict_P_STN
        if opt.dataset.lower() == 'random_placement_mnist':
            parameter_dict['resulting_size_localizer'] = 6174
        return parameter_dict


class MnistSTN(STN):
    def __init__(self, opt):
        self.parameter_dict = self.load_specifications(opt)
        super().__init__(opt)

        self.test_samples = opt.test_samples
        self.channels = 1

    def init_localizer(self, opt):

        # Spatial transformer localization-network
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
        elif opt.transformer_type == 'diffeomorphic':
            self.fc_loc = nn.Sequential(
                nn.Linear(self.parameter_dict['resulting_size_localizer'],
                          self.parameter_dict['hidden_layer_localizer']),
                nn.ReLU(True),
                nn.Linear(self.parameter_dict['hidden_layer_localizer'], self.theta_dim)
            )

    def init_classifier(self, opt):
        self.classifier = MnistClassifier(opt)

    def load_specifications(self, opt):

        parameter_dict = parameter_dict_STN
        if opt.dataset.lower() == 'random_placement_mnist':
            parameter_dict['resulting_size_localizer'] = 7938

        return parameter_dict


class MnistClassifier(nn.Module):
    def __init__(self, opt):
        super(MnistClassifier, self).__init__()
        self.parameter_dict = self.load_specifications(opt)

        self.CNN = nn.Sequential(
            # first conv layer
            nn.Conv2d(
                self.parameter_dict['color_channels'], self.parameter_dict['CNN_filters1'],
                kernel_size=self.parameter_dict['CNN_kernel_size']),
            nn.MaxPool2d(2, stride=2),  # 2 for 28 x 28 datasets
            nn.ReLU(True),
            # second conv layer
            nn.Conv2d(
                self.parameter_dict['CNN_filters1'], self.parameter_dict['CNN_filters2'],
                kernel_size=self.parameter_dict['CNN_kernel_size']),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fully_connected = nn.Sequential(
            # first fully connected layer
            nn.Linear(self.parameter_dict['resulting_size_classifier'], self.parameter_dict['hidden_layer_classifier']),
            nn.ReLU(True),
            nn.Dropout(),
            # second fully connected layer
            nn.Linear(self.parameter_dict['hidden_layer_classifier'], self.parameter_dict['nr_target_classes']))

    def classifier(self, x):
        x = self.CNN(x)
        x_flat = x.view(-1, self.parameter_dict['resulting_size_classifier'])
        pred = self.fully_connected(x_flat)
        return pred

    def forward(self, x):
        x = self.classifier(x)
        probs = F.log_softmax(x , dim=1)
        return probs

    def load_specifications(self, opt):
        if opt.model.lower() == 'cnn':
            parameter_dict = parameter_dict_classifier_MNIST_CNN
            if opt.dataset.lower() == 'random_placement_mnist':
                 parameter_dict['resulting_size_classifier'] = 24 * 21 * 21
        elif opt.model.lower() in ['stn']:
            parameter_dict = parameter_dict_classifier_MNIST_STN
            if opt.dataset.lower() == 'random_placement_mnist':
                parameter_dict['resulting_size_classifier'] = 20 * 21 * 21
        elif opt.model.lower() == 'pstn':
            parameter_dict = parameter_dict_classifier_MNIST_P_STN
            if opt.dataset.lower() == 'random_placement_mnist':
                parameter_dict['resulting_size_classifier'] = 20 * 21 * 21
        else:
            print('Pass valid model!')
        return parameter_dict
