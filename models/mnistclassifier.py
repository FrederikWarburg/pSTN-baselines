from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

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

class MnistClassifier(nn.Module):
    def __init__(self, opt):
        super(MnistClassifier, self).__init__()
        self.parameter_dict = self.load_specifications(opt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            if 'random_placement' in opt.dataset.lower(): # random placement MNIST & random placement fashion MNIST
                parameter_dict['resulting_size_classifier'] = 20 * 21 * 21
                parameter_dict['CNN_filters2'] = 20

        elif opt.model.lower() in ['stn']:
            parameter_dict = parameter_dict_classifier_MNIST_STN
            if  'random_placement' in  opt.dataset.lower():
                parameter_dict['resulting_size_classifier'] = 20 * 21 * 21
        elif opt.model.lower() == 'pstn':
            parameter_dict = parameter_dict_classifier_MNIST_P_STN
            if  'random_placement' in opt.dataset.lower():
                parameter_dict['resulting_size_classifier'] = 20 * 21 * 21
        else:
            print('Pass valid model!')
        return parameter_dict
