from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from  .parameter_dicts import *

class MnistClassifier(nn.Module):
    def __init__(self, opt):
        super(MnistClassifier, self).__init__()
        self.parameter_dict = load_specifications(opt)

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


def load_specifications(opt):
    print('loading parameter dict')
    if opt.model.lower() == 'cnn':
        # if opt.dataset == 'random_rotation_mnist':
        #     print('not implemented yet')
            # parameter_dict = parameter_dict_classifier_rotMNIST_CNN
        if opt.dataset.lower() == "random_placement_fashion_mnist" and not opt.freeze_classifier:
            parameter_dict = parameter_dict_classifier_RandomPlacementMNIST_CNN
        elif "mnist" in opt.dataset.lower():
            parameter_dict = parameter_dict_classifier_MNIST_CNN

    if opt.model.lower() in ['stn']:
        # if opt.dataset == 'random_rotation_mnist':
        #     print('not implemented yet')
            # parameter_dict = parameter_dict_classifier_rotMNIST_STN
        if opt.dataset == "random_placement_fashion_mnist" and not opt.freeze_classifier:
            parameter_dict = parameter_dict_classifier_RandomPlacementMNIST_STN
        elif  "mnist" in opt.dataset.lower():
            parameter_dict = parameter_dict_classifier_MNIST_STN

    elif opt.model.lower() == 'pstn':
        # if opt.dataset == 'random_rotation_mnist':
        #     print('not implemented yet')
        #     parameter_dict = parameter_dict_classifier_rotMNIST_STN
        if opt.dataset == "random_placement_fashion_mnist" and not opt.freeze_classifier:
            parameter_dict = parameter_dict_classifier_RandomPlacementMNIST_P_STN
        elif "mnist" in opt.dataset.lower():
            parameter_dict = parameter_dict_classifier_MNIST_P_STN

    else:
        print('Pass valid model!')
    return parameter_dict
