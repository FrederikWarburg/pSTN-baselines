import torch
import torch.nn as nn
from torch import distributions
from models.parameter_dicts import load_specifications_localizer, load_specifications_classifier
from utils.transformers import init_transformer
from models.stn import STN
from models.pstn import PSTN
import torch.nn.functional as F


class MnistPSTN(PSTN):
    def __init__(self, opt):
        self.parameter_dict = load_specifications_localizer(opt)
        super().__init__(opt)

    def init_localizer(self, opt):
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

    def init_classifier(self, opt):
        self.classifier = MnistClassifier(opt)


class MnistSTN(STN):
    def __init__(self, opt):
        self.parameter_dict = load_specifications_localizer(opt)
        super().__init__(opt)

        self.channels = 1

    def init_localizer(self, opt):

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

    def init_classifier(self, opt):
        self.classifier = MnistClassifier(opt)


class MnistClassifier(nn.Module):
    def __init__(self, opt):
        super(MnistClassifier, self).__init__()
        self.parameter_dict = load_specifications_classifier(opt)
        self.CNN, self.fully_connected = self.make_classifier(opt)

    def classifier(self, x):
        x = self.CNN(x)
        x_flat = x.view(-1, self.parameter_dict['resulting_size_classifier'])
        pred = self.fully_connected(x_flat)
        return pred

    def forward(self, x):
        x = self.classifier(x)
        probs = F.log_softmax(x , dim=1)
        return probs
    
    def make_classifier(self, opt):
        if not 'nn' in opt.modeltype_classifier: # the convolutional model types
            CNN = nn.Sequential(
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
            fully_connected = nn.Sequential(
                # first fully connected layer
                nn.Linear(self.parameter_dict['resulting_size_classifier'], self.parameter_dict['hidden_layer_classifier']),
                nn.ReLU(True),
                nn.Dropout(),
                # second fully connected layer
                nn.Linear(self.parameter_dict['hidden_layer_classifier'], self.parameter_dict['nr_target_classes']))
        else: 
            CNN = nn.Sequential() # no CNN in any of these
            if opt.modeltype_classifier == 'nn1_classifier': # 1 layer classifier
                fully_connected = nn.Sequential(
                    nn.Linear(self.parameter_dict['resulting_size_classifier'], self.parameter_dict['nr_target_classes']))

            elif opt.modeltype_classifier == 'nn2_classifier': # 2 layer classifier
                fully_connected = nn.Sequential(
                    nn.Linear(self.parameter_dict['resulting_size_classifier'], 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # second fully connected layer
                    nn.Linear(128, self.parameter_dict['nr_target_classes'])
                    )
            elif opt.modeltype_classifier == 'nn3_classifier': # 3 layer classifier
                fully_connected = nn.Sequential(
                    nn.Linear(self.parameter_dict['resulting_size_classifier'], 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # second fully connected layer
                    nn.Linear(256, 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # third
                    nn.Linear(128, self.parameter_dict['nr_target_classes'])
                    )          
            elif opt.modeltype_classifier == 'nn4_classifier': # 4 layer classifier
                fully_connected = nn.Sequential(
                    nn.Linear(self.parameter_dict['resulting_size_classifier'], 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # second fully connected layer
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # third
                    nn.Linear(256, 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # fourth
                    nn.Linear(128, self.parameter_dict['nr_target_classes'])
                    )      
            elif opt.modeltype_classifier == 'nn5_classifier': # 4 layer classifier
                fully_connected = nn.Sequential(
                    nn.Linear(self.parameter_dict['resulting_size_classifier'], 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # second fully connected layer
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # third
                    nn.Linear(256, 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # fourth
                    nn.Linear(128, 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # fifth
                    nn.Linear(128, self.parameter_dict['nr_target_classes'])
                    )    

        return CNN, fully_connected