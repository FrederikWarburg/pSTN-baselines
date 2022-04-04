import torch
import torch.nn as nn
from models.stn import STN
from models.pstn import PSTN
import torch.nn.functional as F

class CelebaPSTN(PSTN):
    def __init__(self, opt):

        # hyper parameters
        self.channels = 3
        self.input_size = 512

        super().__init__(opt)
        # num trainable params
        # 275 K

    def init_localizer(self, opt):
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_mu = nn.Sequential(
            nn.Linear(self.input_size, 100),
            nn.ReLU(True),
            nn.Linear(100, self.theta_dim)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_beta = nn.Sequential(
           nn.Linear(self.input_size, 100),
           nn.ReLU(True),
           nn.Linear(100, self.theta_dim),
           nn.Softplus()
        )

    def init_classifier(self, opt):
        self.classifier = CelebaClassifier(opt)


class CelebaSTN(STN):
    def __init__(self, opt):

        # hyper parameters
        self.channels = 3
        self.input_size = 512

        super().__init__(opt)
        # num trainable params
        # 223 K 

    def init_localizer(self, opt):
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
           nn.Linear(self.input_size, 100),
           nn.ReLU(True),
           nn.Linear(100, self.theta_dim)
        )

    def init_classifier(self, opt):
        self.classifier = CelebaClassifier(opt)


class CelebaClassifier(nn.Module):
    def __init__(self, opt):
        super(CelebaClassifier, self).__init__()
        # num trainable parameters
        # 154 K

        # hyper parameters
        self.dropout_rate = opt.dropout_rate
        self.num_classes = opt.num_classes
        self.feature_size = 640
        self.channels = 3

        # initialize a classifier for each branch
        self.CNN = nn.Sequential(
            nn.Conv2d(self.channels, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=5),
            nn.Dropout2d(self.dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(self.feature_size, 200)
        self.fc2 = nn.Linear(200, self.num_classes)

    def classifier(self, x):
        # get input dimensions
        batch_size, C, W, H = x.shape
        # print('classifier img shape:', x.shape)
        x = self.CNN(x)
        x = x.view(batch_size, self.feature_size)
        x = F.relu(self.fc1(x))
        # use drop out during training
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.classifier(x)
        probs = F.log_softmax(x , dim=1)
        return probs