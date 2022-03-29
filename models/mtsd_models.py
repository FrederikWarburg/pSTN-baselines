from builtins import breakpoint
import torch
import torch.nn as nn
from models.stn import STN
from models.pstn import PSTN
import torch.nn.functional as F
import torchvision.models as models


class MtsdPSTN(PSTN):
    def __init__(self, opt):

        # hyper parameters
        self.channels = 3
        self.input_size = 128

        super().__init__(opt)
        # num trainable params
        # 275 K
        
    def init_localizer(self, opt):
        # initialize a classifier for each branch
        net = models.resnet18(pretrained=True)
        self.localization = nn.Sequential(*[net.conv1,net.bn1, net.relu, net.maxpool, net.layer1,net.layer2, net.layer3, net.layer4])

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_mu = nn.Sequential(nn.Linear(512, self.theta_dim))
        self.fc_loc_beta = nn.Sequential(
           nn.Linear(512, self.theta_dim),
           nn.Softplus()
        )

        self.pool = net.avgpool
    
    def init_classifier(self, opt):
        self.classifier = MtsdClassifier(opt)

    def compute_theta_beta(self, x):
        bs, c, h, w = x.shape
        x = self.localization(x)
        x = self.pool(x).view(bs, -1)
        mu = self.fc_loc_mu(x)
        beta = self.fc_loc_beta(x)

        return mu, beta


class MtsdSTN(STN):
    def __init__(self, opt):

        # hyper parameters
        self.channels = 3
        self.input_size = 128

        super().__init__(opt)
        # num trainable params
        # 223 K 

    def init_localizer(self, opt):
        # initialize a classifier for each branch
        net = models.resnet18(pretrained=True)
        self.localization = nn.Sequential(*[net.conv1,net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4])
        self.pool = net.avgpool

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(512, self.theta_dim))

    def init_classifier(self, opt):
        self.classifier = MtsdClassifier(opt)

    def compute_theta(self, x):
        bs, c, h, w = x.shape
        x = self.localization(x)
        x = self.pool(x).view(bs, -1)
        x = self.fc_loc(x)
        return x


class MtsdClassifier(nn.Module):
    def __init__(self, opt):
        super(MtsdClassifier, self).__init__()
        # num trainable parameters
        # 154 K

        # hyper parameters
        self.dropout_rate = opt.dropout_rate
        self.num_classes = opt.num_classes
        self.feature_size = 128

        self.channels = 3

        # initialize a classifier for each branch
        self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(512, self.num_classes)

    def classifier(self, x):
        return self.net(x)

    def forward(self, x, high_res=None):
        x = self.classifier(x)
        probs = F.log_softmax(x , dim=1)
        return probs