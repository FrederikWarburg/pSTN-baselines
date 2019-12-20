from __future__ import print_function
import torch.nn as nn

import torch.nn.functional as F
from utils.utils import make_affine_parameters
import torchvision.models as models
import torch

class PSTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.num_classes = opt.num_classes
        self.num_samples = opt.test_samples
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.num_param = opt.num_param

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

    def init_localizer(self, opt):
        if opt.basenet.lower() == 'inception':
            from .inceptionlocalizer import InceptionPSTN
            self.pstn = InceptionPSTN(opt)
        elif opt.basenet.lower() == 'simple':
            from .simplelocalizer import SimplePSTN
            self.pstn = SimplePSTN(opt)

    def init_classifier(self, opt):
        if opt.basenet.lower() == 'inception':
            from .inceptionclassifier import InceptionClassifier
            self.classifier = InceptionClassifier(opt)
        elif opt.basenet.lower() == 'simple':
            from .simpleclassifier import SimpleClassifier
            self.classifier = SimpleClassifier(opt)

    def forward(self, x):

        if self.training:
            self.pstn.S = self.train_samples
        else:
            self.pstn.S = self.test_samples

        x, theta, _ = self.pstn(x)

        x = self.classifier(x)

        x = x.view(-1, self.num_samples, self.num_classes)

        if self.training:
            mu, sigma = theta
            x = (x.mean(dim=1), mu, sigma)
        else:
            x = torch.log(torch.tensor(1/self.num_samples)) + torch.logsumexp(x, dim=1)

        return x



