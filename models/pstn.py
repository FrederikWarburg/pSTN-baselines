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
        self.num_param = opt.num_param

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

    def init_localizer(self, opt):
        if opt.basenet.lower() in ['inception', 'resnet50', 'resnet34', 'inception_v3']:
            from .inceptionlocalizer import InceptionPSTN
            self.pstn = InceptionPSTN(opt)
        elif opt.basenet.lower() == 'simple':
            from .simplelocalizer import SimplePSTN
            self.pstn = SimplePSTN(opt)

        elif opt.basenet.lower() == 'mnist':
            from .mnistlocalizer import MnistPSTN
            self.model = MnistPSTN(opt)
        elif opt.basenet.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesPSTN
            self.model = TimeseriesPSTN(opt)

    def init_classifier(self, opt):
        if opt.basenet.lower() in ['inception', 'resnet50', 'resnet34', 'inception_v3']:
            from .inceptionclassifier import InceptionClassifier
            self.classifier = InceptionClassifier(opt)
        elif opt.basenet.lower() == 'simple':
            from .simpleclassifier import SimpleClassifier
            self.classifier = SimpleClassifier(opt)

        elif opt.basenet.lower() == 'mnist':
            from .mnistclassifier import MnistClassifier
            self.model = MnistClassifier(opt)
        elif opt.basenet.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesClassifier
            self.model = TimeseriesClassifier(opt)

    def forward(self, x):

        batch_size, c, w, h = x.shape

        x, theta, _ = self.pstn(x)

        x = self.classifier(x)

        x = torch.stack(x.split([batch_size]*self.pstn.S))
        x = x.view(self.pstn.S, batch_size*self.num_classes)

        if self.training:
            mu, sigma = theta

            x = x.mean(dim=0)
            x = x.view(batch_size, self.num_classes)
            return (x, mu, sigma)
        else:
            x = torch.log(torch.tensor(1/self.pstn.S)) + torch.logsumexp(x, dim=0)
            x = x.view(batch_size, self.num_classes)

        return x



