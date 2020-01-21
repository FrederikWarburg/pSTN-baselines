from __future__ import print_function
import torch.nn as nn
import torch

class STN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.num_param = opt.num_param
        self.N = opt.N

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

    def init_localizer(self, opt):
        if opt.basenet.lower() in ['inception', 'resnet50', 'resnet34', 'inception_v3']:
            from .inceptionlocalizer import InceptionSTN
            self.stn = InceptionSTN(opt)
        elif opt.basenet.lower() == 'simple':
            from .simplelocalizer import SimpleSTN
            self.stn = SimpleSTN(opt)

    def init_classifier(self, opt):

        if opt.basenet.lower() in ['inception', 'resnet50', 'resnet34', 'inception_v3']:
            from .inceptionclassifier import InceptionClassifier
            self.classifier = InceptionClassifier(opt)
        elif opt.basenet.lower() == 'simple':
            from .simpleclassifier import SimpleClassifier
            self.classifier = SimpleClassifier(opt)

    def forward(self, x):

        x, _, _ = self.stn(x)

        x = self.classifier(x)

        return x



