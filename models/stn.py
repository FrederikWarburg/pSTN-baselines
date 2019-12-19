from __future__ import print_function
import torch.nn as nn

class STN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Spatial transformer localization-network
        self.init_localizer(opt)
        self.init_classifier(opt)

    def init_localizer(self, opt):
        if opt.basenet.lower() == 'inception':
            from .inceptionlocalizer import InceptionLocalizer
            self.stn = InceptionLocalizer(opt)
        elif opt.basenet.lower() == 'simple':
            from .simplelocalizer import SimpleLocalizer
            self.stn = SimpleLocalizer(opt)

    def init_classifier(self, opt):

        if opt.basenet.lower() == 'inception':
            from .inceptionclassifier import InceptionClassifier
            self.classifier = InceptionClassifier(opt)
        elif opt.basenet.lower() == 'simple':
            from .simpleclassifier import SimpleClassifier
            self.classifier = SimpleClassifier(opt)

    def forward(self, x):

        x, _, _ = self.stn(x)

        x = self.classifier(x)

        return x



