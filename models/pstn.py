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

        batch_size, c, w, h = x.shape

        x, theta, _ = self.pstn(x)

        # [im1_crop1, im1_crop2]
        # [im2_crop1, im2_crop2],
        # [im1_crop1, im1_crop2]
        # [im2_crop1, im2_crop2]
        print(x.shape)
        x = torch.stack(x.split(self.pstn.S))
        print(x.shape)
        x = torch.stack(x.split(self.pstn.N))
        print(x.shape)

        import matplotlib.pyplot as plt
        for i in range(len(x))
            im = xs[0]
            print(im.shape)
            im = im.detach().numpy()[0]
            print(im.shape)
            plt.imshow(im)
            plt.show()

        exit()

        x = self.classifier(x)

        x = x.view(-1, self.num_classes,  self.pstn.S)

        if self.training:
            mu, sigma = theta
            x = (x.sum(dim=0), mu, sigma)
        else:
            x = torch.log(torch.tensor(1/self.pstn.S)) + torch.logsumexp(x, dim=0)


        return x



