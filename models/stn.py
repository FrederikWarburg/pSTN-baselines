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
            from .inceptionlocalizer import InceptionSTN
            self.stn = InceptionSTN(opt)
        elif opt.basenet.lower() == 'simple':
            from .simplelocalizer import SimpleSTN
            self.stn = SimpleSTN(opt)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, self.num_param*self.N)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        if self.num_param == 2:
            # Center initialization
            #self.fc_loc[2].bias.data.copy_(torch.zeros(self.num_param*self.N, dtype=torch.float))

            # Tiling
            bias = torch.tensor([[-1,-1],[1,1],[1,-1],[-1,1]], dtype=torch.float)*0.5
            self.fc_loc[2].bias.data.copy_(bias[:self.N].view(-1))
        if self.num_param == 6:
            self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,
                                                         0,1,0]*self.N, dtype=torch.float))


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



