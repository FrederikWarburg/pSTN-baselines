import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# possible base networks and their feature sizes
FEATURE_SIZES = {'inception': 1024,
                 'inception_v3': 2048,
                 'resnet50': 2048,
                 'resnet34': 512}


class CubClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # hyper parameters
        self.N = opt.N
        self.S = opt.test_samples
        self.dropout_rate = opt.dropout_rate
        self.train_samples = opt.train_samples
        self.test_samples = opt.test_samples
        self.feature_size = FEATURE_SIZES[opt.basenet.lower()]
        self.T = torch.ones(1, requires_grad=False)  # softmax temperature parameter

        # initialize a classifier for each branch
        self.model = nn.Module()
        for branch_ix in range(self.N):
            # create an instance of the classifier
            encoder = self.init_classifier_branch(opt)

            # add it to the model with an unique name
            self.model.add_module('branch_{}'.format(branch_ix), encoder)

        # create final layers
        self.bn = nn.BatchNorm1d(self.N * self.feature_size, eps=1e-05, momentum=0.1, affine=True)
        self.fc1 = nn.Linear(self.N * self.feature_size, opt.num_classes)

    def init_classifier_branch(self, opt):

        # Initialized base network
        if opt.basenet.lower() == 'inception':
            basenet = models.googlenet(pretrained=True)
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'inception_v3':
            basenet = models.inception_v3(pretrained=True)
            layers = list(basenet.children())[:-1]

        elif opt.basenet.lower() == 'resnet50':
            basenet = models.resnet50(pretrained=True)
            layers = list(basenet.children())[:-1]

        elif opt.basenet.lower() == 'resnet34':
            basenet = models.resnet34(pretrained=True)
            layers = list(basenet.children())[:-1]

        # add to encoder
        encoder = nn.Sequential(*layers)

        # we add the option to freeze some of the layers to avoid overfitting
        count = 0
        for i, child in enumerate(encoder.children()):
            for param in child.parameters():
                if count < opt.freeze_layers:
                    param.requires_grad = False

                count += 1

        return encoder

    def forward(self, x):

        # get input dimensions
        batch_size, C, W, H = x.shape

        # number of samples depends on training or testing setting
        self.S = self.train_samples if self.training else self.test_samples

        # calculate original batch size
        batch_size = batch_size // (self.N * self.S)

        # split data into original batch size dimensions
        xs = torch.stack(x.split([self.N] * self.S * batch_size))

        # calculate the features for each transformed image and store in features.
        # this correspond to concatenating the N descriptors as to decribed in the paper for S samples
        features = torch.empty(batch_size * self.S, self.feature_size * self.N, requires_grad=False, device=x.device)
        for branch_ix in range(self.N):
            x = self.model._modules['branch_{}'.format(branch_ix)].forward(xs[:, branch_ix, :, :, :])
            features[:, branch_ix * self.feature_size:(branch_ix + 1) * self.feature_size] = x.view(batch_size * self.S,
                                                                                                    self.feature_size)

        # make a classification based on the concatenated features
        x = self.bn(features)

        x = F.dropout(x, training=self.training)

        x = self.fc1(x)

        return F.log_softmax(x / self.T, dim=1)
