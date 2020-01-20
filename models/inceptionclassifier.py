import torchvision.models as models
import torch.nn as nn
import torch

FEATURE_SIZES = {'inception' : 1024,
                 'resnet50'  : 2048,
                 'resnet34'  : 512}

class InceptionClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.resume_ckpt != None:
            raise NotImplementedError
        else:
            self.N = opt.N
            self.feature_size = FEATURE_SIZES[opt.basenet.lower()]
            self.model = nn.Module()

            for branch_ix in range(self.N):
                encoder = self.init_classifier_branch(opt)
                self.model.add_module('branch_{}'.format(branch_ix), encoder)

            self.dropout = nn.Dropout(opt.dropout_rate)
            self.fc1 = nn.Linear(self.N * self.feature_size, opt.num_classes)

            self.logsoftmax = nn.LogSoftmax(dim=1)

    def init_classifier_branch(self, opt):

        # "Inception architecture with batch normalisation pretrained on ImageNet"
        if opt.basenet.lower() == 'inception':
            basenet = models.googlenet(pretrained=True)
            # "remove the last layer (1000-way ILSVRC classifier)"
            layers = list(basenet.children())[:-2]

        elif opt.basenet.lower() == 'resnet50':
            basenet = models.resnet50(pretrained = True)
            layers = list(basenet.children())[:-1]

        elif opt.basenet.lower() == 'resnet34':
            basenet = models.resnet34(pretrained = True)
            layers = list(basenet.children())[:-1]

        encoder = nn.Sequential(*layers)

        if opt.is_train:
            count = 0
            for i, child in enumerate(encoder.children()):
                for param in child.parameters():
                    if count < opt.freeze_layers:
                        param.requires_grad = False

                    count += 1

        return encoder

    def forward(self, x):

        batch_size = x.shape[0] // self.N
        xs = x.split([batch_size]*self.N)

        features = torch.empty(batch_size, self.feature_size*self.N, requires_grad = False, device=x.device)
        for branch_ix in range(self.N):
            x = self.model._modules['branch_{}'.format(branch_ix)].forward(xs[branch_ix])
            features[:, branch_ix*self.feature_size:(branch_ix+1)*self.feature_size] = x.view(batch_size, self.feature_size)

        x = self.dropout(features)
        x = self.fc1(x)

        x = self.logsoftmax(x)

        return x
