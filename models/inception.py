import torchvision.models as models
import torch.nn as nn

class InceptionClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.resume_ckpt != None:
            raise NotImplementedError
        else:
            # "Inception architecture with batch normalisation pretrained on ImageNet"
            inception = models.googlenet(pretrained=True)

            # "remove the last layer (1000-way ILSVRC classifier)"
            layers = list(inception.children())[:-2]

            self.encoder = nn.Sequential(*layers)

            if opt.is_train:
                count = 0
                for i, child in enumerate(self.encoder.children()):
                    for param in child.parameters():
                        if count < opt.freeze_layers:
                            param.requires_grad = False

                        count += 1

            self.dropout = nn.Dropout(opt.dropout_rate)
            self.fc1 = nn.Linear(1024, opt.num_classes)

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(-1, 1024)

        x = self.dropout(x)
        x = self.fc1(x)

        return x