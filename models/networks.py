import torchvision.models as models
import torch.nn as nn
import torch

class InceptionClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.resume_ckpt != None:
            raise NotImplementedError
        else:
            # "Inception architecture with batch normalisation pretrained on ImageNet"
            inception = models.googlenet(pretrained = True)

            # "remove the last layer (1000-way ILSVRC classifier)"
            layers = list(inception.children())[:-1]
            self.encoder = nn.Sequential(*layers)

            self.fc1 = nn.Linear(1024,200)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.encoder(x)
        x = torch.squeeze(x)

        x = self.fc1(x)
        x = self.softmax(x)

        return x

def inceptionv3(resume_ckpt = None, output_dim = 200, N=1):

    if resume_ckpt != None:
        raise NotImplementedError
    else:
        # "Inception architecture with batch normalisation pretrained on ImageNet"
        inception = models.googlenet(pretrained = True) # inceptionnet v2 (I think this is the one used in the paper)

        # "remove the last layer (1000-way ILSVRC classifier)"
        layers = list(inception.children())[:-1]
        classifier = nn.Sequential(*layers)

        # "remove the last pooling layer to preserve the spatial information"
        layers = list(inception.children())[:-3]

        # add three weight layers

        # 1) "1 x 1 conv layer to reduce the number of feature channels from 1024 to 128"
        layers.append(nn.Conv2d(1024, 128, 1))

        # 2) "fully-connected layer with 128-D output"
        layers.append(nn.ReLU(nn.Linear(128, 128)))

        # 3) #fully connected layer with 2N-D output, where N is the number of transformers.
        layers.append(nn.Linear(128, 2*N))

        localizer = nn.Sequential(*layers)

        model = STN(localizer, classifier, N)

    return model



class STN(nn.Module):
    def __init__(self, localizer, describer, N):
        super().__init__()

        self.localizer = localizer
        self.describer = describer

        self.fc1 = nn.Linear(N*1024, 200)
        self.softmax = nn.Softmax(200)

    def localize(self, x):
        return self.localizer(x)

    def sample(self, x, theta):

        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        print("rois found to be of size:{}".format(rois.size()))
        return rois, affine_grid_points

    def describe(self, x):
        return self.describer(x)

    def classifier(self, x):

        x = self.softmax(self.fc1(x))

        return x

    def forward(self, x):

        theta = self.locaize(x)
        crops = self.sample(x, theta)
        descriptors = self.describe(crops)
        predictions = self.classify(descriptors)

        return predictions

