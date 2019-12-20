import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils import make_affine_parameters

class SimpleLocalizer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.N = opt.N
        self.num_param = opt.num_param

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

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
            self.fc2.bias.data.copy_(bias[:self.N].view(-1))
        if self.num_param == 6:
            self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,
                                                         0,1,0]*self.N, dtype=torch.float))


    def forward(self, x):

        batch_size, c, w, h = x.shape

        xs = self.localization(x)

        xs = xs.view(batch_size, -1)

        theta = self.fc_loc(xs)

        theta = theta.view(-1, self.N * self.num_param)

        x = x.repeat(self.N, 1, 1, 1)
        theta_upsample = torch.empty(batch_size*self.N, self.num_param, requires_grad=False, device=theta.device)
        for i in range(self.N):
            theta_upsample[i*batch_size:(i+1)*batch_size, :] = theta[:, i*self.num_param: (i+1)*self.num_param]

        affine_params = make_affine_parameters(theta_upsample)

        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)

        return x, theta, affine_params
