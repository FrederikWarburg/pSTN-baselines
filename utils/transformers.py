import torch.nn.functional as F
import torch.nn as nn
from libcpab.cpab import Cpab
import torch
from torch.distributions.utils import _standard_normal


class diffeomorphic_transformation(nn.Module):
    def __init__(self, opt):
        if opt.xdim == 2:
            self.T = Cpab(tess_size=[3, 3], device='gpu', zero_boundary=True, backend='pytorch')
        elif opt.xdim == 1:
            self.T = Cpab(tess_size=[10], device='gpu', zero_boundary=True, backend='pytorch')
        else:
            raise NotImplementedError

    def forward(self, x, mean_params, sigma_params=None, sigma_prior=None):
        diffeo_params = self.make_diffeomorphic_parameters(mean_params, sigma_params, sigma_prior)
        x = self.T.transform_data(x, diffeo_params, outsize=x.shape[2:])
        return x

    def make_diffeomorphic_parameters(self, mean_params, sigma_params=None, sigma_prior=0.1):
        if sigma_params is not None:
            eps = _standard_normal(mean_params.shape, dtype=mean_params.dtype, device=mean_params.device)
            eps *= sigma_prior
            params = eps * sigma_params + mean_params
        else:
            params = mean_params
        return params


class affine_transformation(nn.Module):
    def __int__(self):
        pass

    def forward(self, x, mean_params, sigma_params=None, sigma_prior=None):
        affine_params = self.make_affine_parameters(mean_params, sigma_params, sigma_prior)
        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)
        return x

    def make_affine_parameters(self, mean_params, sigma_params=None, sigma_prior=0.1):

        if sigma_params is not None:
            eps = _standard_normal(mean_params.shape, dtype=mean_params.dtype, device=mean_params.device)
            eps *= sigma_prior
            mean_params = eps * sigma_params + mean_params

        if mean_params.shape[1] == 2:  # only perform crop - fix scale and rotation.
            theta = torch.zeros(mean_params.shape[0], device=mean_params.device)
            scale = 0.5 * torch.ones(mean_params.shape[0], device=mean_params.device)
            translation_x = mean_params[:, 0]
            translation_y = mean_params[:, 1]
        elif mean_params.shape[1] == 4:
            theta = mean_params[:, 0]
            scale = mean_params[:, 1]
            translation_x = mean_params[:, 2]
            translation_y = mean_params[:, 3]
        elif mean_params.shape[1] == 6:
            affine_matrix = mean_params.view([-1, 2, 3])
            return affine_matrix

        # theta is rotation angle in radians
        a = scale * torch.cos(theta)
        b = -scale * torch.sin(theta)
        c = translation_x

        d = scale * torch.sin(theta)
        e = scale * torch.cos(theta)
        f = translation_y

        param_tensor = torch.stack([a, b, c, d, e, f], dim=1)

        affine_matrix = param_tensor.view([-1, 2, 3])

        return affine_matrix
