from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcpab.cpab import Cpab
from torch.distributions.utils import _standard_normal

def init_transformer(opt):
    if opt.transformer_type == 'affine':
        return AffineTransformer(), opt.N * opt.num_param
    elif opt.transformer_type == 'diffeomorphic':
        transformer = DiffeomorphicTransformer(opt)
        theta_dim = transformer.T.get_theta_dim()
        opt.num_param = theta_dim
        return transformer, theta_dim * opt.N


class DiffeomorphicTransformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        print("Use diffeomorphic transformer")

        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if opt.xdim == 2:
            self.T = Cpab(tess_size=[3, 3], device=self.device, zero_boundary=True, backend='pytorch')
        elif opt.xdim == 1:
            self.T = Cpab(tess_size=[10], device=self.device, zero_boundary=True, backend='pytorch')
        else:
            raise NotImplementedError

        self.sigma_p = opt.sigma_p

    def forward(self, x, mean_params, std_params=None):
        diffeo_params = self.make_diffeomorphic_parameters(mean_params, std_params)
        x = self.T.transform_data(x, diffeo_params, outsize=x.shape[2:])
        return x, diffeo_params

    def make_diffeomorphic_parameters(self, mean_params, std_params=None):
        if std_params is not None:
            eps = _standard_normal(mean_params.shape, dtype=mean_params.dtype, device=mean_params.device)
            eps *= self.sigma_p
            params = eps * std_params + mean_params
        else:
            params = mean_params
        return params


class AffineTransformer(nn.Module):
    def __int__(self):
        super().__init__()
        print("Use affine transformer")

    def forward(self, x, mean_params, std_params=None):
        # 1. sample parameters from mean + std 
        params = self.sample_parameters(mean_params, std_params)
        # 2. convert to affine matrix
        affine_params = make_affine_parameters(params)

        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)

        return x, params

    def sample_parameters(self, mean_params, std_params=None):
        if std_params is None:
            params = mean_params
        else:
            eps = _standard_normal(
                mean_params.shape, dtype=mean_params.dtype, device=mean_params.device)
            params = eps * std_params + mean_params
        return params


def make_affine_matrix(theta, scale, translation_x, translation_y):
    # theta is rotation angle in radians
    a = scale * torch.cos(theta)
    b = -scale * torch.sin(theta)
    c = translation_x

    d = scale * torch.sin(theta)
    e = scale * torch.cos(theta)
    f = translation_y

    param_tensor = torch.stack([a, b, c, d, e, f], dim=-1)

    affine_matrix = param_tensor.view([-1, 2, 3])
    return affine_matrix

def make_affine_parameters(params):
    if params.shape[1] == 2:  # only perform crop - fix scale and rotation.
        theta = torch.zeros(params.shape[0], device=params.device)
        scale = 0.5 * torch.ones(params.shape[0], device=params.device)
        translation_x = params[:, 0]
        translation_y = params[:, 1]
        affine_matrix = make_affine_matrix(theta, scale, translation_x, translation_y)

    elif params.shape[1] == 4:
        theta = params[:, 0]
        scale = params[:, 1]
        translation_x = params[:, 2]
        translation_y = params[:, 3]
        affine_matrix = make_affine_matrix(theta, scale, translation_x, translation_y)

    elif params.shape[1] == 6:
        affine_matrix = params.view([-1, 2, 3])

    return affine_matrix
