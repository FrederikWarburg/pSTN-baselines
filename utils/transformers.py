from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
#from libcpab.cpab import Cpab
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

    def forward(self, x, params):
        x = self.T.transform_data(x, params, outsize=x.shape[2:])
        return x


class AffineTransformer(nn.Module):
    def __int__(self):
        super().__init__()
        print("Use affine transformer")

    def forward(self, x, params):
        affine_params = make_affine_parameters(params)
        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid)
        return x


def make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y):
    # theta is rotation angle in radians
    a = scale_x * torch.cos(theta)
    b = - torch.sin(theta)
    c = translation_x

    d = torch.sin(theta)
    e = scale_y * torch.cos(theta)
    f = translation_y

    param_tensor = torch.stack([a, b, c, d, e, f], dim=-1)

    affine_matrix = param_tensor.view([-1, 2, 3])
    return affine_matrix

def make_affine_parameters(params):
    if params.shape[-1] == 2:  # only perform crop - fix scale and rotation.
        theta = torch.zeros([params.shape[0]], device=params.device)
        scale_x = 0.5 * torch.ones([params.shape[0]], device=params.device)
        scale_y = 0.5 * torch.ones([params.shape[0]], device=params.device)
        translation_x = params[:, 0]
        translation_y = params[:, 1]
        affine_matrix = make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

    elif params.shape[-1] == 3:  # crop with learned scale, isotropic, and tx/tx
        theta = torch.zeros([params.shape[0]], device=params.device)
        scale_x = params[:, 0]
        scale_y = params[:, 0]
        translation_x = params[:, 1]
        translation_y = params[:, 2]
        affine_matrix = make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

    elif params.shape[-1] == 4:  # "full afffine" with isotropic scale
        theta = params[:, 0]
        scale = params[:, 1]
        scale_x, scale_y = scale, scale
        translation_x = params[:, 2]
        translation_y = params[:, 3]
        affine_matrix = make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

    elif params.shape[-1] == 5:  # "full afffine" with anisotropic scale
        theta = params[:, 0]
        scale_x = params[:, 1]
        scale_y = params[:, 2]
        translation_x = params[:, 3]
        translation_y = params[:, 4]
        affine_matrix = make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

    elif params.shape[-1] == 6: # full affine, raw parameters
        affine_matrix = params.view([-1, 2, 3])

    return affine_matrix # [S * bs, 2, 3]
