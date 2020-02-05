import os
import torch
import numpy as np
import cv2
from torch.distributions.utils import _standard_normal


def get_exp_name(opt):

    modelname = "d={}_m={}_b={}_n={}_p={}".format(opt.dataset, opt.model, opt.basenet, opt.N, opt.num_param)

    if opt.dataset.lower() == 'celeba':
        modelname += '_a=' + str(opt.target_attr)

    if opt.model.lower() == 'pstn':
        modelname += '_kl=' + opt.annealing
    else:
        modelname += '_kl=None'

    modelname += '_seed=' + str(opt.seed)
    modelname += '_s=' + str(opt.sigma)
    modelname += '_lr=' + str(opt.lr)

    if opt.model.lower() in ['stn','pstn']:
        modelname += '_lrloc=' + str(opt.lr_loc)
    else:
        modelname += '_lrloc=None'

    return modelname

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_affine_parameters(mean_params, sigma_params = None, sigma_prior = 0.1):

    if sigma_params is not None:
        eps = _standard_normal(mean_params.shape, dtype = mean_params.dtype, device=mean_params.device)
        eps *= sigma_prior
        mean_params = eps * sigma_params + mean_params

    if mean_params.shape[1] == 2: # only perform crop - fix scale and rotation.
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
