import os
import torch
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denormalize(image):
    im = (image - np.min(image)) / (np.max(image) - np.min(image))
    return im

def make_affine_parameters(mean_params):

    if mean_params.shape[1] == 2: # only perform crop - fix scale and rotation.
        theta = torch.ones(mean_params.shape[0])
        scale = 0.5 * torch.ones(mean_params.shape[0])
        translation_x = mean_params[:, 0]
        translation_y = mean_params[:, 1]
    else:
        theta = mean_params[:, 0]
        scale = mean_params[:, 1]
        translation_x = mean_params[:, 2]
        translation_y = mean_params[:, 3]

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
