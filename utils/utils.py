import os
import torch

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_affine_parameters(mean_params):
    theta = mean_params[0]
    scale = mean_params[1]
    translation_x = mean_params[2]
    translation_y = mean_params[3]
    # theta is rotation angle in radians
    a = scale * torch.cos(theta)
    b = -scale * torch.sin(theta)
    c = translation_x

    d = scale * torch.sin(theta)
    e = scale * torch.cos(theta)
    f = translation_y

    param_tensor = torch.stack([a, b, c, d, e, f])

    affine_matrix = param_tensor.view([-1, 2, 3])

    return affine_matrix
