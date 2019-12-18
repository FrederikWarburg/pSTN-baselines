import os
import torch
import numpy as np
import cv2
from torch.distributions.utils import _standard_normal

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denormalize(image):
    im = (image - np.min(image)) / (np.max(image) - np.min(image))
    return im

def add_bounding_boxes(image, affine_params, num_branches, num_samples, mode_ = 'crop'):

    color = [(255, 0, 0) ,(0, 255, 0),(0, 0, 255), (255, 255, 0),(255, 0, 255),(0, 255, 255)]

    image *= 255
    im = image.astype(np.uint8).copy()

    if mode_ == 'crop':
        w = int(im.shape[0])
        h = int(im.shape[1])

    for i in range(num_branches):

        for j in range(num_samples):

            if mode_ == 'crop':
                print(affine_params)
                print(affine_params.shape)
                print(i, num_samples, j)
                x = affine_params[i*num_samples+j, 0, 2]
                y = affine_params[i*num_samples+j, 1, 2]
                print(x, y)
                x = int(x*w//2 + w//4)
                y = int(y*h//2 + h//4)
                print(x, y)
                cv2.rectangle(im, (x,y),(x + w//2, y + h//2), color[i%len(color)], 5)

    return im


def make_affine_parameters(mean_params, sigma_params = None):

    if sigma_params is not None:
        eps = _standard_normal(mean_params.shape, dtype = mean_params.dtype, device=mean_params.device)
        mean_params = eps * sigma_params + mean_params

    if mean_params.shape[1] == 2: # only perform crop - fix scale and rotation.
        theta = torch.zeros(mean_params.shape[0], device=mean_params.device)
        scale = 0.5 * torch.ones(mean_params.shape[0], device=mean_params.device)
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
