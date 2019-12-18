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

def add_bounding_boxes(image, theta_mu, theta_sigma, num_param, num_samples = 1):

    color = [(255, 0, 0) ,(0, 255, 0),(0, 0, 255), (255, 255, 0),(255, 0, 255),(0, 255, 255)]

    image *= 255
    im = image.astype(np.uint8).copy()
    w = int(0.5 * im.shape[0])
    h = int(0.5 * im.shape[1])

    theta_mu = theta_mu.reshape(-1).cpu().numpy()
    theta_sigma = theta_sigma.reshape(-1).cpu().numpy()

    for i in range(len(theta_mu)//num_param):

        for j in range(num_samples):


            eps = np.random.standard_normal(2)

            x = eps[0] * theta_sigma[i*num_param] + theta_mu[i*num_param]
            y = eps[1] * theta_sigma[i*num_param + 1] + theta_mu[i*num_param + 1]

            if num_param == 2:
                x = int(w//2 - x * w * 2)
                y = int(h//2 - y * h * 2)

                cv2.rectangle(im, (x,y),(x + w, y + h), color[i%len(color)], 5)

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
