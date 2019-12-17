import os
import torch
import numpy as np
import cv2
from torch import distributions


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denormalize(image):
    im = (image - np.min(image)) / (np.max(image) - np.min(image))
    return im

def add_bounding_boxes(image, theta, num_param):

    color = [(255, 0, 0) ,(0, 255, 0),(0, 0, 255), (255, 255, 0),(255, 0, 255),(0, 255, 255)]

    image *= 255
    im = image.astype(np.uint8).copy()

    theta = theta.reshape(-1).cpu().numpy()

    for i in range(len(theta)//num_param):

        if num_param == 2:
            w = int(0.5 * im.shape[0])
            h = int(0.5 * im.shape[1])

            x = int(w//2 - theta[i*num_param] * w * 2)
            y = int(h//2 - theta[i*num_param + 1] * h * 2)

            cv2.rectangle(im, (x,y),(x + w, y + h), color[i%len(color)], 5)

    return im

def make_affine_parameters(mean_params, sigma_params = None):

    if sigma_params != None:
        # combine mean and std and reshape
        gaussian = distributions.normal.Normal(0, 1)
        # split up the diagonal-cov. multivariate Gaussian into 1d Gaussians and perform sampling
        epsilon = gaussian.sample(sample_shape=torch.Size([S_total * self.theta_dim])).to(self.device)
        epsilon = epsilon.view(S_total, self.theta_dim)
        epsilon_times_sigma = torch.mul(sigma_q_tensor.to(self.device), epsilon)

        mean_params = (mean_theta_tensor.to(self.device) + epsilon_times_sigma).view(S_total, self.theta_dim)

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
