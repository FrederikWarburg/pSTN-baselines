import os
import torch
import numpy as np
import cv2

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denormalize(image):
    im = (image - np.min(image)) / (np.max(image) - np.min(image))
    return im

def add_bounding_boxes(im, theta, num_param):
    for i in range(len(theta)//num_param):

        if num_param == 2:
            w = 0.5 * input.shape[2]
            h = 0.5 * input.shape[3]
            x = theta[i*num_param] - w//2
            y = theta[i*num_param + 1] - h//2

            # Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
            cv2.rectangle(im, (x,y),(w,h),(0,255,0),2)

    return im

def make_affine_parameters(mean_params):

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
