import torch
import numpy as np
import torchvision
from torchvision import transforms
import cv2

def convert_image_np(inp, dataset = 'mnist'):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if 'mnist' in dataset:
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

    return inp


def normalize(images):


    return images / torch.max(images)

def visualize_stn(model, data, opt):

    with torch.no_grad():

        data = data[:16] # just visualize the first 16

        in_grid = convert_image_np(torchvision.utils.make_grid(data.cpu()), opt.dataset.lower())
        in_grid = (in_grid*255).astype(np.uint8)
        in_grid = np.transpose(in_grid, (2,0,1))

        if opt.model.lower() == 'cnn':
            return in_grid, None, None, None
        elif opt.model.lower() == 'stn':
            transformed_input_tensor, theta, affine_params = model.stn(data)
        elif opt.model.lower() == 'pstn':
            transformed_input_tensor, theta, affine_params = model.pstn(data)

        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor.cpu()),opt.dataset.lower())
        out_grid = (out_grid*255).astype(np.uint8)
        out_grid = np.transpose(out_grid, (2,0,1))

        bbox_images = visualize_bbox(data.cpu(), affine_params, opt)

        # Plot the results side-by-side
    return in_grid, out_grid, theta, bbox_images


def visualize_bbox(data, affine_params, opt):

    batch_size = data.shape[0]
    affine_params = torch.stack(affine_params.split([opt.N]*opt.test_samples*batch_size))
    sorted_params = torch.zeros(batch_size, opt.test_samples, opt.N, 2, 3, dtype=affine_params.dtype)
    for i in range(len(affine_params)):
        # im, sample, N, 2, 3
        sorted_params[i % batch_size, i // batch_size, :, :, :] = affine_params[i, :, :]

    images = []
    for j, (im, params) in enumerate(zip(data, sorted_params)):

        im = np.transpose(im.numpy(),(1,2,0))
        im = denormalize(im)

        if im.shape[2] == 1:
            im = np.stack((im[:,:,0],)*3, axis=-1)

        if torch.isnan(params).any(): continue

        im = add_bounding_boxes(im, params, opt.N, opt.test_samples, mode_= 'crop', heatmap = opt.heatmap)

        images.append(np.transpose(im,(2,0,1)))

    if len(images) > 0:

        images = torch.FloatTensor(images)
        if 'mnist' in opt.dataset.lower():
            images = convert_image_np(torchvision.utils.make_grid(images-0.3), opt.dataset.lower())
            images = (images*255).astype(np.uint8)
            images = np.transpose(images, (2,0,1))
        else:
            images = torchvision.utils.make_grid(images, normalize = True)

        return images

    return None

def denormalize(image):
    im = (image - np.min(image)) / (np.max(image) - np.min(image))
    return im

def add_bounding_boxes(image, affine_params, num_branches, num_samples, mode_ = 'crop', heatmap = True):

    color = [(255, 0, 0) ,(0, 255, 0),(0, 0, 255), (255, 255, 0),(255, 0, 255),(0, 255, 255)]

    image *= 255
    im = image.astype(np.uint8).copy()

    if mode_ == 'crop':
        w = int(im.shape[0])
        h = int(im.shape[1])

    for j in range(num_samples):
        for i in range(num_branches):
            if mode_ == 'crop':
                x = affine_params[j, i, 0, 2]
                y = affine_params[j, i, 1, 2]

                # define bbox by top left corner and define coordinates system with origo in top left corner
                x = int(x*w//2 + w//4)
                y = int(y*h//2 + h//4)

                if heatmap:

                    overlay = im.copy()
                    cv2.rectangle(overlay, (x,y),(x + w//2, y + h//2), color[i%len(color)], -1)  # A filled rectangle

                    alpha = 0.05 # Transparency factor.

                    # Following line overlays transparent rectangle over the image
                    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
                else:
                    cv2.rectangle(im, (x,y),(x + w//2, y + h//2), color[i%len(color)], 1)

    return im
