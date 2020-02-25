import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch
import torchvision


def convert_image_np(inp, dataset='mnist'):
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
    print('VISUALIZING')
    with torch.no_grad():
        # IMAGE VISUALIZATION
        if opt.xdim == 2:
            print('IMAGE DIMS')
            data = data[:16]  # just visualize the first 16
            if opt.model.lower() == 'stn':
                transformed_input_tensor, theta, affine_params = model.stn(data)

            elif opt.model.lower() == 'pstn':
                transformed_input_tensor, theta, affine_params = model.pstn(data)

            in_grid = convert_image_np(torchvision.utils.make_grid(data.cpu()), opt.dataset.lower())
            in_grid = (in_grid * 255).astype(np.uint8)
            in_grid = np.transpose(in_grid, (2, 0, 1))

            if opt.model.lower() == 'cnn':
                return in_grid, None, None, None

            out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor.cpu()),
                                        opt.dataset.lower())
            out_grid = (out_grid * 255).astype(np.uint8)
            out_grid = np.transpose(out_grid, (2, 0, 1))
            bbox_images = visualize_bbox(data.cpu(), affine_params, opt)

        # TIMESERIES VISUALIZATIONS
        if opt.xdim == 1:
            bbox_images = None
            nr_plots = 3  # just visualize the first 3
            input_array = data.cpu().numpy()
            in_fig, in_ax = plt.subplots(nr_plots, figsize=(20, 10))
            for ix in range(nr_plots):
                in_ax[ix].plot(input_array[ix, 0, :], c='darkblue')

            # print figure to numpy array
            fig = Figure()
            canvas = FigureCanvas(in_fig)
            ax = fig.gca()
            ax.axis('off')
            canvas.draw()
            in_grid = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(1000, 2000, 3)
            #plt.savefig('test.png')

            if opt.model.lower() == 'cnn':
                print('returning ', in_grid.shape, in_grid.dtype)
                return in_grid, None, None, None  #in_grid, None, None, None

            out_fig, out_ax = plt.subplots(nr_plots, figsize=(20, 10))
            if opt.model.lower() == 'stn':
                transformed_input, _, theta = model.stn(data)
                transformed_input = transformed_input.cpu().numpy()
                theta = theta.cpu().numpy()
                colors = ['darkgoldenrod', 'sienna', 'darkred']
                for ix in range(nr_plots):
                    out_ax[ix].plot(transformed_input[ix, 0, :], c=colors[ix])

            if opt.model.lower() == 'pstn':
                # model.eval()
                batch_size = data.shape[0]
                transformed_input, _, sampled_theta = model.pstn(data).cpu().numpy()
                theta = sampled_theta.cpu().numpy()
                nr_samples = 3
                colors = ['darkgreen', 'darkgoldenrod', 'darkred']
                alphas = [0.5, 0.75, 1]
                # Plot the results side-by-side
                for data_ix in range(nr_plots):
                    for sample_ix in range(nr_samples):
                        out_ax[data_ix].plot(transformed_input[data_ix + (sample_ix * batch_size), 0, :],
                                               alpha=alphas[ix], c='sienna')
            out_fig.canvas.draw()
            out_grid = np.array(out_fig.canvas.renderer.buffer_rgba())

    # Plot the results side-by-side
    return in_grid, out_grid, theta, bbox_images


def visualize_bbox(data, affine_params, opt):
    batch_size = data.shape[0]
    affine_params = torch.stack(affine_params.split([opt.N] * opt.test_samples * batch_size))
    sorted_params = torch.zeros(batch_size, opt.test_samples, opt.N, 2, 3, dtype=affine_params.dtype)
    for i in range(len(affine_params)):
        # im, sample, N, 2, 3
        sorted_params[i % batch_size, i // batch_size, :, :, :] = affine_params[i, :, :]

    images = []
    for j, (im, params) in enumerate(zip(data, sorted_params)):

        im = np.transpose(im.numpy(), (1, 2, 0))
        im = denormalize(im)

        if im.shape[2] == 1:
            im = np.stack((im[:, :, 0],) * 3, axis=-1)

        if torch.isnan(params).any(): continue

        im = add_bounding_boxes(im, params, opt.N, opt.test_samples, mode_='crop', heatmap=opt.heatmap)

        images.append(np.transpose(im, (2, 0, 1)))

    if len(images) > 0:

        images = torch.FloatTensor(images)
        if 'mnist' in opt.dataset.lower():
            images = convert_image_np(torchvision.utils.make_grid(images - 0.3), opt.dataset.lower())
            images = (images * 255).astype(np.uint8)
            images = np.transpose(images, (2, 0, 1))
        else:
            images = torchvision.utils.make_grid(images, normalize=True)

        return images

    return None


def denormalize(image):
    im = (image - np.min(image)) / (np.max(image) - np.min(image))
    return im


def add_bounding_boxes(image, affine_params, num_branches, num_samples, mode_='crop', heatmap=True):
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

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
                x = int(x * w // 2 + w // 4)
                y = int(y * h // 2 + h // 4)

                if heatmap:

                    overlay = im.copy()
                    cv2.rectangle(overlay, (x, y), (x + w // 2, y + h // 2), color[i % len(color)],
                                  -1)  # A filled rectangle

                    alpha = 0.05  # Transparency factor.

                    # Following line overlays transparent rectangle over the image
                    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
                else:
                    cv2.rectangle(im, (x, y), (x + w // 2, y + h // 2), color[i % len(color)], 1)

    return im
