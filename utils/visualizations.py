import torch
import numpy as np
import torchvision
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt


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

        if opt.model.lower() == 'stn':
            transformed_input_tensor, theta, affine_params = model.stn(data)
        elif opt.model.lower() == 'pstn':
            transformed_input_tensor, theta, affine_params = model.pstn(data)

        in_grid = convert_image_np(torchvision.utils.make_grid(data.cpu()), opt.dataset.lower())
        in_grid = (in_grid*255).astype(np.uint8)
        in_grid = np.transpose(in_grid, (2,0,1))

        if opt.model.lower() == 'cnn':
            return in_grid, None, None, None

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



def visualize_timeseries_stn(device, model, data, epoch, DA, path):
    DA_flag = 'DA' if DA else 'noDA'
    print('visualizing STN!')
    plt.rcParams.update({'font.size': 12})

    with torch.no_grad():
        input_array = data.cpu().numpy()

        # Get a batch of training data
        transformed_input, theta = model.stn(data)
        transformed_input = transformed_input.cpu().numpy()
        theta = theta.cpu().numpy()
        print('Theta shape:', theta.shape)
        #batch_size = transformed_input.shape[0]
        batch_size = 3 # NOT PLOT ALL FOR NOW
        colors = ['darkgoldenrod', 'sienna', 'darkred']

        # Plot the results side-by-side
        f, axarr = plt.subplots(batch_size, 2, figsize=(20, 20))
        #plt.subplots_adjust(hspace=0.3)
        for ix in range(batch_size):
            axarr[ix, 0].plot(input_array[ix, 0, :], c='darkblue')
            #axarr[ix, 0].set_title('Original Timeseries, Epoch %s' % epoch)

            axarr[ix, 1].plot(transformed_input[ix, 0, :], c=colors[ix])
            #axarr[ix, 1].set_title('Trafo: %s' % theta[ix, :])
            # plt.ioff()
            # plt.show()
        plt.savefig(path + '_' + str(epoch) + '.png')
        #plt.savefig('TIMESERIES_DEBUG/visualizations/STN_%s_epoch_%s_%s.png'% (model.dataset, epoch, DA_flag))
        #plt.savefig('TIMESERIES_TEST/visualizations/STN_%s_epoch_%s_%s.png'% (model.dataset, epoch, DA_flag))
    return ...

def visualize_timeseries_p_stn(device, model, data, epoch, DA, path):
    DA_flag = 'DA' if DA else 'noDA'
    print('visualizing P_STN!')
    with torch.no_grad():
        input_array = data.cpu().numpy()
        model.eval()

        # Get a batch of training data
        transformed_input = model.probabilistic_stn(data)[0].cpu().numpy()
        print('S is', model.S)
        print(transformed_input.shape)
        batch_size = transformed_input.shape[0]

        # Plot the results side-by-side
        f, axarr = plt.subplots(2)
        axarr[0].plot(input_array[0, 0, :], c='darkblue')
        #axarr[0].get_xaxis().set_visible(False)
        #axarr[0].axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(hspace=0.3)
        #axarr[0].set_title('Original 1st Timeseries from batch')
        colors = ['darkgreen', 'darkgoldenrod', 'darkred']
        alphas = [0.5, 0.75, 1]

        for ix in range(3):
            axarr[1].plot(transformed_input[ix, 0, :], alpha=alphas[ix], c= 'sienna')#colors[ix])
            #axarr[1].set_title('Transformed 1st Timeseries, Epoch %s' % epoch)
            # plt.ioff()
            # plt.show()
            #axarr[1].get_xaxis().set_visible(False)
            #axarr[1].axes.get_yaxis().set_visible(False)
        plt.savefig(path + '_' + str(epoch) + '.pdf', bbox_inches='tight')
        #print('save image at', 'TIMESERIES_TEST/visualizations/P_STN_%s_epoch_%s_%s.png'% (model.dataset, epoch, DA_flag))
        #plt.savefig('TIMESERIES_TEST/visualizations/P_STN_%s_epoch_%s_%s.png'% (model.dataset, epoch, DA_flag))
