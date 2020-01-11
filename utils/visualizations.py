import torch
import numpy as np
import torchvision

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(model, data, opt):

    with torch.no_grad():

        data = data[:16] # just visualize the first 16
        input_tensor = data.cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        in_grid = (in_grid*255).astype(np.uint8)
        in_grid = np.transpose(in_grid, (2,0,1))

        if opt.model.lower() == 'cnn':
            return in_grid, None
        elif opt.model.lower() == 'stn':
            transformed_input_tensor, theta, affine_params = model.stn(data)
        elif opt.model.lower() == 'pstn':
            transformed_input_tensor, theta, affine_params = model.pstn(data)

        transformed_input_tensor = transformed_input_tensor.cpu()

        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))
        out_grid = (out_grid*255).astype(np.uint8)
        out_grid = np.transpose(out_grid, (2,0,1))

        # Plot the results side-by-side
    return in_grid, out_grid
