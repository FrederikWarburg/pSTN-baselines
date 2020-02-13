import torch.nn.functional as F

class diffeomorphic_transformation:
    def __init__(self):

        # if 2D
            # T =
        # elif 1D
            # T =
        raise NotImplemented

    def forward(self, x, theta_upsample):

        return x

class affine_transformation:
    def __int__(self):
        pass

    def forward(self, x, theta_upsample):

        affine_params = self.make_affine_parameters(theta_upsample)

        grid = F.affine_grid(affine_params, x.size())

        x = F.grid_sample(x, grid)

        return x

    def make_affine_parameters(self, mean_params, sigma_params = None, sigma_prior = 0.1):

        if sigma_params is not None:
            eps = _standard_normal(mean_params.shape, dtype = mean_params.dtype, device=mean_params.device)
            eps *= sigma_prior
            mean_params = eps * sigma_params + mean_params

        if mean_params.shape[1] == 2: # only perform crop - fix scale and rotation.
            theta = torch.zeros(mean_params.shape[0], device=mean_params.device)
            scale = 0.5 * torch.ones(mean_params.shape[0], device=mean_params.device)
            translation_x = mean_params[:, 0]
            translation_y = mean_params[:, 1]
        elif mean_params.shape[1] == 4:
            theta = mean_params[:, 0]
            scale = mean_params[:, 1]
            translation_x = mean_params[:, 2]
            translation_y = mean_params[:, 3]
        elif mean_params.shape[1] == 6:
            affine_matrix = mean_params.view([-1, 2, 3])
            return affine_matrix

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
