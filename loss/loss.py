import torch.nn as nn
from .functional import elbo
import torch


def initialize_mu_prior(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.moving_mean:
        mu_p = None  # in this case it will get updated on the fly

    elif opt.transformer_type == 'diffeomorphic':
        theta_dim = opt.num_param * opt.N
        mu_p = torch.zeros((1, theta_dim)).to(device)

    elif opt.transformer_type == 'affine':
        if opt.num_param == 4:
            mu_p = torch.Tensor([0, 1, 0, 0]).to(device)
        # TODO?

    else:
        raise NotImplementedError

    return mu_p


class Elbo(nn.Module):

    def __init__(self, opt, annealing='reduce_kl'):
        super(Elbo, self).__init__()
        self.moving_mean = opt.moving_mean
        self.sigma_p = opt.sigma_p
        self.mu_p = initialize_mu_prior(opt)
        self.iter = 0.0

        # number of batches in epoch (only used for cyclic kl weighting)
        self.M = None

        if annealing == 'no_annealing':
            from .functional import no_annealing as annealing
        elif annealing == 'no_kl':
            from .functional import no_kl as annealing
        elif annealing == 'reduce_kl':
            from .functional import reduce_kl as annealing
        elif annealing == 'increase_kl':
            from .functional import increase_kl as annealing
        elif annealing == 'cyclic_kl':
            from .functional import cyclic_kl as annealing
        else:
            raise NotImplemented

        self.annealing = annealing

    def forward(self, x, theta, label):
        mu, sigma = theta

        # calculate terms of elbo
        self.nll, self.kl, self.rec = elbo(x, mu, sigma, label, mu_p=self.mu_p, sigma_p=self.sigma_p, moving_mean=self.moving_mean)

        # increment counter for each update
        self.iter += 1.0

        # weighting of kl term
        alpha = self.annealing(self.iter, self.M)

        return self.nll + alpha * self.kl + self.rec
