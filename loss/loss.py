import torch.nn as nn

from .functional import elbo


class Elbo(nn.Module):

    def __init__(self, sigma_p=0.1, annealing='reduce_kl'):
        super(Elbo, self).__init__()

        self.sigma_p = sigma_p
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

    def forward(self, x, label):

        # split x into its stacked components
        x, mu, sigma = x

        # calculate terms of elbo
        self.nll, self.kl, self.rec = elbo(x, mu, sigma, label, sigma_p=self.sigma_p)

        # increment counter for each update
        self.iter += 1.0

        # weighting of kl term
        alpha = self.annealing(self.iter, self.M)

        return self.nll + alpha * self.kl + self.rec

