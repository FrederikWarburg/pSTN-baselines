import torch.nn as nn


from .functional import elbo

class Elbo(nn.Module):

    def __init__(self, sigma_prior = 0.1, annealing = 'reduce_kl'):
        super(Elbo, self).__init__()

        self.sigma_prior = sigma_prior
        self.iter = 0.0

        if annealing == 'no_annealing':
            self.annealing = self.__no_annealing__
        elif annealing == 'no_kl':
            self.annealing = self.__no_kl__
        elif annealing == 'reduce_kl':
            self.annealing = self.__reduce_kl__
        elif annealing == 'increase_kl':
            self.annealing = self.__increase_kl__

    def forward(self, x, label):

        # split x into its stacked components
        x, mu, sigma = x

        self.nll, self.kl, self.rec = elbo(x, mu, sigma, label, sigma_prior = self.sigma_prior)

        # increment counter for each update
        self.iter += 1.0

        return self.nll + self.annealing(self.iter)*self.kl + self.rec

    def __no_annealing__(self, iter):
        return 1

    def __no_kl__(self, iter):
        return 0

    def __reduce_kl__(self, iter):
        return 1.0/iter

    def __increase_kl__(self, iter):
        return 1 - self.__reduce_kl__(iter)
