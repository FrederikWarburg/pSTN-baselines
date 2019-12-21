import torch.nn as nn


from .functional import elbo

class Elbo(nn.Module):

    def __init__(self, sigma_prior = 0.1, alpha = 0.01):
        super(Elbo, self).__init__()

        self.sigma_prior = sigma_prior
        self.alpha = alpha

    def forward(self, x, label):

        # split x into its stacked components
        x, mu, sigma = x

        self.nll, self.kl, self.rec = elbo(x, mu, sigma, label, sigma_prior = self.sigma_prior)


        return self.nll + self.alpha*self.kl + self.rec
