import torch.nn as nn


from .functional import elbo

class Elbo(nn.Module):

    def __init__(self, sigma = 0.1):
        super(Elbo, self).__init__()

        self.sigma_prior = sigma

    def forward(self, x, label):

        # split x into its stacked componets
        x, mu, sigma = x

        return elbo(x, mu, sigma, label, sigma_prior = self.sigma_prior)
