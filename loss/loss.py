import torch.nn as nn


from .functional import elbo

class Elbo(nn.Module):

    def __init__(self, sigma_prior = 0.1, alpha = 0.01):
        super(Elbo, self).__init__()

        self.sigma_prior = sigma_prior
        self.alpha = alpha
        self.iter = 0.0

    def forward(self, x, label):

        # split x into its stacked components
        x, mu, sigma = x

        self.nll, self.kl, self.rec = elbo(x, mu, sigma, label, sigma_prior = self.sigma_prior)

        # increment counter for each update
        self.iter += 1.0
        alpha = min(1, 1.0/self.iter)

        return self.nll + alpha*self.kl + self.rec
