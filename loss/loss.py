import torch.nn as nn


from .functional import elbo

class Elbo(nn.Module):

    def __init__(self, sigma_prior = 0.1):
        super(Elbo, self).__init__()

        self.sigma_prior = sigma_prior

    def forward(self, x, label):

        # split x into its stacked componets
        x, mu, sigma = x

        self.nll, self.kl, self.rec = elbo(x, mu, sigma, label, sigma_prior = self.sigma_prior)

        sum_ = self.nll + self.kl + self.rec
        return self.nll/sum_ + self.kl/sum_ + self.rec/sum_
