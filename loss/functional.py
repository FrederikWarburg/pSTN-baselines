import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl
import pickle


def kl_div(mu, sigma, mu_p, sigma_p, reduction='mean', prior_type='zero_mean_gaussian', weights=None):
    batch_size, params = mu.shape

    sigma = torch.diag_embed(sigma)
    q = MultivariateNormal(loc=mu, scale_tril=sigma)

    if prior_type in ['mean_zero_gaussian', 'moving_mean']:
        print('FIRST CASE: prior type is', prior_type)
        mu_prior = mu if (prior_type == 'moving_mean') else mu_p.repeat(batch_size, 1)  # am I missing an N here?
        sigma_p = sigma_p * torch.eye(params, device=mu.device).unsqueeze(0).repeat(batch_size, 1, 1)
        p = MultivariateNormal(loc=mu_prior, scale_tril=sigma_p)
        kl_loss = kl.kl_divergence(q, p)

    elif prior_type == 'mixture_of_gaussians':
        weights = pickle.load(open('priors/mog_weights.p', 'rb'))
        kl_loss = 0
        for component in range(8):  # TODO: make nr_components a variable
            mu_prior = mu_p[component].repeat(batch_size, 1)  # am I missing an N here?
            sigma_prior = sigma_p[component] * torch.eye(params, device=mu.device).unsqueeze(0).repeat(batch_size, 1, 1)
            p = MultivariateNormal(loc=mu_prior, scale_tril=sigma_prior)
            kl_loss += weights[component] * kl.kl_divergence(q, p)

    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()


def elbo(x, mu, sigma, label, mu_p, sigma_p=0.1, prior_type='mean_zero_gaussian'):
    # NLL LOSS
    nll_loss = F.nll_loss(x, label, reduction='mean')

    # KL LOSS
    kl_loss = kl_div(mu, sigma, mu_p, sigma_p, reduction='mean', prior_type=prior_type)

    # RECONSTRUCTION LOSS
    reconstruction_loss = 0

    return nll_loss, kl_loss, reconstruction_loss


def no_annealing(iter, M=None):
    return 1

def no_kl(iter, M = None):
    return 0

def reduce_kl(iter, M = None):
    return 1.0 / iter

def increase_kl(iter, M = None):
    return 1 - reduce_kl(iter)

def cyclic_kl(iter, M):
    # https://arxiv.org/pdf/1505.05424.pdf

    iter = (iter - 1.0) % M

    return 2.0**(M - iter) / (2.0**M - 1.0)
