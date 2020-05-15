import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl


def kl_div(mu, sigma, mu_p, sigma_p, reduction='mean', moving_mean=False):
    batch_size, params = mu.shape

    sigma = torch.diag_embed(sigma)
    mu_prior = mu if moving_mean else mu_p

    sigma_p = sigma_p * torch.eye(params, device=mu.device).unsqueeze(0).repeat(batch_size, 1, 1)

    p = MultivariateNormal(loc=mu_prior, scale_tril=sigma_p)
    q = MultivariateNormal(loc=mu, scale_tril=sigma)
    print('Moving mean is', moving_mean)
    print('devices are', mu.device, sigma.device, mu_prior.device, sigma_p.device)
    kl_loss = kl.kl_divergence(q, p)

    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()


def elbo(x, mu, sigma, label, mu_p, sigma_p=0.1, moving_mean=False):
    # NLL LOSS
    nll_loss = F.nll_loss(x, label, reduction='mean')

    # KL LOSS
    kl_loss = kl_div(mu, sigma, mu_p, sigma_p, reduction='mean', moving_mean=moving_mean)

    # RECONSTRUCTION LOSS
    reconstruction_loss = 0

    return nll_loss, kl_loss, reconstruction_loss


def no_annealing(iter, M = None):
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
