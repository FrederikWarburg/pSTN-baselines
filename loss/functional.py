import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl


def kl_div(mu, sigma, sigma_p, reduction='mean'):
    batch_size, params = mu.shape

    sigma = torch.diag_embed(sigma)
    mu_prior = torch.zeros_like(mu, device=mu.device)
    sigma_p = sigma_p * torch.eye(params, device=mu.device).unsqueeze(0).repeat(batch_size, 1, 1)

    p = MultivariateNormal(loc=mu, scale_tril=sigma_p)
    q = MultivariateNormal(loc=mu, scale_tril=sigma)

    kl_loss = kl.kl_divergence(q, p)

    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()


def elbo(x, mu, sigma, label, sigma_p=0.1):
    # NLL LOSS
    nll_loss = F.nll_loss(x, label, reduction='mean')

    # KL LOSS
    kl_loss = kl_div(mu, sigma, sigma_p, reduction='mean')

    # RECONSTRUCTION LOSS
    reconstruction_loss = 0

    return nll_loss, kl_loss, reconstruction_loss


def no_annealing(iter):
    return 1

def no_kl(iter):
    return 0

def reduce_kl(iter):
    return 1.0 / iter

def increase_kl(iter):
    return 1 - reduce_kl(iter)
