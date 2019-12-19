import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl

def kl_div(mu, sigma, sigma_prior):

    batch_size, params = mu.shape

    sigma = torch.diag_embed(sigma)
    mu_prior = torch.zeros_like(mu, device=mu.device)
    sigma_prior = sigma_prior * torch.eye(params, device=mu.device).unsqueeze(0).repeat(batch_size, 1, 1)

    p = MultivariateNormal(loc=mu_prior, scale_tril=sigma_prior)
    q = MultivariateNormal(loc=mu, scale_tril=sigma)

    kl_loss = kl.kl_divergence(q, p)

    return kl_loss.mean()

def elbo(x, mu, sigma, label, sigma_prior = 0.1):

    # NLL LOSS
    nll_loss = F.nll_loss(x, label, reduction='mean')

    # KL LOSS
    kl_loss = kl_div(mu, sigma, sigma_prior)

    # RECONSTRUCTION LOSS
    reconstruction_loss = 0

    return nll_loss, kl_loss, reconstruction_loss
