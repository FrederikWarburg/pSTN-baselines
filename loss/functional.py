import torch
import torch.nn.functional as F
from torch import distributions as dist

def kl_div(mu, sigma, sigma_prior):

    mu = mu.view(-1)
    sigma = sigma.view(-1)
    mu_prior = torch.zeros_like(mu, device=mu.device)
    #if opt == 'affine': mu_prior[:, 1] = 1

    p = dist.MultivariateNormal(loc=mu_prior, scale_tril=sigma_prior*torch.eye(len(mu_prior)))
    q = dist.MultivariateNormal(loc=mu, scale_tril=sigma*torch.eye(len(mu)))
    kl_loss = dist.kl_divergence(q, p)

    return kl_loss

def elbo(x, mu, sigma, label, sigma_prior = 0.1):

    # NLL LOSS
    nll_loss = F.nll_loss(x, label, reduction='sum')

    # KL LOSS
    #TODO: maybe we can use built in pytorch class: KLDivLoss()
    kl_loss = kl_div(mu, sigma, sigma_prior)

    # RECONSTRUCTION LOSS
    reconstruction_loss = 0

    loss = nll_loss + kl_loss + reconstruction_loss

    return loss
