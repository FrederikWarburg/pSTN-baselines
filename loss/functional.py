import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl, gamma
import pickle


# def find_closest_ix(mu, mu_p):
#     mu = mu.repeat(8, 1, 1)  # repeat along nr_components, TODO: remove hard coding
#     diff = mu - mu_p
#     abs_diff = torch.norm(diff, dim=(2))
#     ix = torch.argmin(abs_diff, dim=0)
#     return ix


def kl_div(x, beta,  alpha_p, beta_p, reduction='mean', weights=None):
    q = gamma.Gamma(alpha_p, beta)
    p = gamma.Gamma(alpha_p, beta_p)
    # print('alpha_p', alpha_p.device, 'beta_p', beta_p.device, 'beta', beta.device)
    # exit()

    kl_loss = kl.kl_divergence(q, p)

    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()


def elbo(x, beta, label, alpha_p, beta_p):
    # NLL LOSS
    nll_loss = F.nll_loss(x, label, reduction='mean')

    # KL LOSS
    kl_loss = kl_div(x, beta, alpha_p, beta_p, reduction='mean')

    return nll_loss, kl_loss


def no_annealing(iter, M=None, base_kl=None):
    return 1


def no_kl(iter, M=None, base_kl=None):
    return 0


def reduce_kl(iter, M=None, base_kl=None):
    return 1.0 / (1 + iter / 500)


def increase_kl(iter, M=None, base_kl=None):
    return 1 - reduce_kl(iter)


def cyclic_kl(iter, M, base_kl=None):
    # https://arxiv.org/pdf/1505.05424.pdf

    iter = (iter - 1.0) % M
    return 2.0**(M - iter) / (2.0**M - 1.0)


def scaled_kl(iter, M=None, base_kl=None):
    scaling_factor = 1 / base_kl
    # print('scaling kl by', scaling_factor)
    return scaling_factor
