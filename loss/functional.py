import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl, gamma
import pickle


def kl_div(x, beta,  alpha_p, beta_p, reduction='mean', weights=None):
    q = gamma.Gamma(alpha_p, beta)
    p = gamma.Gamma(alpha_p, beta_p)

    kl_loss = kl.kl_divergence(q, p)

    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()


def elbo(x, beta, label, alpha_p, beta_p, reduction="mean"):
    # NLL LOSS
    if reduction == "mean":
        nll_loss = F.nll_loss(x, label, reduction='mean')
    else:
        nll_loss = F.nll_loss(x, label, reduction='none').min()

    # KL LOSS
    kl_loss = kl_div(x, beta, alpha_p, beta_p, reduction='mean')

    return nll_loss, kl_loss


def no_annealing(iter, M=None, base_kl=None, weight=None):
    return 1.

def weight_kl(iter, M=None, base_kl=None, weight=1.):
    return weight

def no_kl(iter, M=None, base_kl=None, weight=None):
    return 0


def reduce_kl(iter, M=None, base_kl=None, weight=None):
    return 1.0 / (1 + iter / 500)


def increase_kl(iter, M=None, base_kl=None, weight=None):
    return 1 - reduce_kl(iter)


def cyclic_kl(iter, M, base_kl=None, weight=None):
    # https://arxiv.org/pdf/1505.05424.pdf

    iter = (iter - 1.0) % M
    return 2.0**(M - iter) / (2.0**M - 1.0)


def scaled_kl(iter, M=None, base_kl=None, weight=None):
    scaling_factor = 1 / base_kl
    # print('scaling kl by', scaling_factor)
    return scaling_factor
