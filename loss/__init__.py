import torch

def create_criterion(opt):

    if opt.criterion.lower() == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif opt.criterion.lower() == 'nllloss':
        criterion = torch.nn.NLLLoss()
    elif opt.criterion.lower() == 'elbo':
        from .loss import Elbo
        criterion = Elbo(opt.sigma)
    else:
        raise ValueError('Unsupported or optimizer: {}!'.format(opt.criterion))

    return criterion
