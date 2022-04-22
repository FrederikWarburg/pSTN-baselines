import torch


def evaluate(pred, labels):
    # pstn's predictions during training is a tuple of (pred, theta_mu, theta_std)
    if type(pred) is tuple:
        pred, _, _ = pred

    pred = torch.argmax(pred, dim=1)
    correct = pred.eq(labels).sum()

    return correct, len(labels)


def accuracy(pred, labels, reduction='mean'):

    # pstn's predictions during training is a tuple of (pred, theta_mu, theta_std)
    if type(pred) is tuple:
        pred, _, _ = pred
    
    if reduction == "min" and pred.ndim == 3:
        b, s, c = pred.shape
        pred = torch.log(torch.tensor(1.0 / float(s))) + torch.logsumexp(pred, dim=1)
        pred = pred.view(b, c)

    pred = torch.argmax(pred, dim=1)
    acc = (pred.eq(labels).sum()).float() / float(len(labels))

    return acc.float()
