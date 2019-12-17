import torch

def evaluate(pred, labels):

    # pstn's predictions during training is a tuple of (pred, theta_mu, theta_sigma)
    if type(pred) is tuple:
        pred, _, _ = pred

    pred = torch.argmax(pred, dim = 1)
    correct = pred.eq(labels).sum()

    return correct.item(), len(labels)
