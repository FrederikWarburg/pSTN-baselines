import torch

def evaluate(pred, labels):

    pred = torch.argmax(pred, dim = 1)
    correct = pred.eq(labels).sum()

    return correct.item(), len(labels)
