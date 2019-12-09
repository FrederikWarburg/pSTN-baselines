import torch
from os.path import join

def create_model(opt):
    if opt.model.lower() == 'inception':
        from .inception import InceptionClassifier
        model = InceptionClassifier(opt)
    elif opt.model.lower() == 'stn':
        from .stn import STN
        model = STN(opt)
    else:
        raise ValueError('Unsupported or model: {}!'.format(opt.model))

    return model


def create_optimizer(model, opt):

    if opt.optimizer.lower() == 'sgd':
        if opt.model.lower() == 'stn':
            # the learning rate of the parameters that are part of the localizer are multiplied 1e-4
            optimizer = torch.optim.SGD([
                {'classifier': model.classifier.parameters(), 'lr': opt.lr},
            ], lr=opt.lr * 1e-4, momentum=opt.momentum)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                                            momentum=opt.momentum,
                                            weight_decay=opt.weightDecay)



        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

    else:
        raise ValueError('Unsupported or optimizer: {}!'.format(opt.optimizer))

    return optimizer, scheduler

def create_criterion(opt):

    if opt.criterion.lower() == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unsupported or optimizer: {}!'.format(opt.criterion))

    return criterion


def save_network(model, opt, which_epoch, is_best = False):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(opt.save_dir, save_filename)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.cpu().state_dict(), save_path)
            if is_best:
                torch.save(model.module.cpu().state_dict(), join(opt.save_dir, "best_net.pth"))
        else:
            torch.save(model.cpu().state_dict(), save_path)
            if is_best:
                torch.save(model.cpu().state_dict(), join(opt.save_dir, "best_net.pth"))
