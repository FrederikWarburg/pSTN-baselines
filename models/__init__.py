import json
import os
from os.path import join

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from data import DataLoader
from loss import create_criterion
from utils.evaluate import accuracy
from utils.utils import get_exp_name
from utils.visualizations import visualize_stn


def create_model(opt):
    if opt.model.lower() == 'cnn':
        from .cnn import CNN
        model = CNN(opt)
    elif opt.model.lower() == 'stn':
        from .stn import STN
        model = STN(opt)
    elif opt.model.lower() == 'pstn':
        from .pstn import PSTN
        model = PSTN(opt)
    else:
        raise ValueError('Unsupported or model: {}!'.format(opt.model))

    return model


def create_optimizer(model, opt):
    if opt.optimizer.lower() == 'sgd':
        if opt.model.lower() == 'stn' and opt.lr_loc > 0:
            # the learning rate of the parameters that are part of the localizer are multiplied 1e-4
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.stn.parameters()), 'lr': opt.lr_loc * opt.lr},
                {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': opt.lr},
            ], momentum=opt.momentum, weight_decay=opt.weightDecay)
        elif opt.model.lower() == 'pstn' and opt.lr_loc > 0:
            # the learning rate of the parameters that are part of the localizer are multiplied 1e-4
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.pstn.parameters()), 'lr': opt.lr_loc * opt.lr},
                {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': opt.lr},
            ], momentum=opt.momentum, weight_decay=opt.weightDecay)
        else:
            print("=> SGD all parameters chosen")
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                                        momentum=opt.momentum, weight_decay=opt.weightDecay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

    elif opt.optimizer.lower() == 'adam':
        if 'MNIST' in opt.dataset.lower():  # straight forward for MNIST, and CNN for all datasets
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.lr)
        else:  # different learning rates for timeseries STN/P_STN model parts
            if opt.model.lower() == 'pstn':
                optimizer = torch.optim.Adam(
                    [{'params': model.fc_loc_mean.parameters(), 'lr': opt.lr / 10},
                     {'params': model.localization.parameters(), 'lr': opt.lr / 10},
                     {'params': model.fc_loc_std.parameters(), 'lr': opt.lr},
                     {'params': model.CNN.parameters(), 'lr': opt.lr},
                     {'params': model.fully_connected.parameters(), 'lr': opt.lr}],
                    weight_decay=opt.weightDecay)
            elif opt.model.lower() == 'stn':
                optimizer = torch.optim.Adam(
                    [{'params': model.fc_loc.parameters(), 'lr': opt.lr / 10},
                     {'params': model.localization.parameters(), 'lr': opt.lr / 10},
                     {'params': model.CNN.parameters(), 'lr': opt.lr},
                     {'params': model.fully_connected.parameters(), 'lr': opt.lr}],
                    weight_decay=opt.weightDecay)

    else:
        raise ValueError('Unsupported or optimizer: {}!'.format(opt.optimizer))

    return optimizer, scheduler


def save_network(model, opt, which_epoch, is_best=False):
    """save model to disk"""
    device = model.device
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

    model.to(device)


class System(pl.LightningModule):

    def __init__(self, opt):
        super(System, self).__init__()
        # not the best model...
        self.model = create_model(opt)
        self.hparams = opt
        self.opt = opt
        self.criterion = create_criterion(opt)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx, hidden=0):

        x, y = batch

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        if self.opt.model.lower() == 'pstn':
            y_hat = y_hat[0]

        acc = accuracy(y_hat, y)
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc, 'train_nll': F.nll_loss(y_hat, y, reduction='mean')}

        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self.forward(x)

        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        if batch_idx == 0:
            grid_in, grid_out, theta, bbox_images = visualize_stn(self.model, x, self.opt)
            self.add_images(grid_in, grid_out, bbox_images)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self.forward(x)

        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        return {'test_loss': loss, 'test_acc': acc}

    def test_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}

        if self.opt.save_results:
            self.save_results(avg_loss, avg_acc)

        return {'test_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):

        optimizer, scheduler = create_optimizer(self.model, self.opt)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.opt, mode='train')

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.opt, mode='val')

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.opt, mode='test')

    def add_images(self, grid_in, grid_out, bbox_images):

        self.logger.experiment.add_image('grid_in', grid_in, self.global_step)

        if self.opt.model.lower() in ['stn', 'pstn']:
            self.logger.experiment.add_image('grid_out', grid_out, self.global_step)

            if bbox_images is not None:
                self.logger.experiment.add_image('bbox', bbox_images, self.global_step)

    def save_results(self, avg_loss, avg_acc):

        modelname = get_exp_name(self.opt)

        basepath = self.opt.savepath if not None else os.getcwd()
        if not os.path.isdir(os.path.join(basepath, 'results')): os.makedirs(os.path.join(basepath, 'results'))

        with open(os.path.join(basepath,
                               'results',
                               modelname + '.json'), 'w') as fp:
            json.dump({'test_loss': float(avg_loss), 'test_acc': float(avg_acc)}, fp)
