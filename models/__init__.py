import json
import os
from os.path import join, isdir

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from data import DataLoader
from loss import create_criterion
from utils.evaluate import accuracy
from utils.utils import get_exp_name, save_results
from utils.visualizations import visualize_stn
from collections import OrderedDict


def create_model(opt):
    # initalize model based on model type

    if opt.model.lower() == 'cnn':
        from .cnn import CNN as Model
    elif opt.model.lower() == 'stn':
        from .stn import STN as Model
    elif opt.model.lower() == 'pstn':
        from .pstn import PSTN as Model
    else:
        raise ValueError('Unsupported or model: {}!'.format(opt.model))

    model = Model(opt)

    return model


def create_optimizer(model, opt):
    """
    Returns an optimizer and scheduler based on chosen criteria
    """

    if opt.optimizer.lower() == 'sgd':
        from torch.optim import SGD as Optimizer
        opt_param = {'momentum' : opt.momentum, 'weight_decay' : opt.weightDecay}
    elif opt.optimizer.lower() == 'adam':
        from torch.optim import Adam as Optimizer
        opt_param = {'weight_decay' : opt.weightDecay}
    else:
        print("{} is not implemented yet".format(opt.optimizer.lower()))
        raise NotImplemented

    if opt.model.lower() == 'stn':
        # enables the lr for the localizer to be lower than for the classifier
        optimizer = Optimizer([
            {'params': filter(lambda p: p.requires_grad, model.stn.parameters()), 'lr': opt.lr_loc * opt.lr},
            {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': opt.lr},
        ], **opt_param)

    elif opt.model.lower() == 'pstn' :
        # enables the lr for the localizer to be lower than for the classifier
        optimizer = Optimizer([
            {'params': filter(lambda p: p.requires_grad, model.pstn.parameters()), 'lr': opt.lr_loc * opt.lr},
            {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': opt.lr},
        ], **opt_param)

    else:

        optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, **opt_param)

    # create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

    return optimizer, scheduler


class System(pl.LightningModule):

    def __init__(self, opt):
        super(System, self).__init__()

        # hyper parameters
        self.hparams = opt
        self.opt = opt
        self.batch_size = opt.batch_size

        # initalize model
        self.model = create_model(opt)

        # initialize criterion
        self.criterion = create_criterion(opt)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx, hidden=0):

        # unpack batch
        x, y = batch

        # forward and calculate loss
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        if self.opt.model.lower() == 'pstn':
            # the output is packaged a bit differently for pstn during training
            y_hat = y_hat[0]

        # calculate the accuracy
        acc = accuracy(y_hat, y)

        # log everything with tensorboard
        tensorboard_logs = OrderedDict({'train_loss': loss, 'train_acc': acc, 'train_nll': F.nll_loss(y_hat, y, reduction='mean')})

        return OrderedDict({'loss': loss, 'acc': acc, 'log': tensorboard_logs})

    def validation_step(self, batch, batch_idx):

        # unpack batch
        x, y = batch

        # forward
        y_hat = self.forward(x)

        # calculate nll and accuracy
        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        # for the first batch in an epoch visualize the predictions for better debugging
        if batch_idx == 0:
            # calculate different visualizations
            grid_in, grid_out, theta, bbox_images = visualize_stn(self.model, x, self.opt)
            # add these to tensorboard
            self.add_images(grid_in, grid_out, bbox_images)

        return OrderedDict({'val_loss': loss, 'val_acc': acc})

    def validation_end(self, outputs):

        # calculate mean of nll and accuarcy
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        # add to tensorboard
        tensorboard_logs = OrderedDict({'val_loss': avg_loss, 'val_acc': avg_acc})

        return OrderedDict({'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs})

    def test_step(self, batch, batch_idx):

        # unpack batch
        x, y = batch

        # forward image
        y_hat = self.forward(x)

        # calculate nll and loss
        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        return OrderedDict({'test_loss': loss, 'test_acc': acc})

    def test_end(self, outputs):

        # calculate mean of nll and accuarcy
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        # add to tensorboard
        tensorboard_logs = OrderedDict({'test_loss': avg_loss, 'test_acc': avg_acc})

        # write results to json file also
        if self.opt.save_results:
            save_results(self.opt, avg_loss, avg_acc)

        return OrderedDict({'test_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs})

    def configure_optimizers(self):

        # configure optimizer and scheduler
        optimizer, scheduler = create_optimizer(self.model, self.opt)

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.opt, mode='train', shuffle = True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.opt, mode='val', shuffle = False)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.opt, mode='test', shuffle = False)

    def add_images(self, grid_in, grid_out, bbox_images):

        # add different visualizations to tensorboard depending on the model

        # add input images
        self.logger.experiment.add_image('grid_in', grid_in, self.global_step)

        if self.opt.model.lower() in ['stn', 'pstn']:
            # add output of localizer
            self.logger.experiment.add_image('grid_out', grid_out, self.global_step)

            if bbox_images is not None:
                # add bounding boxes visualizations
                self.logger.experiment.add_image('bbox', bbox_images, self.global_step)
