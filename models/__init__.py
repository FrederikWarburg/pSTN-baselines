import json
import os
from os.path import join, isdir

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from data import DataLoader
from loss import create_criterion
from utils.evaluate import accuracy
from utils.utils import get_exp_name
from utils.visualizations import visualize_stn


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

    elif opt.optimizer.lower() == 'adam' or opt.model.lower() == 'cnn':
        from torch.optim import Adam as Optimizer
        if 'MNIST' in opt.dataset.lower():  # straight forward for MNIST, and CNN for all datasets
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        else:  # different learning rates for timeseries STN/P_STN model parts
            if opt.model.lower() == 'pstn':
                optimizer = torch.optim.Adam(
                    [{'params': model.pstn.fc_loc_mean.parameters(), 'lr': opt.lr / 10},
                     {'params': model.pstn.localization.parameters(), 'lr': opt.lr / 10},
                     {'params': model.pstn.fc_loc_std.parameters(), 'lr': opt.lr},
                     {'params': model.classifier.CNN.parameters(), 'lr': opt.lr},
                     {'params': model.classifier.fully_connected.parameters(), 'lr': opt.lr}],
                    weight_decay=opt.weightDecay)
            elif opt.model.lower() == 'stn':
                optimizer = torch.optim.Adam(
                    [{'params': model.stn.fc_loc.parameters(), 'lr': opt.lr / 10},
                     {'params': model.stn.localization.parameters(), 'lr': opt.lr / 10},
                     {'params': model.classifier.CNN.parameters(), 'lr': opt.lr},
                     {'params': model.classifier.fully_connected.parameters(), 'lr': opt.lr}],
                    weight_decay=opt.weightDecay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=1) # no decay

    else:
        print("{} is not implemented yet".format(opt.optimizer.lower()))
        raise NotImplemented

    return optimizer, scheduler


class System(pl.LightningModule):

    def __init__(self, opt):
        super(System, self).__init__()

        # hyper parameters
        self.hparams = opt
        self.opt = opt

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
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc, 'train_nll': F.nll_loss(y_hat, y, reduction='mean')}

        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        # unpack batch
        x, y = batch

        # forward
        y_hat = self.forward(x)

        # calculate nll and accuracy
        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

<<<<<<< HEAD
        #if batch_idx == 0:
        #    grid_in, grid_out, theta, bbox_images = visualize_stn(self.model, x, self.opt)
        #    self.add_images(grid_in, grid_out, bbox_images)
=======
        # for the first batch in an epoch visualize the predictions for better debugging
        if batch_idx == 0:
            # calculate different visualizations
            grid_in, grid_out, theta, bbox_images = visualize_stn(self.model, x, self.opt)

            # add these to tensorboard
            self.add_images(grid_in, grid_out, bbox_images)
>>>>>>> fd0a45d3223d080a32490e3483b2c2b2d24553b6

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):

        # calculate mean of nll and accuarcy
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        # add to tensorboard
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):

        # unpack batch
        x, y = batch

        # forward image
        y_hat = self.forward(x)

        # calculate nll and loss
        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        return {'test_loss': loss, 'test_acc': acc}

    def test_end(self, outputs):

        # calculate mean of nll and accuarcy
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        # add to tensorboard
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}

        # write results to json file also
        if self.opt.write_to_json:
            self.write_to_json(avg_loss, avg_acc)

        return {'test_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):

        # configure optimizer and scheduler
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

        # add different visualizations to tensorboard depending on the model

        # add input images
        self.logger.experiment.add_image('grid_in', grid_in, self.global_step)

        if self.opt.model.lower() in ['stn', 'pstn']:
            # add output of localizer
            self.logger.experiment.add_image('grid_out', grid_out, self.global_step)

            if bbox_images is not None:
                # add bounding boxes visualizations
                self.logger.experiment.add_image('bbox', bbox_images, self.global_step)

    def write_to_json(self, avg_loss, avg_acc):

        # get unique experiment name based on hyper parameters
        modelname = get_exp_name(self.opt)

        # path to save results
        basepath = self.opt.savepath if not None else os.getcwd()

        # create folder if not already exists
        if not isdir(join(basepath, 'results')): os.makedirs(join(basepath, 'results'))

        # write json
        with open(join(basepath,'results', modelname + '.json'), 'w') as fp:
            json.dump({'test_loss': float(avg_loss), 'test_acc': float(avg_acc)}, fp)
