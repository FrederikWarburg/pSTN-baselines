import json
import os
from os.path import join, isdir, exists
import pickle

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from torch.utils.data import DataLoader
from loss import create_criterion
from utils.evaluate import accuracy
from utils.utils import get_exp_name, save_UQ_results, save_results, mkdir, save_learned_thetas, save_UQ_results
from utils.visualizations import visualize_stn
from collections import OrderedDict
from data import create_dataset


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


def create_optimizer(model, opt, criterion):
    print('CREATING OPTIMIZER')
    """
    Returns an optimizer and scheduler based on chosen criteria
    """
    if opt.optimizer.lower() == 'sgd':
        from torch.optim import SGD as Optimizer
        opt_param = {'momentum': opt.momentum, 'weight_decay' : opt.weightDecay}
    elif opt.optimizer.lower() == 'adam':
        from torch.optim import Adam as Optimizer
        opt_param = {'weight_decay': opt.weightDecay}
    else:
        print("{} is not implemented yet".format(opt.optimizer.lower()))
        raise NotImplemented

    if opt.model.lower() == 'stn':
        # enables the lr for the localizer to be lower than for the classifier
        optimizer = Optimizer([
            {'params': filter(lambda p: p.requires_grad, model.stn.parameters()), 'lr': opt.lr_loc * opt.lr},
            {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': opt.lr},
        ], **opt_param)

    elif opt.model.lower() == 'pstn':
        # enables the lr for the localizer to be lower than for the classifier
        print('passing to optimizer:',
            {'params': list(filter(lambda p: p.requires_grad, criterion.parameters())), 'lr': opt.lr})
        # exit()
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
        print('system opt', opt)
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
        theta_mu, beta = None, None
        # forward and calculate loss, the output is packaged a bit differently for all models
        if self.opt.model.lower() == 'cnn':
            y_hat = self.forward(x)
            loss = self.criterion(y_hat, y)
        if self.opt.model.lower() == 'stn':
            y_hat, theta_mu = self.forward(x)
            loss = self.criterion(y_hat, y)
        if self.opt.model.lower() == 'pstn':
            y_hat, theta_samples, theta_params = self.forward(x)
            theta_mu, beta = theta_params
            loss = self.criterion(y_hat, beta, y)
        # calculate the accuracy
        acc = accuracy(y_hat, y)

        # Logging
        self.log("train_%s" %self.opt.criterion, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if not self.opt.criterion == 'nll':
            batch_nll = F.nll_loss(y_hat, y, reduction='mean')
            self.log("train_nll", batch_nll, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "theta_mu": theta_mu, "beta": beta} 

    # this might override on_epoch above?
    def training_epoch_end(self, outputs):
        if self.opt.save_training_theta:
            save_learned_thetas(self.opt, outputs, mode='train', epoch=self.current_epoch)
        if self.opt.model.lower() in ['stn', 'pstn']: # log mean trafos
            theta_mu = torch.stack([x['theta_mu'] for x in outputs])
            # log theta samples
            for i in range(self.opt.num_param):
                self.logger.experiment.add_histogram("train_theta_mu_%s" %i, theta_mu[:, :, i], self.current_epoch) # shape = [batches, batch_size, param]
        if self.opt.model.lower() ==  'pstn': # log beta
            beta = torch.stack([x['beta'] for x in outputs])
            # log theta samples
            for i in range(self.opt.num_param):
                self.logger.experiment.add_histogram("train_beta_%s" %i, beta[:, :, i], self.current_epoch)


    def validation_step(self, batch, batch_idx):
        # unpack batch
        x, y = batch
        # forward
        if self.opt.model.lower() == 'cnn':
            y_hat = self.forward(x)
        else:
            y_hat = self.forward(x)[0]
        # calculate nll and accuracy
        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)
        # for the first batch in an epoch visualize the predictions for better debugging
        if batch_idx == 0:
            # calculate different visualizations
            grid_in, grid_out, _, bbox_images = visualize_stn(self.model, x, self.opt)
            # add these to tensorboard
            self.add_images(grid_in, grid_out, bbox_images)
        self.log('val_nll', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # unpack batch
        x, y = batch

        theta_mu, beta = None, None
        # forward image
        if self.opt.model.lower() == 'cnn':
            y_hat = self.forward(x)
            loss = self.criterion(y_hat, y)
        if self.opt.model.lower() == 'stn':
            y_hat, theta_mu = self.forward(x)
            loss = self.criterion(y_hat, y)
        if self.opt.model.lower() == 'pstn':
            y_hat, theta_samples, theta_params = self.forward(x)
            theta_mu, beta = theta_params
            loss = self.criterion(y_hat, beta, y)

        # calculate nll and loss
        batch_nll = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        # for the first batch in an epoch visualize the predictions for better debugging
        if batch_idx == 0:
            print("Visualize during test")
            # calculate different visualizations
            grid_in, grid_out, _, bbox_images = visualize_stn(self.model, x, self.opt)
            # add these to tensorboard
            self.add_images(grid_in, grid_out, bbox_images)

        # compute UQ statistics
        pred = y_hat.max(1, keepdim=True)[1]
        check_predictions = pred.eq(y.view_as(pred)).all(dim=1)

        return {'test_loss': batch_nll, 'test_acc': acc,
                'probabilities': y_hat.data,
                'correct_prediction': y.data,
                'correct': check_predictions.data,
                "theta_mu": theta_mu, 
                "beta": beta}


    def test_epoch_end(self, outputs):
        # calculate mean of nll and accuarcy 
        # we do some manual logging here in order to also save to file
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("test_nll", avg_loss)
        self.log("test_acc", avg_acc)

        if self.opt.save_results:
            # concatenate UQ results manually (since we don't want to aggrehgate mean )
            probabilities = torch.stack([x['probabilities'] for x in outputs]).cpu().numpy()
            correct_predictions = torch.stack([x['correct_prediction'] for x in outputs]).cpu().numpy()
            correct = torch.stack([x['correct'] for x in outputs]).cpu().numpy()

            save_learned_thetas(self.opt, outputs, mode='test')

            # save UQ results
            save_UQ_results(self.opt, probabilities, correct_predictions, correct)

            # write results to json file also
            save_results(self.opt, avg_loss, avg_acc)

    def configure_optimizers(self):
        # configure optimizer and scheduler
        optimizer, scheduler = create_optimizer(self.model, self.opt, self.criterion)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = create_dataset(self.opt, mode='train')
        # dataloader params
        opt = {"batch_size": self.opt.batch_size, "shuffle": True, "pin_memory": True, "num_workers": int(self.opt.num_threads), 'drop_last': True}
        # return data loader
        dataloader = DataLoader(dataset, **opt)
        return dataloader

    def val_dataloader(self):
        # initialize dataset
        dataset = create_dataset(self.opt, mode='val')
        # dataloader params
        opt = {"batch_size": self.opt.batch_size, "shuffle": False, "pin_memory": True, "num_workers": int(self.opt.num_threads), 'drop_last': True}
        # return data loader
        return DataLoader(dataset, **opt)

    def test_dataloader(self):
        # initialize dataset
        dataset = create_dataset(self.opt, mode='test')
        # dataloader params
        opt = {"batch_size": self.opt.batch_size, "shuffle": False, "pin_memory": True, "num_workers": int(self.opt.num_threads), 'drop_last': True}
        # return data loader
        return DataLoader(dataset, **opt)

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
