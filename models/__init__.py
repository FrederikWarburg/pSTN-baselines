import json
import os
from os.path import join, isdir
import pickle

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from torch.utils.data import DataLoader
from loss import create_criterion
from utils.evaluate import accuracy
from utils.utils import get_exp_name, save_results
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


def create_optimizer(model, opt):
    """
    Returns an optimizer and scheduler based on chosen criteria
    """
    if opt.optimize_temperature:
        from torch.optim import SGD as Optimizer
        optimizer = Optimizer([model.model.T], lr=1e-3)

    else:
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
        print('system opt', opt)

        # hyper parameters
        self.hparams = opt

        self.opt = opt
        self.batch_size = opt.batch_size

        # initalize model
        self.model = create_model(opt)

        # initialize criterion
        self.criterion = create_criterion(opt)

    def forward(self, x):
        x, theta = self.model.forward(x)
        return x, theta

    def training_step(self, batch, batch_idx, hidden=0):

        # unpack batch
        x, y = batch

        # forward and calculate loss
        y_hat, theta = self.forward(x)

        if self.opt.model.lower() == 'pstn':
            # the output is packaged a bit differently for pstn during training
            loss = self.criterion(y_hat, theta, y)
        else:
            loss = self.criterion(y_hat, y)

        # calculate the accuracy
        acc = accuracy(y_hat, y)

        # log everything with tensorboard
        if self.opt.model == "pstn":
            T = self.model.classifier.T
        if self.opt.model == "stn":
            T = self.model.classifier.T
        if self.opt.model == "cnn":
            T = self.model.cnn.T
        tensorboard_logs = OrderedDict({'train_loss': loss, 'train_acc': acc, 'train_nll': F.nll_loss(y_hat, y, reduction='mean'), 'T': T})

        return OrderedDict({'loss': loss, 'acc': acc, 'log': tensorboard_logs})

    def validation_step(self, batch, batch_idx):

        # unpack batch
        x, y = batch

        # forward
        y_hat, _ = self.forward(x)

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
        print('batch size is', x.shape[0])

        # forward image
        y_hat, theta = self.forward(x)

        # calculate nll and loss
        loss = F.nll_loss(y_hat, y, reduction='mean')
        acc = accuracy(y_hat, y)

        # for the first batch in an epoch visualize the predictions for better debugging
        if batch_idx == 0:
            print("Visualize during test")
            # calculate different visualizations
            grid_in, grid_out, _, bbox_images = visualize_stn(self.model, x, self.opt)
            # add these to tensorboard
            self.add_images(grid_in, grid_out, bbox_images)

        # unpack theta for logging
        theta_mu = None
        theta_sigma = None
        if 'stn' in self.opt.model.lower():
            theta_mu = theta[0]
            print('theta mu is size', theta_mu.shape)
        if self.opt.model.lower() == 'pstn':
            theta_sigma = theta[1]

        # compute UQ statistics
        pred = y_hat.max(1, keepdim=True)[1]
        check_predictions = pred.eq(y.view_as(pred)).all(dim=1)

        return OrderedDict({'test_loss': loss, 'test_acc': acc,
                      'probabilities': y_hat.data,
                      'correct_prediction': y.data,
                      'correct': check_predictions.data,
                      'theta_mu': theta_mu,
                      'theta_sigma': theta_sigma})

    def test_end(self, outputs):
        modelname = get_exp_name(self.opt)

        # calculate mean of nll and accuarcy
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        if self.opt.save_results:
            # concatenate UQ results
            probabilities = torch.stack([x['probabilities'] for x in outputs]).cpu().numpy()
            correct_predictions = torch.stack([x['correct_prediction'] for x in outputs]).cpu().numpy()
            correct = torch.stack([x['correct'] for x in outputs]).cpu().numpy()

            # concatenate and save thetas
            theta_path = 'theta_stats/' + modelname
            if 'stn' in self.opt.model.lower():
                theta_mu = torch.stack([x['theta_mu'] for x in outputs]).cpu().numpy()
                pickle.dump(theta_mu, open(theta_path + '_mu.p', 'wb'))
            if self.opt.model.lower() == 'pstn':
                theta_sigma = torch.stack([x['theta_sigma'] for x in outputs]).cpu().numpy()
                pickle.dump(theta_sigma, open(theta_path + '_sigma.p', 'wb'))

            # save UQ results
            UQ_path = 'UQ/' + modelname
            results = {'probabilities': probabilities, 'correct_prediction': correct_predictions,
                      'correct': correct}
            pickle.dump(results, open(UQ_path + '_results.p', 'wb'))

            # add to tensorboard
            tensorboard_logs = OrderedDict({'test_loss': avg_loss, 'test_acc': avg_acc})

            # write results to json file also
            save_results(self.opt, avg_loss, avg_acc)

        print('Done testing. Loss:', avg_loss.item(), 'Accuracy:', avg_acc.item())

        return OrderedDict({'test_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs})

    def configure_optimizers(self):

        # configure optimizer and scheduler
        optimizer, scheduler = create_optimizer(self.model, self.opt)

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):

        # initialize dataset
        if self.opt.optimize_temperature:  # learn optimal temperature on the validation data
            dataset = create_dataset(self.opt, mode='val')
        else:
            dataset = create_dataset(self.opt, mode='train')

        # dataloader params
        opt = {"batch_size": self.opt.batch_size, "shuffle": True, "pin_memory": True, "num_workers": int(self.opt.num_threads)}

        # return data loader
        dataloader = DataLoader(dataset, **opt)

        # if we use cyclic kl weigting we need to know how many batches for each epoch
        if self.opt.annealing.lower() == 'cyclic_kl':
            self.criterion.M = len(dataloader)

        return dataloader

    @pl.data_loader
    def val_dataloader(self):

        # initialize dataset
        dataset = create_dataset(self.opt, mode='val')

        # dataloader params
        opt = {"batch_size": self.opt.batch_size, "shuffle": False, "pin_memory": True, "num_workers": int(self.opt.num_threads), 'drop_last': True}

        # return data loader
        return DataLoader(dataset, **opt)

    @pl.data_loader
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
