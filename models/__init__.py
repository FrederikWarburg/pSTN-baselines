from os.path import join
import torch
from torch.nn import functional as F
from data import DataLoader
from utils.evaluate import accuracy
import pytorch_lightning as pl
from utils.visualizations import visualize_stn
from loss import create_criterion

def create_model(opt):
    if opt.model.lower() == 'cnn':
        if opt.basenet.lower() == 'inception':
            from .inceptionclassifier import InceptionClassifier
            model = InceptionClassifier(opt)
        elif opt.basenet.lower() == 'simple':
            from .simpleclassifier import SimpleClassifier
            model = SimpleClassifier(opt)
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
                {'params': model.stn.parameters(),        'lr': opt.lr_loc*opt.lr},
                {'params': model.classifier.parameters(), 'lr': opt.lr},
            ], momentum=opt.momentum, weight_decay=opt.weightDecay)
        elif opt.model.lower() == 'pstn' and opt.lr_loc > 0:
            # the learning rate of the parameters that are part of the localizer are multiplied 1e-4
            optimizer = torch.optim.SGD([
                {'params': model.pstn.parameters(),       'lr': opt.lr_loc*opt.lr},
                {'params': model.classifier.parameters(), 'lr': opt.lr},
            ], momentum=opt.momentum, weight_decay=opt.weightDecay)
        else:
            print("=> SGD all parameters chosen")
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

    else:
        raise ValueError('Unsupported or optimizer: {}!'.format(opt.optimizer))

    return optimizer, scheduler

def save_network(model, opt, which_epoch, is_best = False):
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



class CoolSystem(pl.LightningModule):

    def __init__(self, opt):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.model = create_model(opt)
        self.hparams = opt
        self.opt = opt
        self.criterion = create_criterion(opt)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx, hidden = 0):

        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat,y)

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

        grid_in, grid_out, theta = visualize_stn(self.model, x, self.opt)

        return {'val_loss': loss, 'val_acc': acc, 'grid_in': grid_in, 'grid_out': grid_out, 'theta': theta}

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        self.logger.experiment.add_image('grid_in', outputs[0]['grid_in'], self.global_step)

        if self.opt.model.lower() in ['stn', 'pstn']:
            self.logger.experiment.add_image('grid_out', outputs[0]['grid_out'], self.global_step)

            if self.opt.model.lower() == 'stn':
                mu_mean = torch.stack([x['theta'] for x in outputs]).mean(dim=0).mean(dim=0)
                mu_std = torch.stack([x['theta'] for x in outputs]).std(dim=0).std(dim=0)
            elif self.opt.model.lower() == 'pstn':
                mu_mean = torch.stack([torch.stack([param for param in x['theta'][0]]) for x in outputs]).mean(dim=0).mean(dim=0)
                mu_std = torch.stack([torch.stack([param for param in x['theta'][0]]) for x in outputs]).std(dim=0).std(dim=0)
                sigma_mean = torch.stack([torch.stack([param for param in x['theta'][1]]) for x in outputs]).mean(dim=0).mean(dim=0)
                sigma_std = torch.stack([torch.stack([param for param in x['theta'][1]]) for x in outputs]).std(dim=0).std(dim=0)

            for i in range(len(mu_mean)):
                self.logger.experiment.add_scalar('mu_mean_'.format(i), mu_mean[i], self.global_step)
                self.logger.experiment.add_scalar('mu_std_'.format(i), mu_std[i], self.global_step)

                if self.opt.model.lower() == 'pstn':
                    self.logger.experiment.add_scalar('sigma_mean_'.format(i), sigma_mean[i], self.global_step)
                    self.logger.experiment.add_scalar('sigma_std_'.format(i), sigma_std[i], self.global_step)

        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar':tensorboard_logs}

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

        return {'test_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar':tensorboard_logs}

    def configure_optimizers(self):

        optimizer, scheduler = create_optimizer(self.model, self.opt)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.opt, train=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.opt, val=True)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.opt, test=True)
