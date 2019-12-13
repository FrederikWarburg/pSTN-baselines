import os
import time
import numpy as np
from .utils import denormalize, add_bounding_boxes
from options.test_options import TestOptions
from data import DataLoader
import torch


try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None

class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        #
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_time(self, epoch, i, time_to_fetch_data, time_model):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('time/fetching_data', time_to_fetch_data, iters)
            self.display.add_scalar('time/forward_and_backward_pass', time_model, iters)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc, mode):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, {} ACC: [{:.5} %]\n' \
            .format(epoch, mode.upper(), acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch, mode):
        if self.display:
            self.display.add_scalar('data/{}_acc'.format(mode), acc, epoch)

    def plot_theta(self, image_id, theta, epoch):

        type_ = 'crop' if len(theta) == 2 else 'affine'
        theta = theta.cpu().numpy().reshape(-1)

        for i, val in enumerate(theta):
            self.display.add_scalar('theta_{}/{}_{}'.format(image_id, type_, i), val, epoch)


    def visualize_transformation(self, model, epoch):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('Running Vizualization')
        opt = TestOptions().parse()
        opt.max_dataset_size = 20
        opt.batch_size = 1

        num_param = 2 if opt.fix_scale_and_rot else 4

        dataset = DataLoader(opt)
        model.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(dataset):
                input = input.to(device)
                x_stn, theta = model.stn(input)
                self.plot_theta(i, theta, epoch)

                for im, crop in zip(input, theta):
                    im = np.transpose(im.cpu().numpy(),(1,2,0))
                    im = denormalize(im)

                    im = add_bounding_boxes(im, theta, num_param)

                    im = np.transpose(im, (2,0,1))
                    self.display.add_image("input_{}/input".format(i), im, epoch)

                print("==> theta", theta)

                for j,x in enumerate(x_stn):

                    x = np.transpose(x.cpu().numpy(),(1,2,0))
                    x = denormalize(x)
                    x = np.transpose(x, (2,0,1))

                    self.display.add_image("input_{}/parallel_{}".format(i,j+1), x, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
