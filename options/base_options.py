import argparse
import os

import torch

from utils import utils as util

TIMESERIESDATASETS = [
    'FaceAll', 'wafer', 'uWaveGestureLibrary_X', 'Two_Patterns',
    'StarLightCurves', 'PhalangesOutlinesCorrect', 'FordA']


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', required=True, help='path to images')
        self.parser.add_argument('--dataset', default='cub', help='which dataset to use')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples per epoch')
        self.parser.add_argument('--num_classes', type=int, default=200, help='Maximum number of classes per epoch')
        self.parser.add_argument('--download', action='store_true', help='download dataset')
        self.parser.add_argument('--no_shuffle', action='store_true', help='if true shuffle')
        self.parser.add_argument('--smallest_size', type=int, default=256, help='smallest side of input images')
        self.parser.add_argument('--crop_size', type=int, default=224, help='smallest side of input images')
        self.parser.add_argument('--digits', type=int, default=1, help='number of digits in mnist dataset')
        self.parser.add_argument('--target_attr', type=int, default=1, help='attribute to train for')
        self.parser.add_argument('--transformer_type', type=str, default='affine', help='attribute to train for')
        self.parser.add_argument('--num_param', default=2, type=int,
                                 help='if we use a affine (s, r, tx, ty) or crop (0.5, 1, tx, ty) transformation')
        self.parser.add_argument('--save_results', type=bool, default=False, help='should we save the results?')
        self.parser.add_argument('--savepath', type=str, default='results', help='where should we save the results?')
        # data params - MNIST subset experiment
        self.parser.add_argument('--subset', type=str, default=None, help='using a subset of MNIST? What size?')
        self.parser.add_argument('--fold', type=str, default=None, help='using a subset of MNIST? Which fold?')

        # model params
        self.parser.add_argument('--sigma_p', type=float, default=0.1, help='prior variance')
        self.parser.add_argument('--sigma_n', type=float, default=1e-5, help='color space noise variance')

        # network params
        self.parser.add_argument('--model', type=str, default='cnn', help='model name')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--resume_ckpt', type=str, default=None, help='path to pretrained model')
        self.parser.add_argument('--dropout_rate', type=float, default=0.5)
        self.parser.add_argument('--N', type=int, default=1, help='number of parallel tracks')
        self.parser.add_argument('--test_samples', type=int, default=1, help='number of samples')
        self.parser.add_argument('--train_samples', type=int, default=1, help='number of samples')
        self.parser.add_argument('--basenet', type=str, help='base network to use')

        # general params
        self.parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        self.parser.add_argument('--name', type=str, default='debug',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')

        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='',
                                 help='exports intermediate collapses to this folder')
        self.parser.add_argument('--heatmap', type=bool, default=False, help='visualize bbox as heat map or bbox')
        #
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test
        if not self.opt.is_train:
            self.opt.data_augmentation = self.data_augmentation  # train or test
            self.opt.horizontal_flip = self.horizontal_flip  # train or test
        self.opt.no_shuffle = self.no_shuffle  # train or test
        self.opt.xdim = 1 if self.opt.dataset in TIMESERIESDATASETS else 2

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        self.opt.TIMESERIESDATASETS = TIMESERIESDATASETS

        return self.opt
