import argparse
import os

import torch
import numpy as np

from utils import utils as util

TIMESERIESDATASETS = [
    'FaceAll', 'wafer', 'uWaveGestureLibrary_X', 'Two_Patterns',
    'StarLightCurves', 'PhalangesOutlinesCorrect', 'FordA']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', required=True, help='path to images')
        self.parser.add_argument('--dataset', default='cub', help='which dataset to use')
        self.parser.add_argument('--max_dataset_size', type=int, default=float(np.inf),
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
        # data params - MNIST subset experiment
        self.parser.add_argument('--subset', type=str, default=None, help='using a subset of MNIST? What size?')
        self.parser.add_argument('--fold', type=str, default=None, help='using a subset of MNIST? Which fold?')
        self.parser.add_argument('--add_kmnist_noise', type=bool, default=False, help='add kmnist noise')
        self.parser.add_argument('--normalize', type=str2bool, default=True, help='should we normalize MNIST data?')

        # model params
        self.parser.add_argument('--alpha_p', type=float, default=1, help='prior alpha (posterior when fixed)')
        self.parser.add_argument('--beta_p', type=float, default=1, help='prior beta')
        self.parser.add_argument('--criterion', type=str, default='nll')
        self.parser.add_argument('--annealing', type=str, default='no_annealing')
        self.parser.add_argument('--kl_weight', type=float, default=1.)
        self.parser.add_argument('--reduce_samples', type=str, default='mean')

        # network params
        self.parser.add_argument('--model', type=str, default='cnn', help='model name')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        #self.parser.add_argument('--resume_ckpt', type=str, default=None, help='path to pretrained model')
        self.parser.add_argument('--resume_from_ckpt', type=str2bool, nargs='?', const=True, default=False, help='Load pre-trained model?')
        self.parser.add_argument('--pretrained_model_path', type=str, default=None, help='frozen classifier exp; where to load from')
        self.parser.add_argument('--modeltype', type=str, default='')
        self.parser.add_argument('--modeltype_classifier', type=str, default='')
        self.parser.add_argument('--init_large_variance', type=str2bool, default=False)
        self.parser.add_argument('--var_init', type=float, default=-20.0)


        self.parser.add_argument('--dropout_rate', type=float, default=0.5)
        self.parser.add_argument('--N', type=int, default=1, help='number of parallel tracks')
        self.parser.add_argument('--test_samples', type=int, default=1, help='number of samples')
        self.parser.add_argument('--train_samples', type=int, default=1, help='number of samples')
        self.parser.add_argument('--basenet', type=str, help='base network to use')
        self.parser.add_argument('--freeze_classifier', type=str2bool, nargs='?', const=True, default=False)

        # logger params 
        self.parser.add_argument('--save_results', type=bool, default=False, help='should we save the results?')
        self.parser.add_argument('--save_training_theta', type=bool, default=False, help='should we save thetas during training?')
        self.parser.add_argument('--results_folder', type=str, default='results', help='where should we save the results?')
        
        # general params
        self.parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        self.parser.add_argument('--name', type=str, default='debug',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        self.parser.add_argument('--test_on', type=str, default='test', help='evaluate on validation or test?')
        self.parser.add_argument('--check_already_run', type=bool, default=False, help='check whether this config has already been run')
        

        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='',
                                 help='exports intermediate collapses to this folder')
        self.parser.add_argument('--heatmap', type=bool, default=False, help='visualize bbox as heat map or bbox')
        self.parser.add_argument('--bbox_size', default=0, type=int, help='the sizes of the bounding box around the bounding box for mtsd')

        #
        self.initialized = True



    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        if self.opt.is_train:
            #self.opt.data_augmentation = str2bool(self.opt.data_augmentation)
            self.opt.horizontal_flip = str2bool(self.opt.horizontal_flip )
        else:
            self.opt.data_augmentation = self.data_augmentation
            self.opt.horizontal_flip = self.horizontal_flip

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
