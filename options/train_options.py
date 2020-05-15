import numpy as np

from .base_options import BaseOptions, str2bool


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # hyper parameter
        self.parser.add_argument('--lr', type=float, default=0.1)
        self.parser.add_argument('--lr_loc', type=float, default=1)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--weightDecay', type=float, default=1e-5)

        # optimizer related
        self.parser.add_argument('--optimizer', type=str, default='sgd')
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--step_size', type=float, default=50, help='Scheduler update every n (default 50) epochs')

        # loss related
        self.parser.add_argument('--moving_mean', type=str2bool, nargs='?', const=False, default=False)
        self.parser.add_argument('--criterion', type=str, default='nll')
        self.parser.add_argument('--annealing', type=str, default='no_annealing')

        # network related
        self.parser.add_argument('--freeze_layers', type=int, default=np.inf)

        # pre-processing related
        #self.parser.add_argument('--data_augmentation', type=bool, default=False)
        # self.parser.add_argument('--data_augmentation', type=str2bool, nargs='?', const=True, default=False, help="Activate data augmentation")
        self.parser.add_argument('--horizontal_flip', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--data_augmentation', type=str, default='None')

        # Evaluation related
        self.parser.add_argument('--val_percent_check', type=float, default=1.0, help='percentage of validation set to check')
        self.parser.add_argument('--val_check_interval', type=float, default=1.0, help='the rate for checking')
        self.parser.add_argument('--trainval_split', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--save_dir', type=str, default='', help='path for saving models')

        self.is_train = True
        self.no_shuffle = False  # shuffle
