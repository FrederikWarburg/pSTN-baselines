import numpy as np
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--lr', type=float, default=0.1)
        self.parser.add_argument('--momentum', type=float, default=0)
        self.parser.add_argument('--weightDecay', type=float, default=1e-5)
        self.parser.add_argument('--optimizer', type=str, default='sgd')
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--criterion', type=str, default='crossentropy')
        self.parser.add_argument('--freeze_layers', type=int, default=np.inf)

        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--run_test_freq', type=int, default=1, help='frequency of running test in training script')
        self.parser.add_argument('--save_dir', type=str, default='', help='path for saving models')

        self.parser.add_argument('--no_vis', action='store_true', help='will not use tensorboard')

        self.is_train = True
