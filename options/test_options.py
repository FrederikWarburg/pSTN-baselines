from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--num_visualizations', type=int, default=10, help='number of samples')
        

        self.is_train = False
        self.data_augmentation = False
        self.horizontal_flip = False
        self.no_shuffle = True  # no shuffle
