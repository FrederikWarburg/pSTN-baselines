from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--test_samples', type=int, default=10, help='number of samples')

        self.is_train = False
        self.data_augmentation = False
        self.horizontal_flip = False
        self.no_shuffle = True  # no shuffle
