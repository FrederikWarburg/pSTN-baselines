from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        self.is_train = False
        self.data_augmentation = False
        self.no_shuffle = True  # no shuffle
