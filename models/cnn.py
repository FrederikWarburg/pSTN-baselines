class CNN:
    def __init__(self, opt):

        if opt.dataset.lower() == 'cub':
            from .cubclassifier import CubClassifier
            self.model = CubClassifier(opt)
        elif opt.dataset.lower() == ['celeba','mnistxkmnist']:
            from .celebaclassifier import CelebaClassifier
            self.model = CelebaClassifier(opt)
        elif opt.dataset.lower() == 'mnist':
            from .mnistclassifier import CNNClassifier
            self.model = CNNClassifier(opt)
        elif opt.dataset.lower() == 'timeseries':
            from .timeseriesclassifier import TimeseriesClassifier
            self.model = TimeseriesClassifier(opt)

    def forward(self, x):

        return self.model(x)
