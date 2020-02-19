
class CNN:
    def __init__(self, opt):

        if opt.basenet.lower() in ['inception', 'resnet50', 'resnet34', 'inception_v3']:
            from .inceptionclassifier import InceptionClassifier
            self.model = InceptionClassifier(opt)
        elif opt.basenet.lower() == 'celeba':
            from .simpleclassifier import SimpleClassifier
            self.model = SimpleClassifier(opt)
        elif opt.basenet.lower() == 'mnist':
            from .mnistclassifier import CNNClassifier
            self.model = CNNClassifier(opt)
        elif opt.basenet.lower() == 'timeseries':
            from .timeseriesclassifier import CCNClassifier
            self.model = CNNClassifier(opt)

    def forward(self, x):

        return self.model(x)
