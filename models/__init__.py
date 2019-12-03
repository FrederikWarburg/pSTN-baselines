def create_model(opt):
    if opt.model.lower() == 'inception':
        from .networks import InceptionClassifier
        model = InceptionClassifier(opt)

    return model
