import os


def get_exp_name(opt):
    modelname = "d={}_m={}_b={}_n={}_p={}".format(opt.dataset, opt.model, opt.basenet, opt.N, opt.num_param)

    if opt.dataset.lower() == 'celeba':
        modelname += '_a=' + str(opt.target_attr)

    if opt.model.lower() == 'pstn':
        modelname += '_kl=' + opt.annealing
    else:
        modelname += '_kl=None'

    modelname += '_seed=' + str(opt.seed)
    modelname += '_sigmaP=' + str(opt.sigma_p)
    modelname += '_lr=' + str(opt.lr)

    if opt.model.lower() in ['stn', 'pstn']:
        modelname += '_lrloc=' + str(opt.lr_loc)
    else:
        modelname += '_lrloc=None'

    return modelname


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


