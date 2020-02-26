import os
from os.path import join, isdir

def get_exp_name(opt):
    modelname = "d={}-m={}-b={}-n={}-p={}".format(opt.dataset, opt.model, opt.basenet, opt.N, opt.num_param)

    if opt.dataset.lower() == 'celeba':
        modelname += '-a=' + str(opt.target_attr)

    if opt.model.lower() == 'pstn':
        modelname += '-kl=' + opt.annealing
    else:
        modelname += '-kl=None'

    modelname += '-seed=' + str(opt.seed)
    modelname += '-sigmaP=' + str(opt.sigma_p)
    modelname += '-lr=' + str(opt.lr)

    if opt.model.lower() in ['stn', 'pstn']:
        modelname += '-lrloc=' + str(opt.lr_loc)
    else:
        modelname += '-lrloc=None'

    return modelname


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_results(opt, avg_loss, avg_acc):

    if opt.dataset.lower() == 'celeba':
        save_celeba(opt, avg_loss, avg_acc)

def save_celeba(opt, avg_loss, avg_acc):

    # get model name
    modelname = get_exp_name(opt)

    # remove attr from name and store it seperately
    modelname = modelname.split('-')
    attr = int(modelname[5].replace("a=", ""))
    modelname.pop(5)
    modelname = '-'.join(modelname)

    # make sure results dir exists
    if not isdir(opt.savepath):
        os.makedirs(opt.savepath)

    # open file with given model name and append results from specific target attr
    with open(join(opt.savepath, modelname + ".csv"), "a+") as f:
        f.write("{},{},{}\n".format(attr, avg_loss, avg_acc))
