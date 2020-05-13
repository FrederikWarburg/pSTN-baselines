import os
from os.path import join, isdir
import pickle


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
    if "mnist" in opt.dataset.lower():
        MNIST_RESULTS_PATH = 'experiments/mnist_results/%s_mnist%s_%s_fold_%s_DA=%s_%s_' %(
            opt.model, opt.subset, opt.sigma_p, opt.fold, opt.data_augmentation, opt.transformer_type)
        pickle.dump(avg_acc.cpu().numpy(), open(MNIST_RESULTS_PATH + 'test_accuracy.p', 'wb'))
        pickle.dump(avg_loss.cpu().numpy(), open(MNIST_RESULTS_PATH + 'test_loss.p', 'wb'))


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


def check_learnable_parameters(model, architecture):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print([(p[0], p[1].numel()) for p in model.named_parameters() if p[1].requires_grad])
    print('Number of trainable parameters for %s:' %architecture, pytorch_total_params)
