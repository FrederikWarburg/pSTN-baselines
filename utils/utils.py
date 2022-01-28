import os
from os.path import join, isdir, exists
import pickle
import torch


def get_exp_name(opt):
    modelname = "d={}-m={}-p={}".format(opt.dataset, opt.model, opt.num_param)

    if opt.subset is not None:
        modelname = "d={}{}-m={}-p={}".format(opt.dataset, opt.subset, opt.model, opt.num_param)

    if opt.fold is not None:
        modelname += '-fold=' + str(opt.fold)

    if opt.dataset.lower() == 'celeba':
        modelname += '-a=' + str(opt.target_attr)

    if opt.model.lower() == 'pstn':
        modelname += '-kl=' + opt.annealing
        if opt.annealing == 'weight_kl':
            modelname += '_' + str(opt.kl_weight)
    else:
        modelname += '-kl=None'

    modelname += '-betaP=' + str(opt.beta_p)
    modelname += '-lr=' + str(opt.lr)

    if opt.model.lower() in ['stn', 'pstn']:
        modelname += '-lrloc=' + str(opt.lr_loc)
    else:
        modelname += '-lrloc=None'

    if opt.learnable_prior:
        modelname += '-learnable_prior'

    return modelname


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_results(opt, avg_loss, avg_acc):
    if opt.dataset.lower() == 'celeba':
        save_celeba(opt, avg_loss, avg_acc)
    if "mnist" in opt.dataset.lower():
        save_mnist(opt, avg_loss, avg_acc)
    if opt.dataset in opt.TIMESERIESDATASETS:
        save_timeseries(opt, avg_loss, avg_acc)


def save_timeseries(opt, avg_loss, avg_acc):
    mkdir('experiments/%s' % opt.results_folder)
    RESULTS_PATH = 'experiments/%s/%s_%s_betaP=%s_fold_%s_DA=%s_' % (
        opt.results_folder, opt.model, opt.dataset, opt.beta_p, opt.fold, opt.data_augmentation)
    pickle.dump(avg_acc.cpu().numpy(), open(RESULTS_PATH + 'test_accuracy.p', 'wb'))
    pickle.dump(avg_loss.cpu().numpy(), open(RESULTS_PATH + 'test_loss.p', 'wb'))


def save_mnist(opt, avg_loss, avg_acc):
    results_dir = 'experiments/%s/' % opt.results_folder
    mkdir(results_dir)
    model_name = get_exp_name(opt)
    RESULTS_PATH = results_dir + model_name
    pickle.dump(avg_acc.cpu().numpy(), open(RESULTS_PATH + '_test_accuracy.p', 'wb'))
    pickle.dump(avg_loss.cpu().numpy(), open(RESULTS_PATH + '_test_loss.p', 'wb'))


def save_celeba(opt, avg_loss, avg_acc):

    # get model name
    modelname = get_exp_name(opt)

    # remove attr from name and store it seperately
    modelname = modelname.split('-')
    attr = int(modelname[5].replace("a=", ""))
    modelname.pop(5)
    modelname = '-'.join(modelname)

    # make sure results dir exists
    mkdir('experiments/%s' % opt.results_folder)

    # open file with given model name and append results from specific target attr
    with open(join('experiments/%s/' % opt.results_folder, modelname + ".csv"), "a+") as f:
        f.write("{},{},{}\n".format(attr, avg_loss, avg_acc))


def check_learnable_parameters(model, architecture):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print([(p[0], p[1].numel()) for p in model.named_parameters() if p[1].requires_grad])
    print('Number of trainable parameters for %s:' %architecture, pytorch_total_params)


def save_generating_thetas(opt, dataloader):
    modelname = get_exp_name(opt)

    # concatenate and save thetas
    theta_path = 'experiments/%s/theta_stats/%s/' % (opt.results_folder, modelname)
    if not exists(theta_path):
        mkdir(theta_path)

    pickle.dump(dataloader.dataset.samples, open(theta_path + 'generating_thetas.p', 'wb'))


def save_learned_thetas(opt, outputs, mode='train', epoch=None):
    modelname = get_exp_name(opt)
    if mode == 'train':
        mode_and_epoch = 'train_epoch_' + str(epoch)
    if mode == 'test':
        mode_and_epoch = 'test'
    # concatenate and save thetas
    theta_path = 'experiments/%s/theta_stats/%s/%s' % (opt.results_folder, modelname, mode_and_epoch)
    if not exists(theta_path):
        mkdir(theta_path)

    if 'stn' in opt.model.lower():
        theta_mu = torch.stack([x['theta_mu'] for x in outputs]).cpu().numpy()
        pickle.dump(theta_mu, open(theta_path + '_mu.p', 'wb'))
    if opt.model.lower() == 'pstn':
        beta = torch.stack([x['beta'] for x in outputs]).cpu().numpy()
        pickle.dump(beta, open(theta_path + '_beta.p', 'wb'))


def save_UQ_results(opt, probabilities, correct_predictions, correct):
    modelname = get_exp_name(opt)
    UQ_path = 'experiments/%s/UQ/%s/' % (opt.results_folder, modelname)
    results = {'probabilities': probabilities, 'correct_prediction': correct_predictions,
             'correct': correct}
    if not exists(UQ_path):
        mkdir(UQ_path)
    pickle.dump(results, open(UQ_path + 'UQ_results.p', 'wb'))