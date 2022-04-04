import os
from os.path import join, isdir, exists
import pickle
import torch

def get_exp_name_timeseries(opt):
    modelname = 'd=%s-m=%s-p=0-fold=%s-kl=None-betaP=1-lr=0.001-lrloc=0.1' %(opt.dataset, opt.model, opt.fold)
    return modelname

def get_exp_name(opt):
    modelname = "d={}-m={}-p={}".format(opt.dataset, opt.model, opt.num_param)

    if opt.subset is not None:
        modelname = "d={}{}-m={}-p={}".format(opt.dataset, opt.subset, opt.model, opt.num_param)

    # if 'RandAugment' in opt.data_augmentation:
    #     modelname += '-randaug_N=%s-randaug_M=%s' % (opt.rand_augment_N, opt.rand_augment_M)

    if opt.fold is not None:
        modelname += '-fold=' + str(opt.fold)
        
    if opt.dataset.lower() == 'celeba':
        modelname += '-a=' + str(opt.target_attr)

    if opt.add_kmnist_noise:
        modelname += '-kmnist_noise_'

    if opt.model.lower() == 'pstn':
        modelname += '-kl=' + opt.annealing
        if opt.annealing == 'weight_kl':
            modelname += '_' + str(opt.kl_weight)
    else:
        modelname += '-kl=None'

    #if opt.model.lower() == 'pstn':
    modelname += '-betaP=' + str(opt.beta_p)[0]

    modelname += '-lr=' + str(opt.lr)

    if opt.model.lower() in ['stn', 'pstn']:
        modelname += '-lrloc=' + str(opt.lr_loc)
    else:
        modelname += '-lrloc=None'

    if opt.train_samples > 1:
        modelname += '-trainS=' + str(opt.train_samples)

    # if opt.init_large_variance:
    #     modelname += '-init_large_var'
    if opt.model.lower() in ['pstn', 'stn'] and opt.var_init != -2:
        modelname += '-varinit=' + str(opt.var_init)

    if opt.modeltype in ['large_loc', '2xlarge_loc']:
        modelname += '_' + str(opt.modeltype) 

    if opt.reduce_samples == 'min':
        modelname += '_min_agg'

    if opt.upsample_oldies:
        modelname += '_upsample_oldies=%s' %opt.desired_rate

    if opt.upsample_attractive_oldies:
        modelname += '_upsample_attr_oldies=%s' %opt.desired_rate

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
    results_dir = 'experiments/%s/' % opt.results_folder
    mkdir(results_dir)
    model_name = get_exp_name(opt)
    RESULTS_PATH = results_dir + model_name
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
    results_dir = f'experiments/{opt.results_folder}/{opt.target_attr}'
    mkdir(results_dir)
    model_name = get_exp_name(opt)
    RESULTS_PATH = results_dir + model_name
    pickle.dump(avg_acc.cpu().numpy(), open(RESULTS_PATH + '_test_accuracy.p', 'wb'))
    pickle.dump(avg_loss.cpu().numpy(), open(RESULTS_PATH + '_test_loss.p', 'wb'))


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
        if outputs[0]['theta_mu'] is not None: # DA upsampling experiment (mean id. / None)
            theta_mu = torch.stack([x['theta_mu'] for x in outputs]).cpu().numpy()
            pickle.dump(theta_mu, open(theta_path + '_mu.p', 'wb'))
    if opt.model.lower() == 'pstn':
        if outputs[0]['beta'] is not None:
            beta = torch.stack([x['beta'] for x in outputs]).cpu().numpy()
            pickle.dump(beta, open(theta_path + '_beta.p', 'wb'))

    if outputs[0]['ground_truth_trafo'] is not None:
        target_trafo = torch.stack([x['ground_truth_trafo'] for x in outputs]).cpu().numpy()
        pickle.dump(target_trafo, open(theta_path + '_ground_truth_trafo.p', 'wb'))

def save_UQ_results(opt, probabilities, correct_predictions, correct):
    modelname = get_exp_name(opt)
    UQ_path = 'experiments/%s/UQ/%s/' % (opt.results_folder, modelname)
    results = {'probabilities': probabilities, 'correct_prediction': correct_predictions,
             'correct': correct}
    if not exists(UQ_path):
        mkdir(UQ_path)
    pickle.dump(results, open(UQ_path + 'UQ_results.p', 'wb'))



def get_feature_size(img_size, model):    
    classifier_feature_sizes ={
                32: 160,
                64: 640,
                96: 2560
             }
    localiser_feature_size ={
                32: 128,
                64: 512,
                96: 2048
             }
    if model == 'classifier':
        return classifier_feature_sizes[img_size]
    if model == 'localiser':
        return localiser_feature_size[img_size]
