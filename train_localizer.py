from builtins import breakpoint
import os
from utils.utils import check_learnable_parameters


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from models import System
import torch
from utils.utils import get_exp_name, save_generating_thetas
import copy 


if __name__ == '__main__':

    # get parameters
    opt = TrainOptions().parse()

    # decide unique model name based on parameters
    modelname = get_exp_name(opt)

    # initialize a train logger for experiment
    logger = TensorBoardLogger(
       save_dir=os.getcwd() + "/lightning_logs/%s/" %opt.results_folder,
       name=modelname
    )

    # initialize model
    lightning_system = System(opt)
    # use GPU if available
    num_gpus = torch.cuda.device_count()
    print("Let's use {} GPUS!".format(num_gpus))
    val_check_interval = opt.val_check_interval if opt.val_check_interval < 1 else int(opt.val_check_interval)
    # Initialize pytorch-lightning trainer with good defaults
    if opt.subset in ['10', '20', '100']: # is the subsets are very small log every step
        logger_steps = 1
    else:
        logger_steps = 10
    trainer = Trainer(max_epochs=opt.epochs,
                    log_every_n_steps=logger_steps,
                      # accumulate_grad_batches=num_batches,
                      gpus=num_gpus,
                      logger=logger,
                      check_val_every_n_epoch=val_check_interval,
                      checkpoint_callback=False)
    print('Trainable parameters before freezing')
    check_learnable_parameters(lightning_system.model, opt.model)

    loader_opt = copy.deepcopy(opt)
    loader_opt.model = 'cnn'

    print('Loading pre-trained classifier model.')
    loaded_classifier = System.load_from_checkpoint(
           loader_opt.pretrained_model_path, opt=loader_opt)
    
    if opt.model == "cnn":
        lightning_system.model.net = loaded_classifier.model.net
        # freeze classifier parameters
        for param in lightning_system.model.net.parameters():
            param.requires_grad = False
    else:
        lightning_system.model.classifier.net = loaded_classifier.model.net
        # freeze classifier parameters
        for param in lightning_system.model.classifier.parameters():
            param.requires_grad = False
    
    print('Trainable parameters after freezing')
    check_learnable_parameters(lightning_system.model, opt.model)

    if not opt.model == 'cnn':
        # train model
        trainer.fit(lightning_system)
        trainer.save_checkpoint("checkpoints/%s/%s.ckpt" % (opt.results_folder, modelname))

    # test model
    if opt.test_on == 'test':
        test_dataloader = lightning_system.test_dataloader()

    elif opt.test_on == 'val':
        test_dataloader = lightning_system.val_dataloader()
        
    print('testing the following model:', lightning_system)
    trainer.test(lightning_system, test_dataloaders=test_dataloader)