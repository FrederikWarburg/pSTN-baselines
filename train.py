import os
from utils.utils import check_learnable_parameters


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from models import System
import torch
from utils.utils import get_exp_name, save_generating_thetas


if __name__ == '__main__':

    # get parameters
    opt = TrainOptions().parse()

    # decide unique model name based on parameters
    modelname = get_exp_name(opt)

    # initialize a test logger for experiment
    logger = TestTubeLogger(
       save_dir=os.getcwd() + "/lightning_logs",
       name=modelname,
       debug=False,
       create_git_tag=True
    )

    # initialize model
    lightning_system = System(opt)

    # use GPU if available
    num_gpus = torch.cuda.device_count()
    print("Let's use {} GPUS!".format(num_gpus))

    # use large batch sizes. We accumulate gradients to avoid memory issues
    if opt.dataset == 'MNIST':
      num_batches = None
    else:
      num_batches = 256 // opt.batch_size

    val_check_interval = opt.val_check_interval if opt.val_check_interval < 1 else int(opt.val_check_interval)
    # Initialize pytorch-lightning trainer with good defaults
    trainer = Trainer(max_epochs=opt.epochs,
                      # accumulate_grad_batches=num_batches,
                      gpus=num_gpus,
                      early_stop_callback=None,
                      logger=logger,
                      check_val_every_n_epoch=val_check_interval,
                      val_percent_check=opt.val_percent_check,
                      distributed_backend='dp',
                      checkpoint_callback=False)

    print('printing parameter check:')
    check_learnable_parameters(lightning_system.model, opt.model)

    if opt.resume_from_ckpt:
        print('Loading model.')
        lightning_system = lightning_system.load_from_checkpoint(checkpoint_path="checkpoints/%s.ckpt" % modelname)

    elif opt.optimize_temperature:
        print('Loading model.')
        lightning_system = lightning_system.load_from_checkpoint(checkpoint_path="checkpoints/%s.ckpt" % modelname)
        lightning_system.opt.optimize_temperature = True
        lightning_system.model.model.T.requires_grad = True
        lightning_system.configure_optimizers()
        trainer.fit(lightning_system)  # only fit temperature parameter here
    else:
        # train model
        trainer.fit(lightning_system)
        trainer.save_checkpoint("checkpoints/%s.ckpt" % modelname)

    # test model
    if opt.test_on == 'test':
        test_dataloader = lightning_system.test_dataloader()

    elif opt.test_on == 'val':
        test_dataloader = [lightning_system.val_dataloader()]

    trainer.test(lightning_system, test_dataloaders=test_dataloader)

    if opt.dataset == 'mnistxkmnist':
        save_generating_thetas(opt, test_dataloader)
