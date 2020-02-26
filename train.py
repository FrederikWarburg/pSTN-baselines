import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from models import System
import torch
from utils.utils import get_exp_name

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
       create_git_tag=False
    )

    # initialize model
    model = System(opt)

    # use GPU if available
    num_gpus = torch.cuda.device_count()
    print("Let's use {} GPUS!".format(num_gpus))

    # use large batch sizes. We accumulate gradients to avoid memory issues
    num_batches = 256 // opt.batch_size

    # Initialize pytorch-lightning trainer with good defaults
    trainer = Trainer(max_nb_epochs=opt.epochs,
                      #accumulate_grad_batches=num_batches,
                      gpus=num_gpus,
                      early_stop_callback=None,
                      logger=logger,
                      val_check_interval=opt.val_check_interval,
                      val_percent_check=opt.val_percent_check,
                      distributed_backend='dp')

    # train model
    trainer.fit(model)

    # test model
    trainer.test()
