import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from options.test_options import TestOptions
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from models import System
import torch
from utils.utils import get_exp_name

if __name__ == '__main__':

    # get parameters
    opt = TestOptions().parse()

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


    # Initialize pytorch-lightning trainer with good defaults
    trainer = Trainer(gpus=num_gpus,
                      logger=logger,
                      distributed_backend='dp')

    # test model
    trainer.test()
