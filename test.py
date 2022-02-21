import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# from options.test_options import TestOptions TODO: clean this up
from options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from models import System
import torch
from utils.utils import get_exp_name, mkdir
import pickle

if __name__ == '__main__':

    # get parameters
    opt = TrainOptions().parse()

    # decide unique model name based on parameters
    modelname = get_exp_name(opt)

    # initialize a test logger for experiment
    logger = TensorBoardLogger(
       save_dir=os.getcwd() + "/lightning_logs/%s/" %opt.results_folder,
       name=modelname
    )

    # initialize model
    lightning_system = System(opt)

    # use GPU if available
    num_gpus = torch.cuda.device_count()
    print("Let's use {} GPUS!".format(num_gpus))


    # Initialize pytorch-lightning trainer with good defaults
    trainer = Trainer(gpus=num_gpus,
                      logger=logger)


    print('\n Loading model: ', modelname)

    lightning_system_loaded = System.load_from_checkpoint(
            "checkpoints/%s/%s.ckpt" % (opt.results_folder, modelname), opt=opt)

    lightning_system.model = lightning_system_loaded.model

    # test model
    results = trainer.test(lightning_system)

    # save results 
    results_dir = 'experiments/%s/test_performance/' % opt.results_folder
    RESULTS_PATH = results_dir + modelname
    mkdir(RESULTS_PATH)
    pickle.dump(results, open(RESULTS_PATH + '/test_performance.p', 'wb'))
