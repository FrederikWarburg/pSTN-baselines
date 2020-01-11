import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from models import CoolSystem

if __name__ == '__main__':

    opt = TrainOptions().parse()

    logger = TestTubeLogger(
            save_dir=os.getcwd() + "/lightning_logs",
            name="{}_{}_{}".format(opt.dataset, opt.model, opt.basenet),
            debug=False,
            create_git_tag=False
    )

    model = CoolSystem(opt)

    # most basic trainer, uses good defaults
    trainer = Trainer(logger, val_check_interval=opt.val_check_interval, val_percent_check=opt.val_percent_check)
    trainer.fit(model)

    trainer.test()


########
# TRAIN ON MNIST DATASET
########

# simple classifier
# python train2.py --dataroot ../data/ --model cnn --basenet simple --dataset mnist --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256

# simple stn
# python train2.py --dataroot /scratch/s153847/ --model stn --basenet simple --dataset mnist --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 10 --step_size 100000 --smallest_size 64 --crop_size 64 --lr_loc 1e-02 --seed 42 --lr 0.1

# simple pstn
# python train2.py --dataroot /scratch/s153847/ --model pstn --basenet simple --dataset mnist --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 10 --step_size 100000 --smallest_size 64 --crop_size 64 --lr_loc 1e-02 --seed 42 --lr 0.1 --criterion elbo
