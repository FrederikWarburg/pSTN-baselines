import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from models import CoolSystem
import torch

if __name__ == '__main__':

    opt = TrainOptions().parse()

    logger = TestTubeLogger(
            save_dir=os.getcwd() + "/lightning_logs",
            name="{}_{}_{}_{}".format(opt.dataset, opt.model, opt.basenet, opt.N),
            debug=False,
            create_git_tag=False
    )

    model = CoolSystem(opt)

    num_gpus = torch.cuda.device_count()
    print("Let's use {} GPUS!".format(num_gpus))

    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=opt.epochs,
                      gpus=num_gpus,
                      early_stop_callback=None,
                      logger=logger,
                      val_check_interval=opt.val_check_interval,
                      val_percent_check=opt.val_percent_check,
                      distributed_backend='dp')
    trainer.fit(model)

    trainer.test()


########
# TRAIN ON MNIST DATASET (N = 1) (all models converge after approx 3 epochs)
########

# simple classifier
# python train2.py --dataroot ../data/ --model cnn --basenet simple --dataset mnist --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 10

# simple stn
# python train2.py --dataroot /scratch/s153847/ --model stn --basenet simple --dataset mnist --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 10 --step_size 100000 --smallest_size 64 --crop_size 64 --lr_loc 1e-02 --seed 42 --lr 0.1

# simple pstn
# python train2.py --dataroot /scratch/s153847/ --model pstn --basenet simple --dataset mnist --digits 1 --N 1 --train_samples 1 --test_samples 10 --batch_size 256 --num_classes 10 --step_size 100000 --smallest_size 64 --crop_size 64 --lr_loc 1e-02 --seed 42 --lr 0.1 --criterion elbo

########
# TRAIN ON MNIST DATASET (N = 2)
########

# simple classifier
# python train2.py --dataroot ../data/ --model cnn --basenet simple --dataset mnist --digits 2 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 100 --epochs 50

# simple stn
# python train2.py --dataroot /scratch/s153847/ --model stn --basenet simple --dataset mnist --digits 2 --N 2 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 100 --step_size 3 --smallest_size 64 --crop_size 64 --lr_loc 1e-02 --seed 42 --lr 0.1

# simple pstn
# python train2.py --dataroot /scratch/s153847/ --model pstn --basenet simple --dataset mnist --digits 2 --N 2 --train_samples 2 --test_samples 10 --batch_size 256 --num_classes 100 --step_size 3 --smallest_size 64 --crop_size 64 --lr_loc 1e-02 --seed 42 --lr 0.1 --criterion elbo

# Note that we received significantly better results samples 2 times during training. I conducted several experiments with only one sample during training, but haven't been able to achieve better results.


########
# TRAIN ON CUB DATASET (N = 1)
########

# simple classifier
# python train2.py --dataroot ../data/ --model cnn --basenet inception_v3 --dataset cub --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 200 --step_size 30 --smallest_size 256 --crop_size 224 --val_check_interval 0.004 --epochs 50 --freeze_layers 30 --data_augmentation True --horizontal_flip True --dropout_rate 0.7

# simple stn
# python train2.py --dataroot /scratch/s153847/ --model stn --basenet inception --dataset cub --digits 1 --N 1 --train_samples 1 --test_samples 1 --batch_size 256 --num_classes 200 --step_size 3 --smallest_size 256 --crop_size 224 --lr_loc 1e-02 --seed 42 --lr 0.1 --val_check_interval 0.004

# simple pstn
# python train2.py --dataroot /scratch/s153847/ --model pstn --basenet inception --dataset cub --digits 1 --N 1 --train_samples 1 --test_samples 10 --batch_size 256 --num_classes 200 --step_size 3 --smallest_size 256 --crop_size 224 --lr_loc 1e-02 --seed 42 --lr 0.1 --criterion elbo --val_check_interval 0.004
