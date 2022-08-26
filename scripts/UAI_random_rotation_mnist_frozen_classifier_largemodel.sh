#!/bin/sh

DATAPATH="data"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1.)
TEST_SAMPLES=(1 1 10)

for MODEL in {2..2}
do
    for w in {2..2}
    do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=13
    
     python train_localizer.py --dataroot $DATAPATH \
                    --dataset "random_rotation_mnist" \
                    --normalize False \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 100 \
                    --step_size 10 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --num_param 1 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples 10 \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.001 \
                    --lr_loc 1. \
                    --digits 1 \
        	        --optimizer 'adam' \
         	        --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --val_check_interval 100 \
                    --results_folder "30r_03_UAI_repros_rotMNIST_retry" \
                    --var_init -3 \
                    --test_on "test" \
                    --annealing "weight_kl" \
                    --kl_weight ${W_s[$w]} \
                    --pretrained_model_path 'checkpoints/29_03_UAI_repros_rotMNIST/d=MNIST-m=cnn-p=1-fold=0-kl=None-betaP=1-lr=0.001-lrloc=None.ckpt' \
                    --modeltype 'large_loc' 
    done
done
