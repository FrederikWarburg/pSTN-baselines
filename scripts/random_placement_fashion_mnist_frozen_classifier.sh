#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1.)
TEST_SAMPLES=(1 1 10)
TRAIN_SAMPLES=(1 10 10)
VAR_INIT=(-50 -20 -12 -8 -4 -2)

for MODEL in {2..2}
do
    for var_init in {1..5}
    do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
     python train_parts.py --dataroot $DATAPATH \
                    --dataset "random_placement_fashion_mnist" \
                    --normalize False \
                    --crop_size  96 \
                    --normalize False \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 600 \
                    --step_size 200 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --num_param 2 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples ${TRAIN_SAMPLES[$MODEL]} \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.001 \
                    --lr_loc 0.1 \
                    --digits 1 \
        	        --optimizer 'adam' \
         	        --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --val_check_interval 100 \
                    --results_folder "23_02_random_placement_fashion_MNIST" \
                    --test_on "test" \
                    --annealing "weight_kl" \
                    --kl_weight 0.00003\
                    --pretrained_model_path 'checkpoints/22_02_fashion_mnist_robustness/d=fashion_mnist-m=cnn-p=2-kl=None-betaP=1-lr=0.001-lrloc=None.ckpt' \
                    --modeltype 'large_loc' \
                    --var_init ${VAR_INIT[$var_init]} \
                    --freeze_classifier \
                    --reduce_samples 'min' &
    done
done