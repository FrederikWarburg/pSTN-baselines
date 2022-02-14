#!/bin/sh

SUBSETS=(10 30 100 1000 3000 10000)
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
LRS=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)

for LR in {0..6}
do
    for FOLD in {0..4} # only do 2 folds for the grid search to limit computation time
    do
        for MODEL in {0..0}
        do 
            CUDA_VISIBLE_DEVICES=2 python train.py --dataroot 'data' \
                                --dataset "MNIST" \
                                --subset 100 \
                                --fold ${FOLD} \
                                --batch_size 16 \
                                --num_classes 10  \
                                --num_threads 1 \
                                --epochs 600 \
                                --seed 42 \
                                --model ${MODELS[$MODEL]} \
                                --num_param ${PARAMS[$MODEL]} \
                                --N 1 \
                                --test_samples 10 \
                                --train_samples 1 \
                                --criterion  ${CRITERION[$MODEL]} \
                                --save_results True \
                                --lr ${LRS[$LR]} \
                                --lr_loc 1 \
                                --beta_p 1. \
                                --trainval_split True \
                                --save_results True \
                                --optimizer "adam" \
                                --weightDecay 0.01 \
                                --transformer_type "affine" \
                                --step_size 600 \
                                --val_check_interval 600 \
                                --results_folder "07_02_single_lr_exp" \
                                --test_on "test" \
                                --annealing "weight_kl" \
                                --kl_weight 0.0003
        done
    done
done
