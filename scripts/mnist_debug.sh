#!/bin/sh

DATAPATH="/scratch/frwa/"
MODELS=("cnn" "stn" "pstn")
PARAMS=(0 4 4)
TEST_SAMPELS=(1 1 10)
CRITERION=("nll" "nll" "elbo")
ANNEALING=("None" "None" "reduce_kl")


for MODEL in 2
do
    echo ${MODELS[$MODEL]}
    echo $ATTR
    echo ${PARAMS[$MODEL]}
    echo ${TEST_SAMPELS[$MODEL]}
    echo ${CRITERION[$MODEL]}
    CUDA_VISIBLE_DEVICES=6 python train.py --dataroot 'data' \
                        --dataset "MNIST" \
                        --subset 1000 \
                        --fold 0 \
                        --batch_size 64 \
                        --num_classes 10  \
                        --num_threads 1 \
                        --epochs 600 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --N 1 \
                        --test_samples ${TEST_SAMPELS[$MODEL]} \
                        --train_samples 1 \
                        --criterion  ${CRITERION[$MODEL]} \
                        --save_results True \
                        --lr 0.001 \
                        --lr_loc 0.1 \
                        --beta_p 1. \
                        --num_param ${PARAMS[$MODEL]} \
                        --trainval_split True \
                        --save_results True \
                        --optimizer "adam" \
                        --weightDecay 0.01 \
                        --transformer_type "affine" \
                        --step_size 600 \
                        --val_check_interval 600 \
                        --results_folder "14_02_debug" \
                        --test_on "val" \
                        --annealing "weight_kl" \
                        --kl_weight 0.0001

done
