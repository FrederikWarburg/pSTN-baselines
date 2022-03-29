#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1.)
LRs=(1e-5 1e-4 1e-3 1e-2)
TEST_SAMPLES=(1 1 10)
P=(2 4)

for MODEL in {2..2} 
do
    for w in {2..2} # this is taken from rotMNIST experiment, not sure we need to finetune 
    do
        for lr in {0..3}
        do
        echo $MODEL
        echo ${MODELS[$MODEL]}
        OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=7 python train.py \
                        --dataroot $DATAPATH \
                        --dataset "random_placement_fashion_mnist" \
                        --crop_size  96 \
                        --batch_size 64 \
                        --num_classes 10 \
                        --num_threads 2 \
                        --epochs 100 \
                        --step_size 10 \
                        --seed 42 \
                        --data_augmentation None \
                        --model ${MODELS[$MODEL]} \
                        --modeltype 'large_loc' \
                        --num_param 4 \
                        --N 1 \
                        --test_samples ${TEST_SAMPLES[$MODEL]} \
                        --train_samples 10 \
                        --criterion ${CRITERION[$MODEL]} \
                        --lr ${LRs[$lr]}  \
                        --lr_loc 0.1 \
                        --digits 1 \
                        --optimizer 'adam' \
                        --trainval_split True \
                        --save_results True \
                        --theta_path 'theta_stats' \
                        --download True \
                        --val_check_interval 100 \
                        --annealing "weight_kl" \
                        --kl_weight ${W_s[$w]} \
                        --results_folder "22_02_fashion_mnist_robustness"  \
                        --test_on "test" \
                        --normalize False \
                        --step_size 50 \
                        --init_large_variance True
        done
    done
done
