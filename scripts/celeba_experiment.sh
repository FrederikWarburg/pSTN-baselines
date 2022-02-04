#!/bin/sh

DATAPATH="/data"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")


for ATTR in 23
do
    for MODEL in 2
    do
        echo ${MODELS[$MODEL]}
        OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train.py --dataroot $DATAPATH \
                --dataset "celeba" \
                --target_attr $ATTR \
                --crop_size  64 \
                --batch_size 256 \
                --num_classes 10 \
                --num_threads 2 \
                --epochs 15 \
                --step_size 10 \
                --seed 42 \
                --data_augmentation None \
                --model ${MODELS[$MODEL]} \
                --num_param 2 \
                --N 1 \
                --test_samples 1 \
                --train_samples 10 \
                --criterion ${CRITERION[$MODEL]} \
                --lr 0.001 \
                --lr_loc 0.01 \
                --digits 1 \
                --optimizer 'adam' \
                --trainval_split True \
                --save_results True \
                --theta_path 'theta_stats' \
                --download True \
                --lr_loc 0.01 \
                --val_check_interval 15 \
                --results_folder "04_02_celeba_debug" \
                --test_on "val" \
                --annealing "weight_kl" \
                --kl_weight 0.0003
                --basenet 'resnet34' &
    done
done
