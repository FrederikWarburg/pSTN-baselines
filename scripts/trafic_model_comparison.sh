#!/bin/sh

DATAPATH="/scratch/frwa/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
TEST_SAMPLES=(1 1 10)

for MODEL in 2
do
    echo $ATTR ${MODELS[$MODEL]}
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 python train.py --dataroot $DATAPATH \
            --dataset "mtsd" \
            --batch_size 10 \
            --num_classes 10 \
            --num_threads 4 \
            --epochs 15 \
            --step_size 10 \
            --seed 42 \
            --data_augmentation None \
            --model ${MODELS[$MODEL]} \
            --num_param 2 \
            --N 1 \
            --test_samples ${TEST_SAMPLES[$MODEL]} \
            --train_samples 1 \
            --criterion ${CRITERION[$MODEL]} \
            --lr 0.001 \
            --lr_loc 0.1 \
            --optimizer 'adam' \
            --trainval_split True \
            --save_results True \
            --theta_path 'theta_stats' \
            --val_check_interval 1 \
            --results_folder "21_02_mtsd_pstn_large_crop" \
            --test_on "val" \
            --weightDecay 0.01 \
            --annealing "weight_kl" \
            --kl_weight 0.0001 \
            --beta_p 1 \
            --bbox_size 1
done