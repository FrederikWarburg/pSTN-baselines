#!/bin/sh

DATAPATH="/scratch/frwa/"
MODELS=("stn" "pstn" "pstn" "pstn")
CRITERION=("nll" "elbo" "elbo" "elbo")
TEST_SAMPLES=(1 10 10 10)
TRAIN_SAMPLES=(1 1 10 10)
GPU=(7 6)
KL=(1e-7 1e-7 1e-6 1e-7) 

for MODEL in 0
do
    echo $ATTR ${MODELS[$MODEL]}
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU[$MODEL]} python train.py --dataroot $DATAPATH \
            --dataset "mtsd" \
            --batch_size 10 \
            --num_classes 10 \
            --num_threads 4 \
            --epochs 60 \
            --step_size 20 \
            --seed 42 \
            --data_augmentation None \
            --model ${MODELS[$MODEL]} \
            --num_param 4 \
            --N 1 \
            --test_samples ${TEST_SAMPLES[$MODEL]} \
            --train_samples ${TRAIN_SAMPLES[$MODEL]} \
            --criterion ${CRITERION[$MODEL]} \
            --lr 0.0001 \
            --lr_loc 0.1 \
            --optimizer 'adam' \
            --trainval_split True \
            --save_results True \
            --theta_path 'theta_stats' \
            --val_check_interval 1 \
            --results_folder "24_02_unfrozen_classifier_small_init" \
            --test_on "val" \
            --weightDecay 0 \
            --annealing "weight_kl" \
            --kl_weight ${KL[$MODEL]} \
            --beta_p 1 \
            --bbox_size 1 \
            --pretrained_model_path "checkpoints/22_02_localier_frozen_classifier/d=mtsd-m=stn-p=6-kl=None-betaP=1.0-lr=0.0001-lrloc=0.1.ckpt" &
done