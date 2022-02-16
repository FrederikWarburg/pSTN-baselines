#!/bin/sh

DATAPATH="/scratch/frwa/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
TEST_SAMPLES=(1 1 10)

for NUMPARAM in 4
do
for ATTR in 12
do
    for MODEL in 1
    do
        echo $ATTR ${MODELS[$MODEL]}
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python train.py --dataroot $DATAPATH \
                --dataset "celeba" \
                --target_attr $ATTR \
                --batch_size 64 \
                --num_classes 2 \
                --num_threads 4 \
                --epochs 20 \
                --step_size 10 \
                --seed 42 \
                --data_augmentation None \
                --model ${MODELS[$MODEL]} \
                --num_param $NUMPARAM \
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
                --results_folder "debug" \
                --test_on "val" \
                --weightDecay 0.01 \
                --annealing "weight_kl" \
                --kl_weight 0.0001 \
                --beta_p 1
    done
done
done