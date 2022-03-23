#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
LRs=(4.0e-05 6.0e-05 8.0e-05 1.2e-04 1.4e-04)

TEST_SAMPLES=(1 1 10)
TRAIN_SAMPLES=(1 1 10)

for MODEL in {0..0}
do
    for lr in {0..0}
    do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=7
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
                    --num_param 4 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples ${TRAIN_SAMPLES[$MODEL]} \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr ${LRs[$lr]} \
                    --digits 1 \
        	        --optimizer 'adam' \
         	        --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --val_check_interval 20 \
                    --results_folder "23_02_random_pm_fashion_mnist_robustness_redo" \
                    --test_on "test" \
                    --annealing "weight_kl" \
                    --kl_weight 0.00003 \
                    --pretrained_model_path 'checkpoints/22_02_fashion_mnist_robustness/d=fashion_mnist-m=cnn-p=2-kl=None-betaP=1-lr=0.001-lrloc=None.ckpt' \
                    --var_init -20 \
                    --modeltype 'large_loc' \
                    --reduce_samples 'min' \
                    --freeze_classifier
    done
done
