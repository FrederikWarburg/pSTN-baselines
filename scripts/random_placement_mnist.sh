#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")

for MODEL in {0..2}
do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=3 python train.py --dataroot $DATAPATH \
                    --dataset "random_placement_mnist" \
                    --crop_size  96 \
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
                    --results_folder "03_02_investigate_betas" \
                    --test_on "val" \
                    --annealing "weight_kl" \
                    --kl_weight 0.0003


done
