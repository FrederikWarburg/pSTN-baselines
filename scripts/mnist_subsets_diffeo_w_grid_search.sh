#!/bin/sh

SUBSETS=(10 30 100 1000 3000 10000)
W_s=(0.0 0.00001 0.0001 0.0005 0.001 0.002)

for SUBSET in {0..5}
do
    echo ${SUBSETS[$SUBSET]}
    for w in {0..5}
    do
        for FOLD in {0..4} # only do 2 folds for the grid search to limit computation time
        do
            CUDA_VISIBLE_DEVICES=2 python train.py --dataroot 'data' \
                                --dataset "MNIST" \
                                --subset ${SUBSETS[$SUBSET]} \
                                --fold ${FOLD} \
                                --batch_size 8 \
                                --num_classes 10  \
                                --num_threads 1 \
                                --epochs 600 \
                                --seed 42 \
                                --model "pstn" \
                                --N 1 \
                                --num_param 34 \
                                --test_samples 10 \
                                --train_samples 1 \
                                --criterion  "elbo" \
                                --save_results True \
                                --lr 0.001 \
                                --lr_loc 0.1 \
                                --beta_p 1. \
                                --trainval_split True \
                                --save_results True \
                                --optimizer "adam" \
                                --weightDecay 0.01 \
                                --transformer_type "diffeomorphic" \
                                --step_size 600 \
                                --val_check_interval 600 \
                                --results_folder "18_02_kl_weight_diffeo_new_init" \
                                --test_on "val" \
                                --annealing "weight_kl" \
                                --kl_weight ${W_s[$w]} \
                                --check_already_run True
        done
    done
done
