#!/bin/sh

SUBSETS=(10 30 100 1000 3000 10000)

for SUBSET in {0..5}
do
    echo ${SUBSETS[$SUBSET]}
    for FOLD in {0..4} # only do 2 folds for the grid search to limit computation time
    do
        CUDA_VISIBLE_DEVICES=4 python train.py --dataroot 'data' \
                            --dataset "MNIST" \
                            --subset ${SUBSETS[$SUBSET]} \
                            --fold ${FOLD} \
                            --batch_size 8 \
                            --num_classes 10  \
                            --num_threads 1 \
                            --epochs 600 \
                            --seed 42 \
                            --model "stn" \
                            --num_param 4 \
                            --N 1 \
                            --criterion  "nll" \
                            --save_results True \
                            --lr 0.0001 \
                            --lr_loc 0.1 \
                            --num_param 4 \
                            --trainval_split True \
                            --save_results True \
                            --optimizer "adam" \
                            --weightDecay 0.01 \
                            --transformer_type "affine" \
                            --step_size 600 \
                            --val_check_interval 600 \
                            --results_folder "19_02_MNIST_affine_small_LR_STN" \
                            --test_on "val" 
    done
done
