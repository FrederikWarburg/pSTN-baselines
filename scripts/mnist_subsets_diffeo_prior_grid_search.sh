#!/bin/sh

SUBSETS=(10 30 100 1000 3000 10000)
PRIORS=(0.1 0.3 0.5 0.7 0.9)

for SUBSET in {0..5}
do
    echo ${SUBSETS[$SUBSET]}
    for PRIOR in {0..8}
    do
        for FOLD in {0..1} # only do 2 folds for the grid search to limit computation time
        do
            CUDA_VISIBLE_DEVICES=6 python train.py --dataroot 'data' \
                                --dataset "MNIST" \
                                --subset ${SUBSETS[$SUBSET]} \
                                --fold ${FOLD} \
                                --batch_size 8 \
                                --num_classes 10  \
                                --num_threads 1 \
                                --epochs 600 \
                                --seed 42 \
                                --model "pstn" \
                                --num_param 0 \
                                --N 1 \
                                --test_samples 10 \
                                --train_samples 1 \
                                --criterion  "elbo" \
                                --save_results True \
                                --lr 0.001 \
                                --lr_loc 1 \
                                --sigma_p ${PRIORS[$PRIOR]} \
                                --num_param 4 \
                                --trainval_split True \
                                --save_results True \
                                --optimizer "adam" \
                                --weightDecay 0.01 \
                                --transformer_type "affine" \
                                --step_size 600 \
                                --val_check_interval 600 \
                                --results_folder "grid_search_mnist_diffeo" \
                                --test_on "val"
        done
    done
done
