#!/bin/sh

SUBSETS=(10 30 100 1000 3000 10000)
W_s=(0.001 0.001 0.0003 0.0001 0.00003 0.00001)

for SUBSET in {0..5}
do
    echo ${SUBSETS[$SUBSET]}
        for FOLD in {0..4} # only do 2 folds for the grid search to limit computation time
        do
            CUDA_VISIBLE_DEVICES=4ba python test.py --dataroot 'data' \
                                --dataset "MNIST" \
                                --subset ${SUBSETS[$SUBSET]} \
                                --fold ${FOLD} \
                                --batch_size 8 \
                                --num_classes 10  \
                                --num_threads 1 \
                                --epochs 600 \
                                --seed 42 \
                                --model "cnn" \
                                --beta_p 1 \
                                --num_param 4 \
                                --N 1 \
                                --test_samples 10 \
                                --train_samples 1 \
                                --criterion  "nll" \
                                --save_results True \
                                --num_param 4 \
                                --trainval_split True \
                                --save_results True \
                                --optimizer "adam" \
                                --weightDecay 0.01 \
                                --transformer_type "affine" \
                                --step_size 600 \
                                --val_check_interval 600 \
                                --results_folder "28_01_kl_weight_finer_grid_search" \
                                --test_on "test" \
                                --lr 0.001            

        done
done
