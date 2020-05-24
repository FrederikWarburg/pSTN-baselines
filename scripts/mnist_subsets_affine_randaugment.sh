#!/bin/sh

MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
SUBSETS=(10 30 100 1000 3000)

for SUBSET in {0..4}
do
    echo ${SUBSETS[$SUBSET]}
    for FOLD in {0..2}
    do
        for MODEL in {0..0} # run pstn later when we have optimal sigma vals
        do
            for N in {0..4}
            do
                for M in {0..10}
                do
                    echo ${MODELS[$MODEL]}
                    echo ${PARAMS[$MODEL]}
                    echo ${TEST_SAMPELS[$MODEL]}
                    echo ${TRAIN_SAMPELS[$MODEL]}
                    echo ${CRITERION[$MODEL]}
                    CUDA_VISIBLE_DEVICES=0 python train.py --dataroot 'data' \
                                    --dataset "MNIST" \
                                    --subset ${SUBSETS[$SUBSET]} \
                                    --fold ${FOLD} \
                                    --batch_size 8 \
                                    --num_classes 10  \
                                    --num_threads 1 \
                                    --epochs 600 \
                                    --seed 42 \
                                    --model ${MODELS[$MODEL]} \
                                    --num_param ${PARAMS[$MODEL]} \
                                    --N 1 \
                                    --test_samples ${TEST_SAMPELS[$MODEL]} \
                                    --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                                    --criterion ${CRITERION[$MODEL]} \
                                    --save_results True \
                                    --lr 0.001 \
                                    --lr_loc 1 \
                                    --num_param ${PARAMS[$MODEL]} \
                                    --trainval_split True \
                                    --save_results True \
                                    --savepath "test" \
                                    --optimizer "adam" \
                                    --weightDecay 0.01 \
                                    --transformer_type "affine" \
                                    --step_size 600 \
                                    --val_check_interval 600 \
                                    --optimize_temperature False \
                                    --data_augmentation 'AffineRandAugment' \
                                    --rand_augment_N $N \
                                    --rand_augment_M $M \
                                    --test_on 'val' \
                                    --results_folder 'mnist_randaugment_results'
                done
            done
        done
    done
done
