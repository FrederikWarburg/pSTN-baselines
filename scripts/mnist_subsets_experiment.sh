#!/bin/sh

DATAPATH="../projects/data_augmentation/time_series/UCR_TS_Archive_2015/"

MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
SUBSETS=(10 30 100 1000 3000 10000)

for SUBSETS in {0..5}
do
    echo ${SUBSETS[$SUBSET]}
    for MODEL in {0..0}
    do
        echo ${MODELS[$MODEL]}
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=0 python train.py --dataroot $DATAPATH \
                        --dataset "MNIST" \
                        --subset ${SUBSETS[$SUBSET]} \
                        --batch_size 8 \
                        --num_classes 10  \
                        --num_threads 1 \
                        --epochs 200 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --num_param ${PARAMS[$MODEL]} \
                        --N 1 \
                        --test_samples ${TEST_SAMPELS[$MODEL]} \
                        --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                        --criterion ${CRITERION[$MODEL]} \
                        --save_results True \
                        --lr 0.001 \
                        --sigma_p 0.05 \
                        --run_test_freq 100 \
                        --num_param ${PARAMS[$MODEL]} \
                        --trainval_split True \
                        --save_results True \
                        --savepath "test" \
                        --optimizer "adam" \
                        --weightDecay 0.01 \
                        --transformer_type "affine"
    done
done
