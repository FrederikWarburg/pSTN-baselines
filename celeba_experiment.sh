#!/bin/sh

DATAPATH="/scratch/s153847/"
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 6 6)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 2)
CRITERION=("nll" "nll" "elbo")

for ATTR in {0..40}
do
    for MODEL in {0..2}
    do
        echo ${MODELS[$MODEL]}
        echo $ATTR
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=4 python train2.py --dataroot $DATAPATH \
                        --dataset "celeba" \
                        --batch_size 256 \
                        --num_classes 2 \
                        --num_threads 1 \
                        --epochs 4 \
                        --step_size 1 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --num_param ${PARAMS[$MODEL]} \
                        --N 1 \
                        --test_samples ${TEST_SAMPELS[$MODEL]} \
                        --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                        --criterion ${CRITERION[$MODEL]} \
                        --save_results True \
                        --lr 0.1 \
                        --lr_loc 0.1 \
                        --sigma 0.1 \
                        --smallest_size 64 \
                        --crop_size 64 \
                        --run_test_freq 1 \
                        --num_param ${PARAMS[$MODEL]} \
                        --basenet "simple" \
                        --digits 1 \
                        --target_attr $ATTR \
                        --trainval_split True \
                        --save_results True \
                        --savepath celeba_experiment
    done
done
