#!/bin/sh

DATAPATH="/scratch/s153847/data/"
MODELS=("cnn" "stn" "pstn")
BRANCHES=(1 2 2)
PARAMS=(1 2 2)
TEST_SAMPELS=(0 0 10)
TRAIN_SAMPELS=(1 1 2)
CRITERION=("nll" "nll" "elbo")

for MODEL in 0 1 2
do
    echo ${MODELS[$MODEL]}
    echo ${TRAIN_SAMPELS[$MODEL]}
    CUDA_VISIBLE_DEVICES=4 python train.py --dataroot $DATAPATH \
                    --dataset "mnist_easy" \
                    --batch_size 256 \
                    --num_classes 100 \
                    --num_threads 8 \
                    --epochs 10 \
                    --step_size 3 \
                    --seed 42 \
                    --model ${MODELS[$MODEL]} \
                    --num_param ${PARAMS[$MODEL]} \
                    --N ${BRANCHES[$MODEL]} \
                    --test_samples ${TEST_SAMPELS[$MODEL]} \
                    --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.1 \
                    --sigma 0.1 \
                    --smallest_size 64 \
                    --crop_size 64 \
                    --run_test_freq 1 \
                    --num_param 2 \
                    --lr_loc 0.01 \
                    --basenet "simple" \
		            --digits 2
done
