#!/bin/sh

DATAPATH="/scratch/s153847/data/"
MODELS=("inception" "stn" "pstn")
BRANCHES=(1 2 2)
PARAMS=(2 2 2)
TEST_SAMPELS=(0 0 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")

for MODEL in 0 1 2
do
    echo ${MODELS[$MODEL]}
    echo ${TRAIN_SAMPELS[$MODEL]}
    python train.py --dataroot $DATAPATH \
                    --dataset "mnist" \
                    --batch_size 64 \
                    --num_classes 100 \
                    --max_dataset_size 3 \
                    --num_threads 8 \
                    --epochs 1 \
                    --step_size 5 \
                    --seed 42 \
                    --model ${MODELS[$MODEL]} \
                    --num_param ${PARAMS[$MODEL]} \
                    --N ${BRANCHES[$MODEL]} \
                    --test_samples ${TEST_SAMPELS[$MODEL]} \
                    --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                    --criterion ${CRITERION[$MODEL]} \
                    --visualize True \
                    --lr 0.01 \
                    --sigma 0.1 \
                    --smallest_size 64 \
                    --crop_size 64 \
                    --run_test_freq 1 \
                    --num_param 2 \
                    --lr_loc -1 \
                    --basenet "simple" \
                    --alpha 0.1
done
