#!/bin/sh

DATAPATH="/scratch/s153847/"
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 6 6)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 2)
CRITERION=("nll" "nll" "elbo")
MAXEPOCHS=10
DATASETSIZE=1000

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
        CUDA_VISIBLE_DEVICES=3 python train.py --dataroot $DATAPATH \
                        --dataset "celeba" \
                        --batch_size 256 \
                        --num_classes 2 \
                        --num_threads 4 \
                        --epochs $MAXEPOCHS \
                        --step_size 9999 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --num_param ${PARAMS[$MODEL]} \
                        --max_dataset_size $DATASETSIZE \
                        --test_samples ${TEST_SAMPELS[$MODEL]} \
                        --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                        --criterion ${CRITERION[$MODEL]} \
                        --lr 0.01 \
                        --lr_loc 0.1 \
                        --smallest_size 64 \
                        --crop_size 64 \
                        --num_param ${PARAMS[$MODEL]} \
                        --target_attr $ATTR \
                        --trainval_split True \
                        --save_results True \
                        --savepath celeba_experiment1000
    done
done
