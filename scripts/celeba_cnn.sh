#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(10)
TRAIN_SAMPELS=(10)
CRITERION=("nll" "nll" "elbo")


for MODEL in {0..0}
do
        echo ${MODELS[$MODEL]}
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=0 python train.py --dataroot $DATAPATH \
                        --dataset "celeba" \
                        --batch_size 256 \
                        --num_classes 2 \
                        --num_threads 1 \
                        --epochs 10 \
                        --step_size 2 \
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
                        --crop_size 64 \
                        --num_param ${PARAMS[$MODEL]} \
                        --basenet "simple" \
                        --target_attr 2 \
                        --trainval_split True \
                        --annealing "${ANNEALING[$ANNEAL]}" \
                        --save_results True \
                        --results_folder "24_03_celeba_cnn"  
done
