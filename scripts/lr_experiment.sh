#!/bin/sh

DATAPATH="/scratch/s153847/"
MODELS=("stn" "pstn")
TEST_SAMPELS=(1 10)
TRAIN_SAMPELS=(1 1)
CRITERION=("nll" "elbo")
LEARNING_RATES=(1e-1 1e-2 1e-3 1e-4 1e-5)

for LR in {2..4}
do
    for MODEL in {1..1}
    do
        echo ${MODELS[$MODEL]}
        echo $ATTR
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=1 python train2.py --dataroot $DATAPATH \
                        --dataset "celeba" \
                        --attr 15 \
                        --batch_size 256 \
                        --num_classes 2 \
                        --num_threads 1 \
                        --epochs 10 \
                        --step_size 9999 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --num_param 6 \
                        --N 1 \
                        --test_samples ${TEST_SAMPELS[$MODEL]} \
                        --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                        --criterion ${CRITERION[$MODEL]} \
                        --lr ${LEARNING_RATES[$LR]} \
                        --lr_loc 0.1 \
                        --sigma 0.1 \
                        --smallest_size 64 \
                        --crop_size 64 \
                        --run_test_freq 1 \
                        --basenet "simple" \
                        --digits 1 \
                        --trainval_split True \
                        --save_results True \
                        --results_folder 'lr_experiment'
    done
done

