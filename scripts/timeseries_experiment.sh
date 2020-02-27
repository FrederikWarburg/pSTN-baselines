#!/bin/sh

DATAPATH="../projects/data_augmentation/time_series/UCR_TS_Archive_2015/"

MODELS=("cnn" "stn" "pstn")
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
DATASETS=("FaceAll" "wafer" "uWaveGestureLibrary_X" "Two_Patterns"
 "StarLightCurves" "PhalangesOutlinesCorrect" "FordA")
NR_CLASSES=(14 2 8 14 4 3 2 2)
PRIORS=(0.1 0.1 0.1 0.6 0.2 0.1 0.1)

for DATASET in {0..0}
do
    echo ${DATASETS[$DATASET]}
    for MODEL in {2..2}
    do
        echo ${MODELS[$MODEL]}
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=0 python train.py --dataroot $DATAPATH \
                        --dataset ${DATASETS[$DATASET]} \
                        --batch_size 16 \
                        --num_classes ${NR_CLASSES[$DATASET]}  \
                        --num_threads 1 \
                        --epochs 200 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --num_param 0 \
                        --N 1 \
                        --test_samples ${TEST_SAMPELS[$MODEL]} \
                        --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                        --criterion ${CRITERION[$MODEL]} \
                        --save_results True \
                        --lr 0.001 \
                        --sigma_p ${PRIORS[$DATASET]} \
                        --run_test_freq 100 \
                        --trainval_split True \
                        --save_results True \
                        --savepath "test" \
                        --optimizer "adam" \
                        --weightDecay 0 \
                        --transformer_type "diffeomorphic" \
                        --step_size 200

    done
done
