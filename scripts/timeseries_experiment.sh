#!/bin/sh

DATAPATH="../projects/data_augmentation/time_series/UCR_TS_Archive_2015/"

MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
DATASETS=("FaceAll" "wafer" "uWaveGestureLibrary_X" "FaceAll" "Two_Patterns"
 "StarLightCurves" "PhalangesOutlinesCorrect" "FordA")
NR_CLASSES=(14 2 8 14 4 3 2 2)

for DATASET in {0..}
do
    echo ${DATASETS[$DATASET]}
    for MODEL in {0..2}
    do
        echo ${MODELS[$MODEL]}
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=0 python train2.py --dataroot $DATAPATH \
                        --dataset ${DATASETS[$DATASET]} \
                        --batch_size 16 \
                        --num_classes ${NR_CLASSES[$DATASET]}  \
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
                        --sigma 0.6 \
                        --run_test_freq 100 \
                        --num_param ${PARAMS[$MODEL]} \
                        --basenet "timeseries" \
                        --trainval_split True \
                        --save_results True \
                        --savepath "test"
    done
done
