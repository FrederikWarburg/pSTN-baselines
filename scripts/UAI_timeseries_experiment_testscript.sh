#!/bin/sh
MODELS=("cnn" "stn" "pstn")
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
DATASETS=("FaceAll" "wafer" "uWaveGestureLibrary_X" "Two_Patterns"
 "PhalangesOutlinesCorrect")
NR_CLASSES=(14 2 8 4 2)
WEIGHTS=(0.0001 1e-05 0.001 0.0 0.0001)


for DATASET in {0..4}
do
    echo ${DATASETS[$DATASET]}
    for FOLD in {0..4}
    do
        for MODEL in {2..2}
        do
        echo ${MODELS[$MODEL]}
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=0 python test.py --dataroot 'data' \
                        --dataset ${DATASETS[$DATASET]} \
                        --fold ${FOLD} \
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
                        --lr_loc 0.1 \
                        --run_test_freq 200 \
                        --trainval_split True \
                        --save_results True \
                        --optimizer "adam" \
                        --weightDecay 0 \
                        --transformer_type "diffeomorphic" \
                        --step_size 200 \
                        --val_check_interval 200 \
                        --var_init -20.0 \
                        --annealing "weight_kl" \
                        --kl_weight ${WEIGHTS[$DATASET]} \
                        --results_folder "30_03_UAI_repros_timeseries_noseed" \
                        --test_on "test" 
        done
    done
done
