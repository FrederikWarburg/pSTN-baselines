#!/bin/sh
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
SUBSETS=(10 30 100 1000 3000 10000)

for SUBSET in {0..0}
do
    echo ${SUBSETS[$SUBSET]}
    for FOLD in {0..0}
    do
        for MODEL in {2..2}
        do
            echo ${MODELS[$MODEL]}
            echo ${PARAMS[$MODEL]}
            echo ${TEST_SAMPELS[$MODEL]}
            echo ${TRAIN_SAMPELS[$MODEL]}
            echo ${CRITERION[$MODEL]}
            CUDA_VISIBLE_DEVICES=1 python train.py --dataroot 'data' \
                            --dataset "MNIST" \
                            --subset ${SUBSETS[$SUBSET]} \
                            --fold ${FOLD} \
                            --batch_size 8 \
                            --num_classes 10  \
                            --num_threads 1 \
                            --epochs 600 \
                            --seed 42 \
                            --model ${MODELS[$MODEL]} \
                            --num_param ${PARAMS[$MODEL]} \
                            --N 1 \
                            --test_samples ${TEST_SAMPELS[$MODEL]} \
                            --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                            --criterion ${CRITERION[$MODEL]} \
                            --lr 0.001 \
                            --beta_p 0.001 \
                            --num_param ${PARAMS[$MODEL]} \
                            --trainval_split True \
                            --save_results True \
                            --optimizer "adam" \
                            --weightDecay 0.01 \
                            --transformer_type "affine" \
                            --step_size 600 \
                            --val_check_interval 1 \
                            --test_on 'test' \
                            --results_folder "debugging"
        done
    done
done
