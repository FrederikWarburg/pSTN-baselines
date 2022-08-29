#!/bin/sh
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
TEST_SAMPELS=(1 1 10)
TRAIN_SAMPELS=(1 1 1)
CRITERION=("nll" "nll" "elbo")
# OLD_SUBSETS=(30 100 1000 3000 10000)
SUBSETS=(312 1250 5000 20000 60000)
W_s=(0.001 0.0003 0.0001 0.00003 0.00001) # optimal W_s determined previously 
LR_s=(0.001 0.0001 0.001)

for SUBSET in {0..4}
do
    echo ${SUBSETS[$SUBSET]}
    for FOLD in {0..4}
    do
        for MODEL in {2..2}
        do
            echo ${MODELS[$MODEL]}
            echo ${PARAMS[$MODEL]}
            echo ${TEST_SAMPELS[$MODEL]}
            echo ${TRAIN_SAMPELS[$MODEL]}
            echo ${CRITERION[$MODEL]}
            CUDA_VISIBLE_DEVICES=4 python train.py --dataroot 'data' \
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
                            --lr  ${LR_s[$MODEL]}  \
                            --lr_loc 0.1 \
                            --num_param ${PARAMS[$MODEL]} \
                            --trainval_split True \
                            --save_results True \
                            --optimizer "adam" \
                            --weightDecay 0.01 \
                            --transformer_type "affine" \
                            --step_size 600 \
                            --val_check_interval 200 \
                            --test_on 'test' \
                            --beta_p 1. \
                            --var_init -5 \
                            --annealing "weight_kl" \
                            --kl_weight ${W_s[$SUBSET]} \
                            --results_folder "26_08_mnist_affine_thesis_runs" 
        done
    done
done
