#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
MODELS=("cnn" "stn" "pstn")
PARAMS=(1 4 4)
CRITERION=("nll" "nll" "elbo")
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01)
GPUs=(0 1 2 3 4 5 6 7)

for MODEL in {2..2}
do
    for w in {0..0} 
    do
        echo ${MODELS[$MODEL]}
        echo ${PARAMS[$MODEL]}
        echo ${TEST_SAMPELS[$MODEL]}
        echo ${TRAIN_SAMPELS[$MODEL]}
        echo ${CRITERION[$MODEL]}
        CUDA_VISIBLE_DEVICES=${GPUs[$w]} python train.py --dataroot $DATAPATH \
                        --dataset "celeba" \
                        --test_on "val" \
                        --batch_size 256 \
                        --num_classes 2 \
                        --num_threads 1 \
                        --epochs 10 \
                        --step_size 3 \
                        --seed 42 \
                        --model ${MODELS[$MODEL]} \
                        --num_param ${PARAMS[$MODEL]} \
                        --N 1 \
                        --test_samples 10 \
                        --train_samples 10 \
                        --criterion ${CRITERION[$MODEL]} \
                        --save_results True \
                        --lr 0.1 \
                        --lr_loc 0.1 \
                        --crop_size 64 \
                        --num_param ${PARAMS[$MODEL]} \
                        --basenet "simple" \
                        --target_attr 2 \
                        --trainval_split True \
                        --annealing "weight_kl" \
                        --kl_weight ${W_s[$w]} \
                        --save_results True \
                        --results_folder "debug" 
    done
done
