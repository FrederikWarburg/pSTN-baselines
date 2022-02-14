#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01)
TEST_SAMPLES=(1 1 10)

for MODEL in {1..1}
do
    for w in {2..2}
    do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python train.py --dataroot $DATAPATH \
                    --dataset "random_placement_mnist" \
                    --add_kmnist_noise True \
                    --crop_size  96 \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 100 \
                    --step_size 10 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --num_param 2 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples 1 \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.001 \
                    --lr_loc 0.01 \
                    --digits 1 \
        	        --optimizer 'adam' \
         	        --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --lr_loc 0.1 \
                    --val_check_interval 1000 \
                    --results_folder "08_02_kmnist_noise" \
                    --test_on "test" \
                    --annealing "weight_kl" \
                    --kl_weight ${W_s[$w]}
    done
done
