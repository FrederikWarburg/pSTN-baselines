#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
TEST_SAMPLES=(1 1 10)

for MODEL in {0..0}
do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=6 python train.py --dataroot $DATAPATH \
                    --dataset "fashion_mnist" \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 100 \
                    --step_size 50 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --modeltype 'large_loc' \
                    --num_param 2 \
                    --N 1 \
                    --test_samples 1 \
                    --train_samples 1 \
                    --criterion "nll" \
                    --lr 0.001 \
                    --lr_loc 0.1 \
                    --digits 1 \
                    --optimizer 'adam' \
                    --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --val_check_interval 100 \
                    --results_folder "31_02_fashion_mnist_reproductions_retry" \
                    --test_on "test" \
                    --normalize False 
                    done
