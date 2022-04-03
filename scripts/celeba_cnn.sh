#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
CUDA_VISIBLE_DEVICES=4 

python train.py --dataroot $DATAPATH \
                --dataset "celeba" \
                --batch_size 256 \
                --num_classes 2 \
                --num_threads 1 \
                --epochs 10 \
                --step_size 2 \
                --seed 42 \
                --model "cnn" \
                --N 1 \
                --criterion "nll" \
                --save_results True \
                --lr 0.1 \
                --lr_loc 0.1 \
                --crop_size 64 \
                --basenet "simple" \
                --target_attr 2 \
                --trainval_split True \
                --save_results True \
                --results_folder "01_04_celeba_cnn_upweigh_exp"  
