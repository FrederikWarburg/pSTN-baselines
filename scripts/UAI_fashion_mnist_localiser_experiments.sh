#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
TEST_SAMPLES=(1 1 10)
for MODEL in {2..2}
do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=6 python train_localizer.py --dataroot $DATAPATH \
                    --dataset "random_placement_fashion_mnist" \
                    --freeze_classifier \
                    --crop_size  96 \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 600 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --modeltype 'large_loc' \
                    --num_param 4 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples 10 \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.001 \
                    --lr_loc 0.1 \
                    --digits 1 \
                    --optimizer 'adam' \
                    --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --annealing "weight_kl" \
                    --kl_weight 0.00003 \
                    --download True \
                    --val_check_interval 100 \
                    --results_folder "01_04_fashion_mnist_reproductions_retry" \
                    --pretrained_model_path 'checkpoints/31_02_fashion_mnist_reproductions_retry/d=fashion_mnist-m=cnn-p=2-kl=None-betaP=1-lr=0.001-lrloc=None_large_loc.ckpt' \
                    --test_on "test" \
                    --normalize False \
                    --step_size 200f
done
