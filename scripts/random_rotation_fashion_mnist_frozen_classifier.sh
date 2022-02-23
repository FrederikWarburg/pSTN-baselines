#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "stn" "pstn")
CRITERION=("nll" "nll" "elbo")
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1.)
TEST_SAMPLES=(1 1 10)

for MODEL in {2..2}
do
    for w in {5..5}
    do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=4
     python train_parts.py --dataroot $DATAPATH \
                    --dataset "random_rotation_fashion_mnist" \
                    --normalize False \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 600 \
                    --step_size 200 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --num_param 1 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples 1 \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.001 \
                    --lr_loc 0.1 \
                    --digits 1 \
        	        --optimizer 'adam' \
         	        --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --val_check_interval 100 \
                    --results_folder "22_02_calibration_fashion_MNIST" \
                    --test_on "test" \
                    --annealing "weight_kl" \
                    --kl_weight ${W_s[$w]} \
                    --pretrained_model_path 'checkpoints/22_02_fashion_mnist_robustness/d=fashion_mnist-m=cnn-p=2-kl=None-betaP=1-lr=0.001-lrloc=None.ckpt' \
                    --modeltype 'large_loc' \
                    --init_large_variance True \
                    --var_init 1.
    done
done
