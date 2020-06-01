#!/bin/sh

DATAPATH="data/"
MODELS=("cnn" "cnn" "stn" "stn" "pstn")
BRANCHES=(1 1 1 1 1)
PARAMS=(1 1 2 2 2)
TEST_SAMPELS=(1 1 1 1 10)
TRAIN_SAMPELS=(1 1 1 1 2)
CRITERION=("nll" "nll" "nll" "nll" "elbo")
GPUS=(0 0 0 0 0 0 0 1 2 3 4 1 2 3 4)
DATAAUGMENTATION=("None" "standard" "None" "standard" "None")

for MODEL in 4
do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    echo ${TRAIN_SAMPELS[$MODEL]}
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --dataroot $DATAPATH \
                    --dataset "mnistxkmnist" \
                    --batch_size 256 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 15 \
                    --step_size 10 \
                    --seed 42 \
                    --data_augmentation ${DATAAUGMENTATION[$MODEL]} \
                    --model ${MODELS[$MODEL]} \
                    --num_param ${PARAMS[$MODEL]} \
                    --N ${BRANCHES[$MODEL]} \
                    --test_samples ${TEST_SAMPELS[$MODEL]} \
                    --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.1 \
                    --digits 1 \
                    --dropout_rate 0 \
        	        --optimizer 'sgd' \
         	        --trainval_split True \
                    --prior_type 'mean_zero_gaussian'\
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --lr_loc 0.01 \
		            --save_training_theta True \
    		        --sigma_p 0.1 \
                    --learnable_prior False \
                    --val_check_interval 15
                    #--annealing  None \


done
