#!/bin/sh

DATAPATH="/scratch/frwa/"
MODELS=("cnn" "cnn" "stn" "stn" "pstn")
PARAMS=(1 1 4 4 4)
TEST_SAMPELS=(1 1 1 1 10)
TRAIN_SAMPELS=(1 1 1 1 1)
CRITERION=("nll" "nll" "nll" "nll" "elbo")
MAXEPOCHS=10
#DATASETSIZES=(100 250 500 1000 2500 5000)
DATAAUGMENTATION=("f" "t" "f" "t" "f")
GPUS=(2 3 6 7 7)

#for DATASETSIZE in {0..5}
#do
    for ATTR in 15
    do
        for MODEL in {0..4}
        do
            echo ${MODELS[$MODEL]}
            echo $ATTR
            echo ${PARAMS[$MODEL]}
            echo ${TEST_SAMPELS[$MODEL]}
            echo ${TRAIN_SAMPELS[$MODEL]}
            echo ${CRITERION[$MODEL]}
            OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPUS[$model]} python train.py --dataroot $DATAPATH \
                            --dataset "celeba" \
                            --batch_size 256 \
                            --num_classes 2 \
                            --num_threads 2 \
                            --epochs $MAXEPOCHS \
                            --step_size 5 \
                            --seed 123 \
			    --val_check_interval 1 \
                            --model ${MODELS[$MODEL]} \
                            --num_param ${PARAMS[$MODEL]} \
                            --test_samples ${TEST_SAMPELS[$MODEL]} \
                            --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                            --criterion ${CRITERION[$MODEL]} \
                            --lr 0.1 \
			    --basenet 'resnet34' \
                            --lr_loc 0.01 \
                            --weightDecay 0 \
                            --num_param ${PARAMS[$MODEL]} \
                            --target_attr $ATTR \
                            --trainval_split True \
                            --save_results True \
                            --savepath "celeba_experiment_data_augmentation" &
        done
    done
#done
