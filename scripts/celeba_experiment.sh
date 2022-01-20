#!/bin/sh

DATAPATH="/scratch/frwa/data"
MODELS=("pstn" "pstn" "cnn" "cnn" "stn" "stn" "pstn")
PARAMS=(4 4 1 1 4 4 4)
TEST_SAMPELS=(10 10 1 1 10)
TRAIN_SAMPELS=(2 2 1 1 1)
CRITERION=("elbo" "elbo" "nll" "nll" "nll" "nll" "elbo")
MAXEPOCHS=50
#DATASETSIZES=(100 250 500 1000 2500 5000)
DATAAUGMENTATION=("None" "None" "None" "standard" "None")
GPUS=(0 6 1 2 3 6 6 6)

#for DATASETSIZE in {0..5}
#do
    for ATTR in 15
    do
        for MODEL in 1
        do
            echo ${MODELS[$MODEL]}
            echo $ATTR
            echo ${PARAMS[$MODEL]}
            echo ${TEST_SAMPELS[$MODEL]}
            echo ${TRAIN_SAMPELS[$MODEL]}
            echo ${CRITERION[$MODEL]}
            OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPUS[$MODEL]} python train.py --dataroot $DATAPATH \
                            --dataset "celeba" \
                            --batch_size 256 \
                            --num_classes 2 \
                            --num_threads 2 \
                            --epochs $MAXEPOCHS \
                            --step_size 9999 \
                            --seed 123 \
			    --val_check_interval 1 \
                            --model ${MODELS[$MODEL]} \
                            --num_param ${PARAMS[$MODEL]} \
                            --test_samples ${TEST_SAMPELS[$MODEL]} \
                            --train_samples ${TRAIN_SAMPELS[$MODEL]} \
                            --criterion ${CRITERION[$MODEL]} \
                            --lr 1e-3 \
			    --basenet 'resnet34' \
			    --crop_size 64 \
                            --lr_loc 1 \
                            --weightDecay 0 \
			    --data_augmentation ${DATAAUGMENTATION[$MODEL]} \
                            --num_param ${PARAMS[$MODEL]} \
                            --target_attr $ATTR \
                            --trainval_split True \
                            --save_results True \
                            --results_folder "celeba_experiment_data_augmentation" &
        done
    done
#done
