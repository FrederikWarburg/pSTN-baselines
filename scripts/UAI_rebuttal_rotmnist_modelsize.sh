#!/bin/sh

DATAPATH="data"
MODELS=("stn" "pstn")
CRITERION=("nll" "elbo")
CLASSIFIER_MODELS=('nn1_classifier' 'nn2_classifier' 'nn3_classifier' 'nn4_classifier' 'nn5_classifier')
TEST_SAMPLES=(1 10)
GPUs=(2 3 4 5 6)

for MODEL in {1..1}
do
    for cm in {0..4}
    do
    echo $MODEL
    echo ${MODELS[$MODEL]}
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPUs[$cm]}
    
     python train.py --dataroot $DATAPATH \
                    --dataset "random_rotation_mnist" \
                    --normalize False \
                    --batch_size 64 \
                    --num_classes 10 \
                    --num_threads 2 \
                    --epochs 100 \
                    --step_size 10 \
                    --seed 42 \
                    --data_augmentation None \
                    --model ${MODELS[$MODEL]} \
                    --num_param 1 \
                    --N 1 \
                    --test_samples ${TEST_SAMPLES[$MODEL]} \
                    --train_samples 10 \
                    --criterion ${CRITERION[$MODEL]} \
                    --lr 0.001 \
                    --lr_loc 1. \
                    --digits 1 \
        	        --optimizer 'adam' \
         	        --trainval_split True \
                    --save_results True \
                    --theta_path 'theta_stats' \
                    --download True \
                    --val_check_interval 100 \
                    --results_folder "25_04_rebuttal_layer_exp" \
                    --var_init -3 \
                    --test_on "test" \
                    --annealing "weight_kl" \
                    --kl_weight 0.00003 \
                    --modeltype 'large_loc' \
                    --modeltype_classifier ${CLASSIFIER_MODELS[$cm]} &
    done
done
