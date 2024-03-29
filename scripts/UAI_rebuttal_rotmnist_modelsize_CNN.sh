#!/bin/sh

DATAPATH="data"
CLASSIFIER_MODELS=('nn1_classifier' 'nn2_classifier' 'nn3_classifier' 'nn4_classifier' 'nn5_classifier')
GPUs=(1 2 3 4 5)

for cm in {0..4}
    do
    for FOLD in {0..4}
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
                        --seed ${FOLD} \
                        --data_augmentation None \
                        --model 'cnn' \
                        --criterion 'nll' \
                        --lr 0.001 \
                        --lr_loc 1. \
                        --digits 1 \
                        --optimizer 'adam' \
                        --trainval_split True \
                        --save_results True \
                        --theta_path 'theta_stats' \
                        --download True \
                        --val_check_interval 100 \
                        --results_folder "06_06_rebuttal_layer_folds_seeds" \
                        --test_on "test" \
                        --modeltype_classifier ${CLASSIFIER_MODELS[$cm]} \
                        --fold ${FOLD} &
    done
done
