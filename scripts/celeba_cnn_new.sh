#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
OLD_RATIOS=(0. 0.15 0.23 0.5 0.75 1.)
GPUs=(0 6 7 4 5 0 0)

for r in {5..5}
do CUDA_VISIBLE_DEVICES=${GPUs[$r]} python train.py --dataroot $DATAPATH \
                    --dataset "celeba" \
                    --test_on "test" \
                    --batch_size 256 \
                    --num_classes 2 \
                    --num_threads 1 \
                    --epochs 10 \
                    --step_size 3 \
                    --seed 42 \
                    --model "cnn" \
                    --N 1 \
                    --criterion "nll" \
                    --save_results True \
                    --optimizer 'adam' \
                    --lr 0.001 \
                    --crop_size 64 \
                    --basenet "simple" \
                    --target_attr 2 \
                    --trainval_split True \
                    --save_results True \
                    --val_check_interval 10 \
                    --results_folder "20_04_celeba_cnn_upweigh_new_optim"  \
                    --upsample_attractive_oldies \
                    --desired_rate ${OLD_RATIOS[$r]} 
done
