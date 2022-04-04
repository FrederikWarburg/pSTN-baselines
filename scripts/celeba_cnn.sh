#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
OLD_RATIOS=(0. 0.15 0.23 0.5 0.75 1.)
GPUs=(1 6 6 4 5 1)

for r in {1..1}
do CUDA_VISIBLE_DEVICES=${GPUs[$r]} python test.py --dataroot $DATAPATH \
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
                    --val_check_interval 10 \
                    --results_folder "04_04_celeba_cnn_upweigh_exp"  \
                    --upsample_attractive_oldies \
                    --desired_rate ${OLD_RATIOS[$r]}
done
