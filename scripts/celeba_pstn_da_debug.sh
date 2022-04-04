#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
W_s=(0. 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01)
GPUs=(2 2 3 3 4 4 6 6)


for w in {5..5} 
do
    CUDA_VISIBLE_DEVICES=7 python train.py --dataroot $DATAPATH \
                    --dataset "celeba" \
                    --test_on "val" \
                    --batch_size 256 \
                    --num_classes 2 \
                    --num_threads 1 \
                    --epochs 10 \
                    --step_size 3 \
                    --seed 42 \
                    --model "pstn" \
                    --num_param 4 \
                    --N 1 \
                    --test_samples 10 \
                    --train_samples 10 \
                    --criterion "elbo" \
                    --save_results True \
                    --optimizer 'adam' \
                    --lr 0.001 \
                    --lr_loc 0.1 \
                    --crop_size 64 \
                    --num_param 4\
                    --basenet "simple" \
                    --target_attr 2 \
                    --trainval_split True \
                    --annealing "weight_kl" \
                    --kl_weight ${W_s[$w]} \
                    --save_results True \
                    --results_folder '11_04_celeba_pstn_da_debug' \
                    --identity_mean \
                    --upsample_oldies \
                    --desired_rate 0.5 \
                    --var_init -1 \
                    --aug_training_only
done

