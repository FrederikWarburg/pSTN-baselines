#!/bin/sh

DATAPATH="/scratch/posc/" # on tethys
W_s=(0.000003 0.00001 0.00003)
GPUs=(7 7)
INITs=(-10 -5 -2.5)

for w in {0..0} 
do
    CUDA_VISIBLE_DEVICES=${GPUs[$w]}  python test.py --dataroot $DATAPATH \
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
                    --kl_weight 0.000003 \
                    --save_results True \
                    --results_folder '11_04_celeba_pstn_init_experiments' \
                    --identity_mean \
                    --var_init ${INITs[$w]} 
done

