#!/bin/sh

DATAPATH="/scratch/s153847/"
MODELS=("pstn" "pstn" "pstn" "pstn")
ANNEALING=("no_annealing" "no_kl" "reduce_kl" "increase_kl")
SIGMAS=(0.1 0.3 0.3 1.2)

for SIGMA in {0..3}
do
    for MODEL in {0..3}
    do
        echo "model ${MODELS[$MODEL]}"
        echo "sigma ${SIGMAS[$SIGMA]}"
        echo "annealing ${ANNEALING[$MODEL]}"
        CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python train2.py --dataroot /scratch/s153847/ \
                        --model pstn \
                        --basenet simple \
                        --dataset mnist_easy \
                        --digits 2 \
                        --N 2 \
                        --train_samples 2 \
                        --num_threads 1 \
                        --test_samples 10 \
                        --batch_size 256 \
                        --num_classes 100 \
                        --step_size 3 \
                        --smallest_size 64 \
                        --crop_size 64 \
                        --lr_loc 1e-02 \
                        --seed 42 \
                        --lr 0.1 \
                        --criterion elbo \
                        --trainval_split True \
                        --annealing "${ANNEALING[$MODEL]}" \
                        --sigma "${SIGMAS[$SIGMA]}"
    done
done
