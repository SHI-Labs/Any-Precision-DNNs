#!/bin/bash

{
    curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
    # Change it when necessary
    source activate pt

    train_id="exp_cifar10_resnet20q_124832_recursive"
    train_stamp="$(date +"%Y%m%d_%H%M%S")"
    train_id=${train_id}_${train_stamp}

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python -u train.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_decay "100,150,180" \
        --epochs 200 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "1,2,4,8,32"
} && exit
