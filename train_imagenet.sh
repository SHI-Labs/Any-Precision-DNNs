#!/bin/bash

{
    curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
    # Change it when necessary
    source activate pt
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    train_id="exp_imagenet_resnet50q_1_0.3_lr"
    train_stamp="$(date +"%Y%m%d_%H%M%S")"
    train_id=${train_id}_${train_stamp}

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    # For individual models, change
    # lr=0.1, lr_decay=30,60,85,95,105, epochs=120, and weight-decay=0 or 1e-4.
    python -u train_imagenet.py \
        --model resnet50q \
        --dataset imagenet \
        --lr 0.5 \
        --lr_decay "45,60,70" \
        --epochs 80 \
        --batch-size 256 \
        --optimizer sgd \
        --weight-decay 1e-5 \
        --results-dir $result_dir \
        --bit_width_list "1,2,4,8,32"
} && exit
