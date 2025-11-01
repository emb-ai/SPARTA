#!/bin/bash

run_attack() {
    local attacker=$1
    local segmod=$2
    local exp_id=$3
    local max_dataset_len=200
    local path="experiments/EMNLP-paper/ablations/meta-tokenize/${segmod}"
    echo "Running attack with attacker=$attacker and segmod=$segmod"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=25905 attack_main.py attacker=$attacker dataset=reason_test segmod@_global_=$segmod -o $path -e $exp_id max_dataset_len=$max_dataset_len
    wait
}

run_attack sonar_old lisa-7b-v1-exp rtext-base

run_attack sonar_dummy lisa-7b-v1-exp rtext-no-onehot

run_attack sonar_old lisa-13b-v1 rtext-base

run_attack sonar_dummy lisa-13b-v1 rtext-no-onehot

run_attack sonar_old lisa-7b-v1-exp rtext-base

run_attack sonar_dummy lisa-7b-v1-exp rtext-no-onehot

run_attack sonar_old lisa-13b-v1 rtext-base

run_attack sonar_dummy lisa-13b-v1 rtext-no-onehot