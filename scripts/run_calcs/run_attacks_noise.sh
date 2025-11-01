#!/bin/bash

run_attack() {
    local attacker=$1
    local segmod=$2
    local exp_id=$3
    local max_dataset_len=$4
    local path="experiments/EMNLP-paper/ablations/meta-tokenize/${segmod}"
    echo "Running attack with attacker=$attacker and segmod=$segmod"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=25905 attack_main.py attacker=$attacker dataset=reason_test segmod@_global_=$segmod -o $path -e $exp_id max_dataset_len=$max_dataset_len
    wait
}

run_attack sonar_old lisa-13b-v1-exp rtext-base 779

run_attack sonar_old lisa-13b-v1-exp rtext-base 779

run_attack sonar_old lisa-13b-v1-exp rtext-base 779