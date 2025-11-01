#!/bin/bash

run_attack_iters() {
    local path="experiments/EMNLP-paper/ablations"
    local exp_id="sonar-iters"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 attack_main.py attacker=sonar dataset=reason_test segmod@_global_=lisa-13b-v1-exp -o $path -e $exp_id max_dataset_len=200
    wait
}

run_attack_iters

run_attack_iters

run_attack_iters