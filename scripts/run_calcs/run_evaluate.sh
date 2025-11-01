#!/bin/bash

run_attack() {
    local exp_id=$1
    echo "Running attack with exp_id=$exp_id"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python filter_attacks.py -e experiments/AAAI-RL-attack/targeted/sam_mask -i $exp_id -n $exp_id
    wait
}

run_attack gumbel_target/lisa-7b-v1

run_attack sonar_rl_target/lisa-7b-v1

run_attack sonar_target/lisa-7b-v1