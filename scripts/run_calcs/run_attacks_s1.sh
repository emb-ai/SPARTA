#!/bin/bash

run_attack() {
    local attacker=$1
    local segmod=$2
    local dataset=$3
    local max_dataset_len=$4
    local path="experiments/AAAI-RL-attack/tables/${attacker}/${dataset}"
    echo "Running attack with attacker=$attacker and segmod=$segmod"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=25901 attack_main.py attacker=$attacker dataset=$dataset segmod@_global_=$segmod -o $path -e $segmod max_dataset_len=$max_dataset_len
    wait
}

declare -a configs=(
    "sonar_rl,lisa-7b-v1,reason_test,300"
)

call_count=150

for config in "${configs[@]}"; do
    IFS=',' read -r attacker segmod dataset min_dataset_len <<< "$config"
    run_attack "$attacker" "$segmod" "$dataset" "$min_dataset_len"
    ((call_count++))
done

echo "Total number of calls made: $call_count"