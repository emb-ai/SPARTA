#!/bin/bash

run_attack() {
    local attacker=$1
    local segmod=$2
    local dataset=$3
    local max_dataset_len=$4
    local path="experiments/AAAI-RL-attack/tables/qwen/${dataset}"
    echo "Running attack with attacker=$attacker and segmod=$segmod"
    CUDA_VISIBLE_DEVICES=0,1,7 torchrun --nproc_per_node=3 --master_port=25903 attack_main.py attacker=$attacker dataset=$dataset segmod@_global_=$segmod -o $path -e $segmod max_dataset_len=$max_dataset_len 
    wait
}

declare -a configs=(
    "black_box_json,lisa-7b-v1,reason_test,300"
    "black_box_json,lisa-7b-v1-exp,reason_test,300"
    "black_box_json,lisa-13b-v1,reason_test,300"
    "black_box_json,lisa-13b-v1-exp,reason_test,300"
    "black_box_json,lisa++,reason_test,300"
    "black_box_json,gsva-13b-llama2-ft-res,reason_test,300"
)

call_count=150

for config in "${configs[@]}"; do
    IFS=',' read -r attacker segmod dataset min_dataset_len <<< "$config"
    run_attack "$attacker" "$segmod" "$dataset" "$min_dataset_len"
    ((call_count++))
done

echo "Total number of calls made: $call_count"