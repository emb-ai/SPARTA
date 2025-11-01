#!/bin/bash

run_attack() {
    local exp_id=$1
    local dataset=$2
    echo "Evaluating $exp_id on $dataset"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python filter_attacks.py -e experiments/EMNLP-paper/gbda/$dataset -i $exp_id -n $exp_id
    wait
}

run_attack gsva-7b-ft-res refcocog_test

run_attack lisa-7b-v1 refcocog_test

run_attack lisa-7b-v1-exp refcocog_test

run_attack lisa-13b-v1 refcocog_test

run_attack lisa-13b-v1-exp refcocog_test

run_attack lisa-13b-v0 refcocog_test

run_attack lisa-13b-v0-exp refcocog_test

run_attack lisa++ refcocog_test

run_attack gsva-7b-ft-res llmseg_test

run_attack lisa-7b-v1 llmseg_test

run_attack lisa-7b-v1-exp llmseg_test

run_attack lisa-13b-v1 llmseg_test

run_attack lisa-13b-v1-exp llmseg_test

run_attack lisa-13b-v0 llmseg_test

run_attack lisa-13b-v0-exp llmseg_test

run_attack lisa++ llmseg_test

run_attack gsva-7b-ft-res reason_test

run_attack lisa-7b-v1 reason_test

run_attack lisa-7b-v1-exp reason_test

run_attack lisa-13b-v1 reason_test

run_attack lisa-13b-v1-exp reason_test

run_attack lisa-13b-v0 reason_test

run_attack lisa-13b-v0-exp reason_test

run_attack lisa++ reason_test





