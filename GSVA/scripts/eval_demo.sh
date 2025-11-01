#!/bin/bash

# Arguments instruction:
# --val_dataset="grefcoco|unc|val", the format is Dataset name | version | split, e.g., "grefcoco|unc|val", "refcoco+|unc|testA" .
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/llava-v1_1-7b", path to a directory where LLaVA huggingface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L huggingface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a GSVA checkpoint.
# --precision="fp32", precision for evaluation.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# --eval_only, use this flag to perform evaluation.

export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --val_dataset="ReasonSeg|test" \
  --segmentation_model_path="../pretrained_weights/SAM/sam_vit_h_4b8939.pth" \
  --mllm_model_path="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
  --vision-tower=openai/clip-vit-large-patch14 \
  --dataset_dir=../dataset \
  --weight=../pretrained_weights/GSVA/gsva-llama2-13b-ft-res.bin \
  --precision=fp32 \
  --lora_r=64 \
  --eval_only 