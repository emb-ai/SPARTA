import os, sys
import json
import csv
import logging
import argparse
import click
from vec2text.aliases import load_model_from_alias
from transformers import AutoTokenizer
from tqdm import tqdm
import vec2text
sys.path.append('/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/ref-seg-text-break')
from sentence_optimization.emb_attack import restore_embedding
import functools
import torch
import torch.distributed as dist
from torch.distributed import init_process_group
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

JSON_DIR = "/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/ref-seg-text-break/dataset/reason_seg/ReasonSeg/test"
INVERSION_MODEL_PATH = 'lisa_msmarco__msl32__100epoch'
CORRECTOR_MODEL_PATH = 'lisa_correct_msmarco__msl32__100epoch'
OUTPUT_CSV_TEMPLATE = "./restored_texts_ns={ns}_bsw={bsw}.csv"

lisa_tokenizer = AutoTokenizer.from_pretrained(
    "xinlai/LISA-13B-llama2-v1",
    use_fast=False,
    add_eos_token=True
)
lisa_tokenizer.pad_token = lisa_tokenizer.unk_token


def get_embedding(text, model, tokenizer=lisa_tokenizer, device='cuda'):
    inputs = tokenizer(text, return_tensors='pt', add_eos_token=True).to(device)
    return model.call_embedding_model(inputs['input_ids'], inputs['attention_mask'])

def process_text(text, ns, bsw, inversion, corrector, device='cuda'):
    embedding = get_embedding(text, inversion, device=device)
    restored_text = restore_embedding(embedding, corrector, ns=ns, sbw=bsw)
    return {
        "orig_text": text,
        "restored_text": restored_text,
        "num_steps": ns,
        "beam_width": bsw,
    }

def extract_texts_from_json(json_dir):
    texts = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                texts.extend(data.get("text", []))
    return texts

def save_results_to_csv(results, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["num_steps", "beam_width", "orig_text", "restored_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        
RANK = int(os.environ.get("RANK", '0'))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", '0'))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", '1'))
CUDA_VISIBLE_DEVICES = [int(d) for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
GPUS_SIZE = len(CUDA_VISIBLE_DEVICES)

def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if WORLD_SIZE > GPUS_SIZE:
        gpu_num = RANK % GPUS_SIZE
        raise RuntimeError('The number of processes is more than the number of GPUs')
    else:
        gpu_num = RANK
    torch.cuda.set_device(gpu_num)
    init_process_group(backend='nccl', rank=RANK, world_size=WORLD_SIZE)
    device = torch.device("cuda", gpu_num)
    return device


@click.command()
@click.option('--ns', type=int, default=None, help='Number of steps for restoration')
@click.option('--bsw', type=int, default=0, help='Beam width for restoration')
@click.option('-c', '--cv2_threads', type=int, default=-1, help='Number of threads for `cv2.setNumThreads`')
def cli(ns, bsw, cv2_threads):
    if cv2_threads >= 0:
        cv2.setNumThreads(cv2_threads)

    if WORLD_SIZE > 1:
        device = ddp_setup()
    else:
        device = 'cuda'
    is_master = RANK == 0
    
    texts = extract_texts_from_json(JSON_DIR)
    logging.info(f"Total number of texts: {len(texts)}")
    
    indexes = np.array_split(texts, WORLD_SIZE)[LOCAL_RANK][:16]
    if is_master:
        indexes = tqdm(indexes)

    output_csv = OUTPUT_CSV_TEMPLATE.format(ns=ns, bsw=bsw)

    if os.path.exists(output_csv):
        raise FileExistsError(f"The file '{output_csv}' already exists. Please choose a different name or delete the existing file.")
    logging.info("File does not exist. Proceeding...")
    
    inversion = load_model_from_alias(INVERSION_MODEL_PATH)
    corrector_model = load_model_from_alias(CORRECTOR_MODEL_PATH)
    corrector = vec2text.load_corrector(inversion, corrector_model)
    
    results = []
    for text in indexes:
        result = process_text(text, ns, bsw, inversion, corrector, device=device)
        results.append(result)
        
    if WORLD_SIZE > 1:
        gather_res = [None]*WORLD_SIZE

        dist.barrier()
        dist.all_gather_object(gather_res, results)
        
        results = []
        for r in gather_res:
            results.extend(r)

    if is_master:
        save_results_to_csv(results, output_csv)
        logging.info(f"CSV file saved as {output_csv}")


if __name__ == "__main__":
    cli()