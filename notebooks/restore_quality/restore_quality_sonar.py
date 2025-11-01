import os, sys
import json
import csv
import logging
import click
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed import init_process_group
import cv2
import numpy as np

import sonar
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

logging.basicConfig(level=logging.INFO)

JSON_DIR = "/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/ref-seg-text-break/dataset/reason_seg/ReasonSeg/test"
OUTPUT_CSV_TEMPLATE = "./restored_texts_sonar.csv"


def process_text(text, encode_model, decode_model, detokenizer, device='cuda'):
    embedding = encode_model.predict([text], source_lang="eng_Latn")
    embedding = embedding.clone().requires_grad_(True)
    _, lprobs = decode_model.predict(embedding, target_lang="eng_Latn", max_seq_len=512)
    restored_text = detokenizer(lprobs.argmax(dim=0))
    return {
        "orig_text": text,
        "restored_text": restored_text,
    }

def extract_texts_from_json(json_dir):
    texts = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                texts.extend([data['text'][0]])
    return texts

def save_results_to_csv(results, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["orig_text", "restored_text"]
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
@click.option('-c', '--cv2_threads', type=int, default=-1, help='Number of threads for `cv2.setNumThreads`')
def cli(cv2_threads):
    if cv2_threads >= 0:
        cv2.setNumThreads(cv2_threads)

    if WORLD_SIZE > 1:
        device = ddp_setup()
    else:
        device = 'cuda'
    is_master = RANK == 0
    
    texts = extract_texts_from_json(JSON_DIR)[:200]
    logging.info(f"Total number of texts: {len(texts)}")
    
    indexes = np.array_split(texts, WORLD_SIZE)[LOCAL_RANK]
    if is_master:
        indexes = tqdm(indexes)

    output_csv = OUTPUT_CSV_TEMPLATE

    if os.path.exists(output_csv):
        raise FileExistsError(f"The file '{output_csv}' already exists. Please choose a different name or delete the existing file.")
    logging.info("File does not exist. Proceeding...")
    
    sonar_t2v = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
    sonar_v2t = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder")
    sonar_decoder = sonar_t2v.tokenizer.create_decoder()
    
    results = []
    for text in indexes:
        result = process_text(text, sonar_t2v, sonar_v2t, sonar_decoder, device=device)
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