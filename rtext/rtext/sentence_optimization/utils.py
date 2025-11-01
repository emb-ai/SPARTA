import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import jiwer
import json
import os

from sklearn.decomposition import PCA
from pathlib import Path
from transformers import AutoTokenizer
from typing import Tuple

from lisa.utils.utils import intersectionAndUnionGPU
from lisa.utils.utils import dict_to_cuda


IMAGE_TOKEN_INDEX = -200
LINE_BREAK_INDEX = 13

VAL_QUESTION_LIST = [
    {'start': "\n What is ", 'end': " in this image? Please output segmentation mask."},
    {'start': "\n ", 'end': " Please output segmentation mask."},
    {'start': "<im_end> What is ", 'end': " in this image? Please output segmentation mask."},
    {'start': "<im_end> ", 'end': " Please output segmentation mask."}
]


def get_texts(texts_dir):
    texts = []
    for filename in os.listdir(texts_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(texts_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "text" in data:
                    texts.extend(data["text"])
    return texts


def get_unique_tokens(tokenizer, texts_dir):
    unique_tokens = set()
    for text in get_texts(texts_dir):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        unique_tokens.update(tokens)
    return list(unique_tokens)


def get_pca(embeddings, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    return pca


def relative_error(m, m_approx, eps=1e-10):
    return np.mean(np.abs((m - m_approx) / (m + eps))) * 100


def get_transition(m1, m2):
    T = m2 @ m1.T @ torch.linalg.pinv(m1 @ m1.T)
    m2_approx = T @ m1 
    print("T, Approximation mean error:", f"{relative_error(m2.cpu().numpy(), m2_approx.cpu().numpy()):.2f} %")
    return T


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_sublist(big_list, small_list):
    """
    Finds the first occurrence of the small_list in the big_list.
    Returns the index of the first occurrence or -1 if not found.
    """
    # Convert the lists to strings
    separator = ','
    big_str = separator.join(map(str, big_list))
    small_str = separator.join(map(str, small_list))
    
    # Find the index in the string
    index = big_str.find(small_str)
    
    if index == -1:
        return -1
    
    # Count how many separators appear before the found index
    return big_str[:index].count(separator)


def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def cosine_sim(emb1, emb2):
    emb1_norm = F.normalize(emb1, p=2, dim=-1)
    emb2_norm = F.normalize(emb2, p=2, dim=-1)
    cosine_sim = (emb1_norm * emb2_norm).sum(dim=-1).mean()
    return cosine_sim


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def initialize_log_coeffs(input_ids, vocab_size, constant=15):
    log_coeffs = torch.zeros(len(input_ids), vocab_size) # number of tokens in sentence x voc_size
    indices = torch.arange(log_coeffs.size(0)).long()
    log_coeffs[indices, input_ids] = constant
    log_coeffs = log_coeffs.cuda()
    return log_coeffs.unsqueeze(0)


def calculate_iou(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    intersection, union, _ = intersectionAndUnionGPU(
        y_pred.contiguous().clone(), y_true.contiguous().int(), 2, ignore_index=255 
    )
    intersection = intersection.cpu().numpy()[1]
    union = union.cpu().numpy()[1]

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / (union + 1e-10)


def img_loss(inputs: torch.Tensor, targets: torch.Tensor):
    loss = F.binary_cross_entropy(inputs.sigmoid(), targets.to(inputs.device))
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scale=1000,  # 100000.0,
    eps=1e-6,
    ):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum()


def select_tokens_for_attack(
    text_with_system_prompt: str, 
    tokens: torch.Tensor, 
    tokenizer: AutoTokenizer, 
    crop_attack: bool = False,
    first_part: bool = True,
    max_word_length: int = 15,
    min_sublength: int = 7,
    stop_words: list = ['.', ',', '?', '!'],
    ) -> Tuple[int, int, torch.Tensor]:

    for system_prompt in VAL_QUESTION_LIST:
        start_sp, end_sp = system_prompt['start'], system_prompt['end']
        if start_sp in text_with_system_prompt and end_sp in text_with_system_prompt:
            start_adv_str = text_with_system_prompt.find(start_sp) + len(start_sp)
            end_adv_str = text_with_system_prompt.find(end_sp) 
            break
    else:
        raise ValueError("No valid system prompt found in the text.")
    
    if crop_attack:
        if len(text_with_system_prompt[start_adv_str:end_adv_str].split()) > max_word_length:
            for i, letter in enumerate(text_with_system_prompt[start_adv_str:end_adv_str]):
                if letter in stop_words and \
                    len(text_with_system_prompt[start_adv_str:start_adv_str+i].split()) > min_sublength and \
                    start_adv_str + i + 2 < end_adv_str:
                    if first_part:
                        end_adv_str = start_adv_str + i + 1
                    else:
                        start_adv_str += i + 2
                    break

    phrase_tokens = tokenizer.encode(text_with_system_prompt[start_adv_str:end_adv_str], add_special_tokens=False)
    
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i:i+len(phrase_tokens)] == phrase_tokens:
            return i, i + len(phrase_tokens)   

    print(text_with_system_prompt[start_adv_str:end_adv_str])
    raise ValueError("No valid start and end tokens found.")


def cut_original_context(
    sample: dict,
    model: torch.nn.Module,
    num_image_tokens: int = 256,
    start_adv_token: int = None,
    end_adv_token: int = None,
    device: str = 'cuda',
    ) -> Tuple[str, int]:

    sample = dict_to_cuda(sample, device=device)

    prefix_ids = sample['input_ids'][0, :start_adv_token].detach().clone()
    suffix_ids = sample['input_ids'][0, end_adv_token:].detach().clone()
    
    prefix_attn = sample['attention_masks'][0, :start_adv_token].detach().clone()
    suffix_attn = sample['attention_masks'][0, end_adv_token:].detach().clone()
    
    input_embeds = model.prepare_inputs_labels_for_multimodal(
        sample['input_ids'],
        sample['attention_masks'],
        None,
        sample['labels'],
        sample['images_clip'].half())[3].detach()

    prefix_embeds = input_embeds[0, :start_adv_token + model.insert_tokens].detach().clone()
    suffix_embeds = input_embeds[0, end_adv_token + model.insert_tokens:].detach().clone()
    
    return {
        'prefix_ids': prefix_ids,
        'suffix_ids': suffix_ids,
        'prefix_attn': prefix_attn,
        'suffix_attn': suffix_attn,
        'prefix_embeds': prefix_embeds,
        'suffix_embeds': suffix_embeds,
        'start_adv_token': start_adv_token,
        'end_adv_token': end_adv_token,
    }


def save_masked_image(
    mask: np.ma.MaskedArray, 
    image_path: str, 
    title: str, 
    folder: str = './',
    file_name: str = ''
    ):
    Path(folder).mkdir(parents=True, exist_ok=True)
    save_path = f'{folder}/{file_name}'
    
    plt.figure()
    plt.axis('off')
    plt.imshow(plt.imread(image_path), 'gray', interpolation='none')
    plt.imshow(np.ma.masked_where(mask == 0, mask), 'jet', interpolation='none', alpha=0.7)
    title = title.encode('unicode_escape').decode('utf-8')
    plt.title(f'"{title}"')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    
def save_texts(texts: dict, folder_path: str):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'texts.json', 'w') as f:
        json.dump(texts, f, indent=4)
    print(f"Results were saved to {folder_path}/texts.json")

