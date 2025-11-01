import os
import glob
import json
import re
import torch

import click

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


BASE_IMAGE_DIR = "../../dataset"
MODEL_NAME = "Qwen/Qwen3-32B"

MAIN_PROMPT = """
You need to make a paraphrase of the original query.
It should not be much longer than original one. Also, it should be natural.
Here is the original query:
{orig}
Please provide a paraphrased query that meets the above conditions.
Return only modified query and don't add anything before and after it.
"""


def generate_with_llm(original_text: str, model, tokenizer):
    messages = [{"role": "user", "content": MAIN_PROMPT.format(
        orig=original_text
    )}]

    tokenized_message = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True,
        enable_thinking=False
    )
    response_token_ids = model.generate(
        tokenized_message['input_ids'].cuda(),
        attention_mask=tokenized_message['attention_mask'].cuda(),  
        max_new_tokens=4096, 
        pad_token_id = tokenizer.eos_token_id
    )
    generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return generated_text


def remove_think_tags(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('\n', '')
    return text


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["reason_test", "llmseg_test"], case_sensitive=False),
    required=True,
    help="Which dataset split to process (reason_test or llmseg_test).",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    default=None,
    help="Process only the first N samples (default: all).",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Path to the output JSON file. Defaults to qwen3_paraphrases_<dataset>.json",
)
def cli(dataset: str, num_samples: int | None, output: str | None):
    """Generate paraphrases for the selected dataset split using Qwen3-32B."""

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype='auto',
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = dataset.lower()

    if dataset == "reason_test":
        pattern = os.path.join(BASE_IMAGE_DIR, "reason_seg", "ReasonSeg", "test", "*.json")
        files = sorted(glob.glob(pattern))
        if num_samples is not None:
            files = files[:num_samples]

        original_texts = []
        for path in files:
            with open(path, "r") as f:
                try:
                    original_texts.append(json.load(f)["text"][0])
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue

    elif dataset == "llmseg_test":
        data_json_path = os.path.join(BASE_IMAGE_DIR, "llm-seg40k", "validation.json")
        if not os.path.exists(data_json_path):
            raise FileNotFoundError(
                f"Expected consolidated annotations at {data_json_path}. "
                "Please adjust BASE_IMAGE_DIR if your dataset is elsewhere."
            )

        with open(data_json_path, "r") as f:
            full_json = json.load(f)

        img_names = list(full_json.keys())
        if num_samples is not None:
            img_names = img_names[:num_samples]

        original_texts = [full_json[name]["qa_pairs"][0]["question"] for name in img_names]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


    if output is None:
        output = f"qwen3_paraphrases_{dataset}.json"

    if os.path.exists(output):
        with open(output, "r") as f:
            try:
                paraphrases: dict[str, str] = json.load(f)
            except json.JSONDecodeError:
                paraphrases = {}
    else:
        paraphrases = {}
    breakpoint()
    print(f"Generating paraphrases for {len(original_texts)} samples")

    for original_text in tqdm(original_texts, desc="Generating paraphrases"):
        if original_text in paraphrases:
            continue
        print('processing: \t', original_text)
        paraphrase = generate_with_llm(original_text, model, tokenizer)
        paraphrase = remove_think_tags(paraphrase)
        paraphrases[original_text] = paraphrase

        with open(output, "w") as f:
            json.dump(paraphrases, f, indent=4)


if __name__ == "__main__":
    cli()