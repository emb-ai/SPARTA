import os
import glob
import click
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import random
import numpy as np
import re


def setup_deterministic(seed, deterministic=True):
    """
    set every seed
    """
    torch.set_printoptions(precision=16)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed, deterministic)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_deterministic_debug_mode(1)


def load_raw_attacks(exp_path):
    # ## Unite csv files (in case of multiple runs)
    def parse_samples_res():
        csv_files = exp_path.glob('results_sample=*.csv')
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        df = pd.concat(dataframes, ignore_index=True)
        mask = df.apply(lambda x: x.astype(str).str.contains('\r').any(), axis=1)
        df = df[~mask]
        return df
    return parse_samples_res()


def iou_duplicate_filter(df_raw):
    """Filter csv files: 
    1. $IoU_{orig} > IoU_{adv}$
    2. $IoU_{orig} > 0.1$
    3. drop duplicates"""
    
    df_copy = df_raw.copy()
    new_rows = []
    for sample_idx in df_copy['sample_idx'].unique():
        sample_df = df_copy[df_copy['sample_idx'] == sample_idx]
        orig_iou = sample_df[sample_df['iteration'] == 0]['iou'].values[0]
        if orig_iou < 0.1:
            continue
        sample_df['orig_iou'] = orig_iou
        sample_df = sample_df[sample_df['iteration'] != 0]
        sample_df = sample_df[sample_df['iou'] < sample_df['orig_iou']]
        sample_df = sample_df.drop_duplicates(subset='adv_text')
        new_rows.append(sample_df)

    filtered_df = pd.concat(new_rows, ignore_index=True)

    # print(f'{len(filtered_df)=}, {len(filtered_df['sample_idx'].unique())=}')
    return filtered_df


def nlu_filter(filtered_df, nlu_filtered_csv):
    """NLU filtration
    0. CONTRADICTION
    1. NEUTRAL
    2. ENTAILMENT"""

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")

    def process_row(row):
        def encode_input(t1, t2):
            input_text = f"[CLS] {t1} [SEP] {t2} [SEP]"
            return tokenizer.encode(input_text, padding=True, return_tensors='pt').to('cuda')
        
        encoded_inputs = [
            encode_input(row['orig_text'], row['adv_text']),
            encode_input(row['adv_text'], row['orig_text'])
        ]
        with torch.no_grad():
            logits = [model(encoded)['logits'] for encoded in encoded_inputs]
            labels = [torch.argmax(logit, dim=1).item() for logit in logits]
        return all(label != 0 for label in labels)

    sem_equiv_results = []

    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        setup_deterministic(seed=42, deterministic=True)
        result = process_row(row)
        sem_equiv_results.append(result)

    filtered_df['sem_equiv'] = sem_equiv_results

    len(filtered_df), len(filtered_df['sample_idx'].unique())
    
    filtered_df.to_csv(nlu_filtered_csv, index=False)
    return filtered_df

@torch.no_grad()
def nemotron_scoring(df, nemotron_csv):
    # ### Nemotron
    model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    MAIN_PROMPT = """
    You are an AI assistant who will help me to evaluate the quality of the following paraphrase on a scale from 1 to 5 based on its accuracy in preserving the original meaning for the task of referring segmentation. To mark a response, you should output a single integer between 1 and 5 (including 1, 5).

    All symbols of paraphrase and original text are written inside the <text></text> tag. If there is no text in the paraphrase and only symbols and punctuation marks, then rate it at 1.

    Output only the integer score from 1 to 5, no explanations.
    5 means the paraphrase fully retains the original sentence's meaning without losing important information or altering any key details
    1 means the paraphrase significantly distorts the meaning by inverting important parts of the original sentence

    Please note in your assessment that the reformulations must be meaningful, i.e. must not contain inappropriate symbols and punctuation marks

    Example 1:
    Original: <text>...</text>
    Paraphrase: <text>select the woman on the right in the image</text>
    Your mark: 1

    Example 2:
    Original: <text>select the man on the left in the image</text>
    Paraphrase: <text>select the woman on the right in the image</text>
    Your mark: 1

    Example 3:
    Original: <text>unopened flower bud</text>
    Paraphrase: <text>encumbered sacramento Created live encumberedsacral Based</text>
    Your mark: 1

    Example 4:
    Original: <text>In some circus performances, gymnasts perform aerial acrobatics using long pieces of fabric that hang from the ceiling. What in the picture could be used for this type of performance?</text>
    Paraphrase: <text>In some gymnastics performances, gals use long pieces of corbellage for shooting this type of performance. What can be located in the ceiling?</text>
    Your mark: 3

    Example 5:
    Original: <text>During a meal, people typically use utensils to bring food to their mouths. What tool in the picture can be used to fulfill this need?</text>
    Paraphrase: <text>During a meal, people often use utensils to bring food to their mouths. What tool in the picture can be used to fulfill this need?</text>
    Your mark: 5

    Example 6:
    Original: <text>select the left car in the image</text>
    Paraphrase: <text>select the vehicle shown on the left in the image</text>
    Your mark: 5

    Your Turn:
    Original: <text>{orig}</text>
    Paraphrase: <text>{adv}</text>
    """

    def extract_first_number(text):
        match = re.search(r'\d+', text)  # Ищет первое вхождение числа
        if match:
            return int(match.group())  # Преобразует найденное число в int
        return None
        
    df['SCORE_' + model_name] = None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        setup_deterministic(seed=42, deterministic=True)

        messages = [{"role": "user", "content": MAIN_PROMPT.format(
            orig=row['orig_text'],
            adv=row['adv_text']
        )}]

        tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, 
                                                        add_generation_prompt=True, 
                                                        return_tensors="pt", 
                                                        return_dict=True)
        response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),
                                            attention_mask=tokenized_message['attention_mask'].cuda(),  
                                            max_new_tokens=4096, 
                                            pad_token_id = tokenizer.eos_token_id)
        generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
        generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        try:
            score = int(extract_first_number(generated_text))
        except:
            print(generated_text)
            score = 1

        df.loc[idx, 'SCORE_' + model_name] = score

    df.to_csv(nemotron_csv, index=False)
    return df
    
    
def best_filter(df, best_flter_csv):
    """Choose the best sample with 5 if it exists, otherwise choose the best sample with 4..."""
    df_sorted = df.sort_values(
        by=['sample_idx', 'SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', 'iou_drop'], ascending=[True, False, False])
    final_df = df_sorted.groupby('sample_idx').first().reset_index()
    final_df.to_csv(best_flter_csv, index=False)
    return final_df

    

@click.command('Filter resulted adversarial texts and leave only best paraphrases')
@click.option('-e', '--exp-dir', default='experiments', type=Path)
@click.option('-i', '--exp-id', default=None, type=str)
@click.option('-n', '--name', default='attack', type=str)
@click.option('--stop-nemotron', type=bool, default=False)
def main(exp_dir, exp_id, name, stop_nemotron):

    folder_path = exp_dir / exp_id if exp_id is not None else exp_dir
    method_name = name
    
    df_raw = load_raw_attacks(folder_path)
    df_raw['method_name'] = method_name
    df = iou_duplicate_filter(df_raw)
    df['iou_drop'] = df['orig_iou'] - df['iou']
    df.drop(df[(df['iou_drop'] < 0)].index, inplace=True)
    
    # len(filtered_df), len(filtered_df['sample_idx'].unique())
    # nlu_filtered_csv = folder_path / 'fixed_filtered_df.csv'
    # if not nlu_filtered_csv.exists():
    #     df = nlu_filter(
    #         df,
    #         nlu_filtered_csv)
    # df = pd.read_csv(nlu_filtered_csv, index_col=False)
    # df = df[df.sem_equiv == True]
    
    if stop_nemotron:
        return
    
    nemotron_scores_csv = folder_path / 'fixed_nemotron_scores.csv'
    if not nemotron_scores_csv.exists():
        with torch.no_grad():
            df = nemotron_scoring(
                df,
                nemotron_scores_csv)
    df = pd.read_csv(nemotron_scores_csv, index_col=False)

    # df = df[df.sem_equiv == True]
    
    df.rename(columns={'iou': 'adv_iou'}, inplace=True)
    
    best_flter_csv = folder_path / 'fixed_best_filter.csv'
    if not best_flter_csv.exists():
        df = best_filter(df, best_flter_csv)
    df = pd.read_csv(best_flter_csv, index_col=False)
    
    # Calc success rates
    # df['iou_diff'] = (df['orig_iou'] - df['adv_iou'])
    # df['iou_diff_p'] = (df['orig_iou'] - df['adv_iou']) / df['orig_iou']
    # df['success'] = (df['iou_diff_p'] * 100 > 5) & (df['SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF']==5)

    # best_flter_csv = folder_path / 'sr.csv'
    # df.to_csv(best_flter_csv, index=False)

if __name__=='__main__':
    pd.options.mode.chained_assignment = None
    main()