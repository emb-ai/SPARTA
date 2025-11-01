import torch
import hydra
import pandas as pd
from hydra.utils import instantiate, get_class
from pathlib import Path
import click
import cv2
import os
from glob import glob

from lisa.model.llava import conversation as conversation_lib
from lisa.utils.dataset import collate_fn
from rtext.sentence_optimization.utils import set_seeds
from rtext.sentence_optimization.model_setup import setup_model, setup_tokenizer
from rtext.sentence_optimization.inference import inference
from rtext.sentence_optimization.train import AttackerBase

from metrics import MetricsCalculator
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
import functools
import typing as t

import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_process_group
from omegaconf import OmegaConf, open_dict

import transformers
from transformers import logging
import random

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
    transformers.set_seed(seed) #, deterministic)
    # torch.use_deterministic_algorithms(deterministic)
    # torch.set_deterministic_debug_mode(1)


# Set logging level to ERROR or CRITICAL to suppress most logs
logging.set_verbosity_error()


def merge_results(outdir, exp_id):
    folder_path = os.path.join(outdir, str(exp_id))
    dataframes = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('results_sample')]
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    df = pd.concat(dataframes, ignore_index=True)
    df.to_csv(f'{outdir}/{exp_id}/results_all.csv', index=False)


def chat(name: str, model, tokenizer, cfg, logger):
    cfg.dataset['base_image_dir'] = 'LISA/imgs'
    cfg.dataset['val_dataset'] = f'{name}.jpg'
    dataset = get_class(cfg.dataset_class)(
        **cfg.dataset,
        vision_tower=cfg.vision_tower,
        tokenizer=tokenizer,
    )
    sample = get_sample(dataset, tokenizer)
    output = inference(sample, model)
    logger.vis(
        sample_idx=0,
        adv_text="",
        image=plt.imread(sample['image_paths'][0]),
        pred_mask=(output['pred_masks'][0][0].cpu().detach().numpy() > 0).astype(np.uint8),
        iteration=0,
        adv_indx=0,
        iou=0.0,
        color='Oranges',
        line_color='orange',
    )


def load_config(overrides, config_path='configs', config_name='config'):
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    
    return cfg


def get_sample(dataset, tokenizer, indx=0, req_indx=0):
    sample = collate_fn(
        [dataset[indx]], 
        tokenizer)
    
    sample['input_ids'] = sample['input_ids'][req_indx].unsqueeze(0)
    sample['labels'] = sample['labels'][req_indx].unsqueeze(0)
    sample['attention_masks'] = sample['attention_masks'][req_indx].unsqueeze(0)
    sample['conversation_list'] = [sample['conversation_list'][req_indx]]
    if sample['masks_list'][req_indx].shape[0] > 1:
        sample['masks_list'] = [sample['masks_list'][req_indx][req_indx].unsqueeze(0)]
    else:
        sample['masks_list'] = [sample['masks_list'][req_indx]]
    sample['offset'] = torch.tensor([0, len(sample['conversation_list'])])

    return sample


def get_dist_time():
    now = datetime.now()
    curr_time = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
    if WORLD_SIZE > 1:
        gather_res = [None]*WORLD_SIZE

        dist.barrier()
        dist.all_gather_object(gather_res, curr_time)
        
        return gather_res[0]
    return curr_time

class AttackLogger:
    def __init__(self, cfg, outdir, exp_id):
        self.outdir: Path = Path(outdir) / exp_id
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.cfg = cfg
        OmegaConf.save(self.cfg, self.outdir / 'config.yaml')
    
    def get_outdir(self, iter_num: int):
        outdir = self.outdir / f'iter={iter_num}'
        outdir.mkdir(exist_ok=True, parents=True)
        return outdir
    
    def to_csv(self, results: t.List[t.Dict], outfname='results.csv', text_columns=['adv_text', 'orig_text']):
        df = pd.DataFrame(results)
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.replace('\n', '\\n'))
        df.to_csv(self.outdir / outfname, index=False)
        
    # def print_success(self, sample_idx, outfname='success_idxs.txt'):
    #     with open(self.outdir/outfname, 'a') as f:
    #         f.write(f'{sample_idx} ')
    
    def print_success(self, sample_idx, orig_text, image_path):
        sample = {'sample_idx': sample_idx,
                  'orig_text': orig_text,
                  'image_path': image_path}
        file_name = f'{self.outdir}/progress_prc_{LOCAL_RANK}.csv'
    
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df = pd.concat([df, pd.DataFrame([sample])], ignore_index=True)
        else:
            df = pd.DataFrame([sample])
        df.to_csv(file_name, index=False)

    def vis(
        self,
        sample_idx: int,
        adv_text: str,
        image: np.ndarray,
        pred_mask: np.ndarray,
        iteration: int,
        adv_indx: int,
        iou: float,
        color: str = 'jet',
        line_color: str = None,
        **kwargs,
        ):
        
        outdir = self.get_outdir(iteration)
        outpath = outdir / f'sample={sample_idx}_adv={adv_indx}_iou={iou:.3f}.jpg'
        
        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap=color, interpolation='none', alpha=0.3)
        if line_color is not None:
            contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], color=line_color, linewidth=4)
        title = adv_text.replace('$', '\\$').encode('unicode_escape').decode('utf-8')
        if len(title) < 300: plt.title(f'"{title}"')
        plt.savefig(outpath, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    

class AttackEvaluator:
    
    def __init__(self,
                 exp_id, 
                 dataset, 
                 attacker: AttackerBase, 
                 logger: AttackLogger,
                 metrics: t.Dict[str, t.Callable],
                 cfg: edict,
                 ):
        
        self.exp_id = exp_id
        self.cfg = cfg
        self.attacker = attacker
        self.logger = logger
        self.dataset = dataset
        self.model = self.attacker.model
        self.device = self.attacker.kwargs.device
        self.tokenizer = self.attacker.tokenizer
        self.metrics = metrics
        self.processed_texts = {}
        
        self.result_kwargs = [
            'sample_idx',
            'iteration',
            # 'adv_indx',
            'adv_text',
            'orig_text',
        ] + list(metrics.keys())
        
    def __call__(self, sample_idx: int):
        """Attacking single sample and processing results.

        Parameters
        ----------
        sample_idx : int
            Sample's index in the `self.dataset`
        """
        
        setup_deterministic(seed=42, deterministic=False)
        sample = get_sample(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            indx=sample_idx,
        )
        
        results = []
        
        orig_output = next(self.attacker(sample))
        orig_result = self._prc_orig(orig_output, sample_idx)
        results.append(orig_result)
        self.logger.to_csv(results, outfname=f'results_sample={sample_idx}.csv')
        if orig_result['iou'] < 0.10:
            self.logger.print_success(sample_idx, results[0]['orig_text'], sample['image_paths'])
            return results
        for i, output in enumerate(tqdm(self.attacker(sample), total=len(self.attacker)), 1):
            if i % self.cfg.log_steps == 0 or i == len(self.attacker):
                adv_results = self._prc_adv(output, i, sample_idx)
                results.extend(adv_results)
                self.logger.to_csv(results, outfname=f'results_sample={sample_idx}.csv')
        self.logger.print_success(sample_idx, results[0]['orig_text'], sample['image_paths'])
        return results
    
    @functools.lru_cache(maxsize=10)
    def _image(self, image_path: Path) -> np.ndarray:
        return plt.imread(image_path)
    
    def evaluate(
        self,
        adv_text,
        image,
        gen_output,
        inf_output,
        **kwargs) -> edict:

        pred_mask = (inf_output['pred_masks'][0] > 0).int()
        gt_mask = inf_output['gt_masks'][0].int()
        
        orig_text = self.tokenizer.decode(gen_output.orig_output.input_ids)

        return self._evaluate_impl(
            image=image,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            target_mask=gen_output.target_mask,
            adv_text=adv_text,
            orig_text=orig_text,
            **kwargs)
    
    
    def _evaluate_impl(
        self,
        **kwargs) -> edict:
        
        metrics = {name: metric(**kwargs) for name, metric in self.metrics.items()}
        
        return dict(
            **kwargs,
            **metrics,
        )

    def _prc_orig(self, gen_output, sample_idx):
        
        orig_text = self.tokenizer.decode(gen_output.orig_output.input_ids)
        image = self._image(gen_output.sample['image_paths'][0])

        evalout = self.evaluate(
            orig_text, 
            image,
            gen_output,
            gen_output.orig_output,
            sample_idx=sample_idx,
            iteration=0,
            adv_indx=None,
        )

        # self.logger.vis(**evalout)
        
        return {k: evalout[k] for k in self.result_kwargs}
    
    def _prc_adv(self, gen_output: edict, iter_num: int, sample_idx: int):
        
        # print('Generate adversarial texts')
        adv_texts = self.attacker.generate(gen_output)
        # save_texts(adv_texts, folder_path=self.outdir(iter_num))

        image = self._image(gen_output.sample['image_paths'][0])
        
        # print('Get corresponing adversarial masks')
        
        results = []
        if sample_idx not in self.processed_texts:
            self.processed_texts[sample_idx] = set()
        for j, text in enumerate(adv_texts):
            if (not text or text in self.processed_texts[sample_idx]) and (iter_num != len(self.attacker)): continue
            adv_ids = torch.tensor(self.attacker.tokenizer.encode(text, add_special_tokens=False))
            
            inf_output = inference(
                gen_output.sample,
                self.attacker.model,
                gen_output.original_context,
                adv_ids=adv_ids,
                device=self.device)
            
            evalout: edict = self.evaluate(
                text,
                image,
                gen_output,
                inf_output,
                sample_idx=sample_idx,
                iteration=iter_num,
                adv_indx=j,)
            
            # self.logger.vis(**evalout)
            
            results.append({k: evalout[k] for k in self.result_kwargs})
            
            self.processed_texts[sample_idx].add(text)
        
        return results


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
    init_process_group(
        backend='nccl', 
        rank=RANK, 
        world_size=WORLD_SIZE,
        timeout=timedelta(hours=3), 
    )
    device = torch.device("cuda", gpu_num)
    return device
    


@click.command()
@click.option('-o', '--outdir', default='experiments')
@click.option('-i', '--indexes', default='indexes.txt')
@click.option('-e', '--exp_id', type=str, default=None)
@click.option('-c', '--cv2_threads', type=int, default=-1, help='Number of threads for `cv2.setNumThreads`')
@click.option('--eval_one', type=str, help='Path to the input image')
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def cli(outdir, indexes, exp_id, cv2_threads, eval_one, overrides):
    if WORLD_SIZE > 1:
        device = ddp_setup()
    else:
        device = 'cuda'
    
    is_master = RANK == 0
    if exp_id is None:
        exp_id = get_dist_time()
    # Check if results_all exist then do nothing
    # if os.path.exists(f'{outdir}/{exp_id}/results_all.csv'):
    #     print('####################################################')
    #     print(f'Experiment id {exp_id} has been already done!')
    #     return None
    print('*****************************************')
    print(f'WE ARE DOING EXPERIMENT ID {exp_id}')
    if cv2_threads >= 0:
        cv2.setNumThreads(cv2_threads)

    set_seeds()
    
    cfg = load_config(overrides=overrides)

    logger = AttackLogger(cfg, outdir, exp_id)
        
    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]
    tokenizer = setup_tokenizer(cfg)

    model = setup_model(cfg, tokenizer, device=device)
    
    if eval_one:
        chat(eval_one, model, tokenizer, cfg, logger)
        return

    dataset = get_class(cfg.dataset_class)(
        **cfg.dataset,
        vision_tower=cfg.vision_tower,
        tokenizer=tokenizer,
    )

    attacker: AttackerBase = instantiate(
        cfg.attacker, 
        tokenizer,
        model,
        device=device,
    )

    metrics_calc = MetricsCalculator(attacker)
    
    metrics = dict(
        iou=metrics_calc.iou,
        iou_target=metrics_calc.iou_pred_target,
    )

    evaluator = AttackEvaluator(exp_id, dataset, attacker, logger, metrics=metrics, cfg=cfg)
    
    # try:
    #     # Read the file
    #     with open(indexes, 'r') as file:
    #         data = file.read().strip().split()
    #     indexes_all = np.array(data, dtype=int)
    # except FileNotFoundError:
    #     print(f"Error: The file '{indexes}' was not found.")
    # except ValueError:
    #     print("Error: The file must contain only numbers separated by spaces.")
    
    # indexes_all = np.arange(200)
    indexes_all = np.arange(len(dataset))
    if not (cfg.max_dataset_len is None and cfg.min_dataset_len is None):
         indexes_all = indexes_all[cfg.min_dataset_len:cfg.max_dataset_len]
    
    # finding indexes need to be redone
    files_suc_samples = glob(f'{outdir}/{exp_id}/progress_prc_*')
    if len(files_suc_samples) != 0:
        dfs = []
        for i in files_suc_samples:
            df = pd.read_csv(i)
            dfs.append(df)
        final_suc_samples = pd.concat(dfs, ignore_index=True)
        indexes_to_exclude = final_suc_samples['sample_idx'].values
        indexes_remaining = np.setdiff1d(indexes_all, indexes_to_exclude)
    else:
        # If no files are found, keep all indexes
        indexes_remaining = indexes_all
    print(f'{exp_id}: len(indexes_remaining) = {len(indexes_remaining)}')
    if len(indexes_remaining) == 0:
        merge_results(outdir, exp_id)
    indexes = np.array_split(indexes_remaining, WORLD_SIZE)[LOCAL_RANK]
    
    if is_master:
        indexes = tqdm(indexes)
        indexes = indexes
        
    results = []
    for indx in indexes:
        res_indx = evaluator(indx)
        results.extend(res_indx)
        
    if WORLD_SIZE > 1:
        gather_res = [None]*WORLD_SIZE

        # dist.barrier()
        dist.all_gather_object(gather_res, results)
        
        results = []
        for r in gather_res:
            results.extend(r)
    
    if is_master:
        logger.to_csv(results, outfname='results_all.csv')
    
        
if __name__=='__main__':
    cli()