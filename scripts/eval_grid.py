import subprocess
import click
import itertools
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from glob import glob

def check_remaining_samples(exp_dir, exp_id, n_samples=None):
    if n_samples is None:
        if 'llmseg' in exp_id or 'refcocog' in exp_id:
            n_samples = 1000
        else:
            n_samples = 779
    files_suc_samples = glob(f'{exp_dir}/{exp_id}/progress_prc_*')
    if len(files_suc_samples) != 0:
        dfs = []
        for i in files_suc_samples:
            df = pd.read_csv(i)
            dfs.append(df)
        final_suc_samples = pd.concat(dfs, ignore_index=True)
        indexes_to_exclude = final_suc_samples['sample_idx'].values
        indexes_remaining = np.setdiff1d(np.arange(n_samples), indexes_to_exclude)
        return len(indexes_remaining)
    return n_samples

def is_already_processed(exp_dir, exp_id):
    file = f'{exp_dir}/{exp_id}/fixed_best_filter.csv'
    if Path(file).exists():
        return True
    return False

def get_cmd(
    exp_dir,
    exp_id,
    cuda_ids,
    cosine_threshold=0,
    stop_nemotron=False):
    cmd = [
        'bash', 
        '-c',
        f'CUDA_VISIBLE_DEVICES={cuda_ids} '
        f'python '
        'filter_attacks_qwen.py '
        f'--exp-dir={exp_dir} '
        f'--exp-id={exp_id} '
        f'--name=sonar_val_{exp_id} '
        f'--stop-nemotron={stop_nemotron} '
        f'--cosine-threshold={cosine_threshold} '
    ]
    return cmd


@click.command()
@click.option('-e', '--exp_dir', type=Path)
@click.option('--cuda_ids', type=str)
@click.option('-p', '--parallel_experiments', type=int, default=4)
@click.option('-n', '--num_experiments', type=int, default=None)
@click.option('-cos', '--cosine-threshold', type=float, default=0.0, show_default=True, help='Cosine similarity threshold for filtering')
def cli(exp_dir, cuda_ids, parallel_experiments, num_experiments, cosine_threshold):
    # exp_ids = sorted([path.name for path in exp_dir.glob('*')])
    
    exp_dir = 'experiments/AAAI-RL-attack/tables/qwen-pair'
    exp_ids = [
        'llmseg_test/lisa-7b-v1',
        # 'llmseg_test/lisa-13b-v1',
        # 'llmseg_test/lisa-7b-v1-exp',
        # 'llmseg_test/lisa-13b-v1-exp',
        # 'llmseg_test/lisa++',
        # 'llmseg_test/gsva-13b-llama2-ft-res',
        # 'reason_test/lisa-7b-v1',
        # 'reason_test/lisa-13b-v1',
        # 'reason_test/lisa-7b-v1-exp',
        # 'reason_test/lisa-13b-v1-exp',
        # 'reason_test/lisa++',
        # 'reason_test/gsva-13b-llama2-ft-res',
    ]
    
    print([check_remaining_samples(exp_dir, exp_id, num_experiments) for exp_id in exp_ids])

    # exp_ids = [exp_id for exp_id in exp_ids if check_remaining_samples(exp_dir, exp_id, num_experiments) == 0 and not is_already_processed(exp_dir, exp_id)]
    print(f'Will process {len(exp_ids)} experiments')

    with ThreadPoolExecutor(max_workers=parallel_experiments) as executor:
        future_to_exp = {
            executor.submit(subprocess.run, get_cmd(exp_dir, exp_id, cuda_ids, cosine_threshold), check=True): exp_id
            for exp_id in exp_ids
        }

        for future in as_completed(future_to_exp):
            exp_id = future_to_exp[future]
            try:
                future.result()
                print(f"Задача для {exp_id} завершена успешно.")
            except subprocess.CalledProcessError as e:
                print(f"Ошибка в задаче для {exp_id}: {e}")


if __name__ == '__main__':
    cli()