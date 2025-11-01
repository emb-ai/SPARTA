import subprocess
import click
from pathlib import Path
import tqdm
from glob import glob
import pandas as pd
import numpy as np


def check_remaining_samples(exp_dir, exp_id):
    files_suc_samples = glob(f'{exp_dir}/{exp_id}/progress_prc_*')
    if len(files_suc_samples) != 0:
        dfs = []
        for i in files_suc_samples:
            df = pd.read_csv(i)
            dfs.append(df)
        final_suc_samples = pd.concat(dfs, ignore_index=True)
        indexes_to_exclude = final_suc_samples['sample_idx'].values
        indexes_remaining = np.setdiff1d(np.arange(200), indexes_to_exclude)
        return len(indexes_remaining)
    return 200

def get_cmd(
    exp_dir,
    exp_id
    ):
    cmd = [
        'bash', 
        '-c',
        'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 '
        'torchrun --nproc_per_node=8 --master_port=25901 '
        'filter_attacks.py '
        f'--exp-dir={exp_dir} '
        f'--exp-id={exp_id} '
    ]
    return cmd

@click.command()
@click.option('-e', '--exp_dir', type=Path)
def cli(exp_dir):
    exp_ids = sorted([path.name for path in exp_dir.glob('*')])
    
    progress_bar = tqdm.tqdm(total=len(exp_ids), desc="Processing experiments")

    for exp_id in exp_ids:
        indexes_remaining = check_remaining_samples(exp_dir, exp_id)
        if indexes_remaining == 0:
            print(f"Will process {exp_id} since it was finished")
            try:
                cmd = get_cmd(exp_dir, exp_id)
                print(f"\nProcessing experiment: {exp_id}")
                subprocess.run(cmd, check=True)
                print(f"Successfully completed experiment: {exp_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error in experiment {exp_id}: {e}")
                with open('failed_eval_logs.txt', 'a') as f:
                    f.write(f"Error in experiment {exp_id}: {e}\n")
        else:
            print(f"Skipping {exp_id} since it was not finished, {indexes_remaining} samples processed")
        
        progress_bar.update(1)

    progress_bar.close()

if __name__ == '__main__':
    cli()