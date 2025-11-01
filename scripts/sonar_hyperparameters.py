import subprocess
import click
import concurrent.futures
import itertools
import tqdm
import queue
import os

DICE_WEIGHTS = [0.5]
BCE_WEIGHTS = [2]
PERP_WEIGHTS = [0, 0.5, 1]
SIM_WEIGHTS = [0, 100, 300, 500]
LR_NUM_ITERS = [(1e-3, 15), (3e-4, 30), (1e-4, 60)]

GRID_LIST = list(itertools.product(
    DICE_WEIGHTS,
    BCE_WEIGHTS,
    PERP_WEIGHTS,
    SIM_WEIGHTS,
    LR_NUM_ITERS))

# GRID_LIST = GRID_LIST[2::3][:7]
# GRID_LIST = GRID_LIST[2::3][7:] + GRID_LIST[1::3][:4]
GRID_LIST = GRID_LIST[::3] + GRID_LIST[1::3][4:]

MAX_DATASET_LEN = 200

EXPERIMENTS_DIR = 'experiments_sonar_v0'

@click.command()
def cli():
    # Create a queue for all jobs
    job_queue = queue.Queue()

    # Populate the queue with all job combinations
    for dice, bce, perp, sim, (lr, num_iters) in GRID_LIST:
        job_queue.put((dice, bce, perp, sim, lr, num_iters))

    total_jobs = job_queue.qsize()
    progress_bar = tqdm.tqdm(total=total_jobs, desc="Running Experiments")

    while not job_queue.empty():

        try:
            dice, bce, perp, sim, lr, num_iters = job_queue.get_nowait()
        except queue.Empty:
            break

        success = False
        error = ""

        exp_id = f'{dice=}_{bce=}_{sim=}_{perp=}_{lr=}_{num_iters=}'
        cmd = [
            "bash", "-c",
            f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 '
            f'torchrun --nproc_per_node=8 attack_main.py '
            f'--outdir={EXPERIMENTS_DIR} '
            f'--exp_id={exp_id} '
            f'attacker=sonar_cv attacker.dice_weight={dice} '
            f'attacker.bce_weight={bce} attacker.sim_weight={sim} '
            f'attacker.perp_weight={perp} attacker.lr={lr} '
            f'attacker.num_iters={num_iters} dataset=reason_val segmod@_global_=lisa-13b-v0 '
            f'max_dataset_len={MAX_DATASET_LEN} '
        ]

        try:
            print(cmd)
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            success = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e.stderr}")
            error = e.stderr

        if not success:
            with open('failed_hyper_logs.txt', 'a') as log_file:
                log_file.write(f"Warning: Experiment {exp_id} with error {error}.\n")

        progress_bar.update(1)

    progress_bar.close()

if __name__ == '__main__':
    cli()
