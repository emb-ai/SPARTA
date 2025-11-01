import os
from glob import glob
import pandas as pd
import click


@click.command()
@click.option('-o', '--outdir', default='experiments_gumbel')
@click.option('-e', '--exp_num', default=1)
def cli(outdir, exp_num):
    folder_path = os.path.join(outdir, str(exp_num))
    dataframes = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('results_sample')]
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    df = pd.concat(dataframes, ignore_index=True)
    df.to_csv(f'{outdir}/{exp_num}/results_all.csv', index=False)
    
    
if __name__=='__main__':
    cli()