# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_names = {
    'gbda' : '/home/jovyan/zinkovich/ref-seg-text-break/experiments/EMNLP-paper/orig-comparison/ours-200/fixed_best_filter.csv',
    'sonar' : '/home/jovyan/zinkovich/ref-seg-text-break/experiments/sonar-exps/sonar-old/fixed_best_filter.csv'
}

dfs = []
for method_name, path in df_names.items():
    df = pd.read_csv(path)
    df.rename(columns={'iou': 'adv_iou'}, inplace=True)
    df['method_name'] = method_name
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df['iou_diff'] = (df['orig_iou'] - df['adv_iou'])
df['iou_diff_p'] = (df['orig_iou'] - df['adv_iou']) / df['orig_iou']

evaluation_method = 'nemotron'

if evaluation_method == 'nemotron':
    score = 'SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'   
    max_score = 5
else:
    score = 'score'   
    max_score = 1

import matplotlib.pyplot as plt

thresholds = []
methods_means = {}

for method_name in set(df['method_name']):
    methods_means[method_name] = []

for threshold in range(0, 101, 2):
    df['success'] = df['iou_diff_p'] * 100 > threshold

    for method_name in set(df['method_name']):
        df_method = df[df['method_name'] == method_name]
        df_method = df_method[(df_method[score] == max_score) & df_method['success']]
        methods_means[method_name].append(len(df_method))

    thresholds.append(threshold) 
    
custom_palette = ['#FF0080', '#16C47F', 'red', 'orange',][:len(set(df['method_name']))]

plt.figure(figsize=(5, 4))

for method_name, color in zip(set(df['method_name']), custom_palette):
    plt.plot(thresholds, methods_means[method_name], label=method_name, marker='o',color=color, markersize=4, linewidth=2)

plt.xlabel('IoU Threshold, %')
plt.ylabel('Attack Success Rate')
plt.legend()

print('successfully plotted')
plt.savefig('ASR_plot_high_dpi.png', dpi=300, bbox_inches='tight')
plt.show()