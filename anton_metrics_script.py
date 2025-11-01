import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_names = {
    'sonar_rl' : '/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/ref-seg-text-break/experiments/05-13-2025_10-27-13-232379/fixed_best_filter.csv',
    'rtext' : '/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/ref-seg-text-break/experiments/EMNLP-paper/sonar/reason_test/lisa-13b-v1-exp/fixed_best_filter.csv'
}

dfs = []
for method_name, path in df_names.items():
    df = pd.read_csv(path)
    # df.rename(columns={'iou': 'adv_iou'}, inplace=True)
    df['method_name'] = method_name
    dfs.append(df)

dfs[1] = dfs[1][dfs[1]['sample_idx'].isin(dfs[0]['sample_idx'])]
df = pd.concat(dfs, ignore_index=True)

df['iou_diff'] = df['iou_drop']
df['iou_diff_p'] = (df['orig_iou'] - df['adv_iou']) / df['orig_iou']

evaluation_method = 'nemotron'

if evaluation_method == 'nemotron':
    score = 'SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'   
    valid_max_scores = [5]
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
        
        df_method = df_method[(df_method[score].isin(valid_max_scores)) & df_method['success']]
        methods_means[method_name].append(len(df_method) / sum(df['method_name'] == method_name))

    thresholds.append(threshold) 
    
custom_palette = ['#FF0080', '#16C47F', 'red', 'orange',][:len(set(df['method_name']))]

plt.figure(figsize=(5, 4))

for method_name, color in zip(set(df['method_name']), custom_palette):
    plt.plot(thresholds, methods_means[method_name], label=method_name, marker='o',color=color, markersize=4, linewidth=2)

plt.xlabel('IoU Threshold, %')
plt.ylabel('ratio of Successful Attacks')
plt.legend()
plt.ylim([0, 1])

print('successfully plotted')
plt.savefig('andrew_debug/ASR_plot_high_dpi.png', dpi=300, bbox_inches='tight')
plt.show()

columns = ['sample_idx', 'adv_text', 'orig_text', 'iou_drop', 'SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF']
joined = pd.merge(left=dfs[0][columns], right=dfs[1][columns], on='sample_idx', how='inner', suffixes=['_sonar_rl', '_rtext'])
a = joined.drop(joined[~(joined['SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF_sonar_rl'] > joined['SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF_rtext'])].index)
a.to_csv('andrew_debug/joined_larger_score.csv', index=False)
b = joined.drop(joined[~((joined['iou_drop_sonar_rl'] > joined['iou_drop_rtext']) & (joined['SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF_sonar_rl'] == joined['SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF_rtext']))].index)
b.to_csv('andrew_debug/joined_larger_drop.csv', index=False)