import os
import csv
import pandas as pd

base_dir = '/home/jovyan/zinkovich/ref-seg-text-break/experiments/AAAI-RL-attack/tables/sonar_rl/reason_test/lisa++'
output_file = './processed_texts.txt'

original_texts = []


for root, dirs, files in os.walk(base_dir):
    s = 0
    for file in files:
        if file.startswith('results_sample=') and file.endswith('.csv'):
            s += 1
            df = pd.read_csv(os.path.join(root, file))
            original_texts.append(df['orig_text'][0])
    print(s)

with open(output_file, 'w') as f:
    for text in original_texts:
        f.write(text + '\n')

print(f"Extracted {len(original_texts)} texts to {output_file}")
