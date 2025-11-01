#!/usr/bin/env python3
"""
Convert line-delimited JSON of the form
{"image": "...", "history":[{"step":0, "query": "...", "iou": 0.12}, …]}
into
  results_sample={idx}.csv   (one per input line)
and a merged                  results_all.csv
Columns: sample_idx | iteration | adv_text | orig_text | iou
"""

import json, csv, pathlib, sys
from typing import List, Dict
import pandas as pd          # Only used for the final merge; pip install pandas if needed

# ---------- CONFIG ----------
PATH = "experiments/AAAI-RL-attack/tables/qwen-pair-noexample-100/reason_test"
MODEL = "gsva-13b-llama2-ft-res"
INPUT_FILE  = f"{PATH}/adv_search_results_{MODEL}.json"                   # change if needed
OUT_DIR     = pathlib.Path(f"{PATH}/{MODEL}")
OUT_DIR.mkdir(exist_ok=True)
MERGED_CSV  = f"{PATH}/{MODEL}/results_all.csv"
# ---------- /CONFIG ----------

FIELDNAMES = ["sample_idx", "iteration", "adv_text", "orig_text", "iou"]
all_rows: List[Dict] = []                    # will hold rows from every sample for the merged file

with open(INPUT_FILE, encoding="utf-8") as f_in:
    for sample_idx, line in enumerate(f_in):
        if not line.strip():                 # skip empty lines
            continue

        item = json.loads(line)
        history = item["history"]
        orig_text = history[0]["query"]      # text from step 0

        sample_rows: List[Dict] = []
        for h in history:
            sample_rows.append(
                {
                    "sample_idx": sample_idx,
                    "iteration" : h["step"],
                    "adv_text"  : h["query"],
                    "orig_text" : orig_text,
                    "iou"       : h["iou"],
                }
            )

        # write results_sample={idx}.csv
        sample_path = OUT_DIR / f"results_sample={sample_idx}.csv"
        with sample_path.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(sample_rows)

        all_rows.extend(sample_rows)         # accumulate for the merged file

# ---------- merged CSV ----------
pd.DataFrame(all_rows)[FIELDNAMES].to_csv(MERGED_CSV, index=False)
print(f"Created {len(list(OUT_DIR.glob('results_sample=*.csv')))} per-sample CSVs "
      f"and the merged file: {MERGED_CSV}")