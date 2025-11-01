import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import textwrap

# --------------------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------------------

def build_df_paths(dataset: str, csv_file: str, root: Path, methods: List[str]) -> Dict[str, Path]:
    """Return a mapping of method_name -> Path to csv with metrics for each top-level *method* folder."""

    mapping: Dict[str, Path] = {}
    for prefix in methods:
        def make_path(suffix: str) -> Path:
            return root / prefix / dataset / suffix / csv_file

        mapping.update(
            {
                # f"{prefix}, lisa-7b-v1": make_path("lisa-7b-v1"),
                # f"{prefix}, lisa-7b-v1-exp": make_path("lisa-7b-v1-exp"),
                # f"{prefix}, lisa-13b-v1": make_path("lisa-13b-v1"),
                f"{prefix}, lisa-13b-v1-exp": make_path("lisa-13b-v1-exp"),
                # f"{prefix}, lisa++": make_path("lisa++"),
                # f"{prefix}, gsva-13b-llama2-ft-res": make_path("gsva-13b-llama2-ft-res"),
            }
        )
    return mapping


def load_metrics(datasets: Dict[str, Path]) -> pd.DataFrame:
    """Concatenate csvs into a single dataframe."""
    dfs: List[pd.DataFrame] = []
    for name, csv_path in datasets.items():
        if not csv_path.exists():
            logging.warning("%s does not exist – skipping", csv_path)
            continue
        df = pd.read_csv(csv_path, lineterminator="\n")
        df.rename(columns={"iou": "adv_iou"}, inplace=True)
        df["method_name"] = name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No input csvs found.")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


# --------------------------------------------------------------------------------------
# Attack success helpers
# --------------------------------------------------------------------------------------

def cosine_scoring(df: pd.DataFrame, cos_threshold: float) -> pd.DataFrame:
    """Calculate cosine similarity between original and adversarial texts."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device='cuda')
    encode_fn = lambda texts: model.encode(texts, convert_to_tensor=True, device='cuda', normalize_embeddings=True)

    def cosine_similarity(text1: str, text2: str) -> float:
        emb1, emb2 = encode_fn([text1, text2])
        return float((emb1 @ emb2.T).item())

    df['qwen_cosine'] = df.apply(lambda x: cosine_similarity(x['orig_text'], x['adv_text']), axis=1)
    df.loc[df['qwen_cosine'] < cos_threshold, 'SCORE_Qwen/Qwen3-32B'] = 0
    return df


def check_cos_threshold(df: pd.DataFrame, cos_thrld: float) -> None:
    """Check that all cosine scores in df are greater than cos_threshold."""
    low_cos_mask = (df["qwen_cosine"] < cos_thrld) & (df["SCORE_Qwen/Qwen3-32B"] != 0)
    if low_cos_mask.any():
        df.loc[low_cos_mask, "SCORE_Qwen/Qwen3-32B"] = 0
        print(f"WARNING: {low_cos_mask.sum()} rows have non-zero score and cosine < {cos_thrld} – scores zeroed.")
    else:
        print("All rows with cosine < threshold have zero score.")


def check_texts(df: pd.DataFrame, chosen_texts: Path) -> None:
    """Check that all texts in df are in chosen_texts."""
    with open(chosen_texts, "r") as f:
        chosen_texts = set(f.read().splitlines())
    print(f"\nFound {len(chosen_texts)} chosen texts")
    print(f"Found {len(df['orig_text'].unique())} unique texts in df\n")
    if not df["orig_text"].isin(chosen_texts).all():
        print(f"WARNING: {df['orig_text'].isin(chosen_texts).sum()} texts in df are not in chosen_texts")
        df = df[df["orig_text"].isin(chosen_texts)]
    else:
        print("All texts in df are in chosen_texts!")
    return chosen_texts


def load_raw_attacks(exp_path: Path, chosen_texts: set) -> pd.DataFrame:
    """Read attack result chunks and optionally filter by given texts."""
    csv_files = list(exp_path.glob("results_sample=*.csv"))
    frames = [pd.read_csv(p) for p in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["orig_text"].isin(chosen_texts)]
    return df


def compute_success_counts(df_paths: Dict[str, Path], chosen_texts: set) -> Dict[str, int]:
    """Read raw attack results, since some files may not reach final .csv file."""
    counts: Dict[str, int] = {}
    for name, path in tqdm(df_paths.items(), desc="Parsing raw attacks"):
        exp_dir = path.parent  # directory that contains csv files
        raw_df = load_raw_attacks(exp_dir, chosen_texts)
        counts[name] = raw_df[(raw_df["iteration"] == 0) & (raw_df["iou"] >= 0.1)].shape[0]
    return counts


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------

EVALUATION_SCORE_COLS = {
    "nemotron": {'name': "SCORE_nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", 'accepted_marks': [5]},
    "qwen": {'name': "SCORE_Qwen/Qwen3-32B", 'accepted_marks': [5]},
    "raw": {'name': "score", 'accepted_marks': [1]},
}

def make_threshold_plot(
    df: pd.DataFrame,
    n_success: Dict[str, int],
    evaluation_method: str,
    output: Path,
):
    """Generate IoU-threshold curve and save it to *output*."""

    score_col = EVALUATION_SCORE_COLS[evaluation_method]['name']
    accepted_marks = EVALUATION_SCORE_COLS[evaluation_method]['accepted_marks']

    thresholds = np.arange(0.01, 1.01, 0.01)
    methods = sorted(df["method_name"].unique())
    methods_means = {m: [] for m in methods}

    for thr in thresholds:
        df_thr = df.copy()
        df_thr["success"] = (df_thr["orig_iou"] - df_thr["adv_iou"]) / df_thr["orig_iou"] > thr
        for m in methods:
            d_m = df_thr[df_thr["method_name"] == m]
            d_m = d_m[d_m[score_col].isin(accepted_marks) & d_m["success"]]
            methods_means[m].append(len(d_m) / n_success[m])

    sns.set_palette("husl")
    plt.figure(figsize=(5, 4))
    for m in methods:
        plt.plot(thresholds, methods_means[m], label=m, marker="o", linewidth=2)

    plt.xlabel("IoU Threshold")
    plt.ylabel("Fraction of successful attacks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    logging.info("Saved threshold plot to %s", output)


# --------------------------------------------------------------------------------------
# AUC / metric table
# --------------------------------------------------------------------------------------

def compute_metrics_table(
    df: pd.DataFrame,
    n_success: Dict[str, int],
    evaluation_method: str,
    thresholds_sets: List[List[int]] | None = None,
    accepted_marks: List[int] | None = None,
) -> pd.DataFrame:
    """Return dataframe with AUC / single-threshold metrics for each method."""

    thresholds_sets = thresholds_sets or [[5], [10], list(range(1, 101))]
    score_col = EVALUATION_SCORE_COLS[evaluation_method]['name']
    accepted_marks = EVALUATION_SCORE_COLS[evaluation_method]['accepted_marks']
    methods = sorted(df["method_name"].unique())

    records: List[dict] = []
    for thr_set in thresholds_sets:
        thr_arr = np.array(thr_set) / 100
        for m in methods:
            successes: List[float] = []
            for thr in thr_arr:
                success_mask = (df["orig_iou"] - df["adv_iou"]) / df["orig_iou"] > thr
                d_m = df[(df["method_name"] == m) & success_mask & df[score_col].isin(accepted_marks)]
                successes.append(len(d_m) / n_success[m])
            if len(successes) > 1:
                auc = np.trapz(successes, thr_arr) * 100
                metric_name = f"AUC_{thr_set[0]}_{thr_set[-1]}" if len(thr_set) > 1 else f"thr_{thr_set[0]}"
                records.append({"method_name": m, "metric": metric_name, "value": auc})
            else:
                records.append({"method_name": m, "metric": f"thr_{thr_set[0]}", "value": successes[0] * 100})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot attack metrics across methods.")
    parser.add_argument("--dataset", default="reason_test", choices=["llmseg_test", "reason_test"], help="Dataset name")
    parser.add_argument("--csv-file", default="fixed_best_filter_qwen3_cos=0.825.csv", help="CSV filename containing metrics")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("./experiments/AAAI-RL-attack/tables"),
        help="Root directory that contains method sub-folders",
    )
    parser.add_argument("--evaluation-method", choices=list(EVALUATION_SCORE_COLS.keys()), default="qwen")
    parser.add_argument("--methods", nargs="+", default=["qwen-pair"], help="Top-level method folders to include (space-separated list)")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where to save outputs (plots, csvs)")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--cos-threshold", type=float, default=0.825, help="Cosine threshold for Qwen")
    parser.add_argument("--calculate-cosine", action="store_true", help="Calculate cosine similarity")
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    default_chosen_map = {
        "llmseg_test": Path("./texts_llmseg.txt"),
        "reason_test": Path("./texts_reason.txt"),
    }
    args.chosen_texts = default_chosen_map[args.dataset]

    df_paths = build_df_paths(args.dataset, args.csv_file, args.root_path, args.methods)
    df = load_metrics(df_paths)

    logging.info("adv_iou range: (%.3f, %.3f)", df["adv_iou"].min(), df["adv_iou"].max())
    assert df["adv_iou"].min() >= 0, "some adv_iou is not greater than 0"
    chosen_texts = check_texts(df, args.chosen_texts)
    df = df[df["orig_text"].isin(chosen_texts)]

    n_success = compute_success_counts(df_paths, chosen_texts)
    print(n_success)

    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # plot_path = args.output_dir / "threshold_curve.png"
    # make_threshold_plot(df, n_success, args.evaluation_method, plot_path)

    # фильтр по mehtods_name
    def get_first_bad_or_max(group):
        bad_iter = group.loc[group['adv_iou'] <= 0.1, 'iteration']
        if not bad_iter.empty:
            return bad_iter.min()
        else:
            return group['iteration'].max()

    first_bad_iter = df.groupby('sample_idx').apply(get_first_bad_or_max)
    mask = df['iteration'] <= df['sample_idx'].map(first_bad_iter)
    df = df[mask]   

    df['iou_drop'] = df['orig_iou'] - df['adv_iou']
    df_sorted = df.sort_values(
        by=['sample_idx', 'SCORE_Qwen/Qwen3-32B', 'iou_drop'], ascending=[True, False, False]
    )
    df = df_sorted.groupby('sample_idx').first().reset_index()

    if args.calculate_cosine:
        df = cosine_scoring(df, args.cos_threshold)
        check_cos_threshold(df, args.cos_threshold)

    metrics_df = compute_metrics_table(df, n_success, args.evaluation_method)

    # --------------------------------------------------------------------------------------
    # Console + CSV output in Excel-friendly wide format
    # --------------------------------------------------------------------------------------

    wide = metrics_df.pivot_table(index="method_name", columns="metric", values="value")

    wide = wide.rename(columns={
        "AUC_1_100": "auc@[0.01,1]",
        "thr_5": "y_thr=5",
        "thr_10": "y_thr=10",
    })
    order_map = {
        # "lisa-7b-v1": 0,
        # "lisa-7b-v1-exp": 1,
        # "lisa-13b-v1": 2,
        "lisa-13b-v1-exp": 0,
        # "lisa++": 0,                    
        # "gsva-13b-llama2-ft-res": 5,
    }
    wide = wide.reindex(
        sorted(wide.index, key=lambda m: order_map.get(m.split(",")[-1].strip(), 999))
    )

    header = f"{'Method name':40}\tauc@[0.01,1]\ty_thr=5\ty_thr=10"
    print(header)
    print("-" * len(header))

    for method, row in wide.iterrows():
        print(f"{method:40}\t{row.get('auc@[0.01,1]', float('nan')):.1f}\t{row.get('y_thr=5', float('nan')):.1f}\t{row.get('y_thr=10', float('nan')):.1f}")

    # --------------------------------------------------------------
    # Additional vertical print-out grouped by method prefix
    # --------------------------------------------------------------

    for prefix in args.methods:
        subset = wide[wide.index.str.startswith(prefix)]
        if subset.empty:
            continue
        print("\n" + "=" * 60)
        print(f"Method name = {prefix}")
        for metric_label in ["auc@[0.01,1]", "y_thr=5", "y_thr=10"]:
            print(metric_label)
            for v in subset[metric_label]:
                print(f"{v:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    cli()