import json, pathlib, argparse, requests, tqdm, pprint, sys, os
from typing import Optional, Set

LLM_URL  = "http://localhost:8001/next_query"
SEG_URL  = "http://localhost:8000/get_iou"
# MAX_STEPS       = 10
MAX_STEPS       = 100
SUCCESS_IOU_THR = 0.10
# SUCCESS_IOU_THR = 1e-6
DATA_DIR        = "dataset/reason_seg/ReasonSeg/test"
DEFAULT_PREFIX  = "adv_search_results_debug"
BASE_IMAGE_DIR  = "dataset"  # Root directory where datasets are stored
SUPPORTED_DATASETS = ["reason_test", "llmseg_test"]
# Note: "reason_test" corresponds to ReasonSeg test split and is the previous default.
# SAVE_PREFIX is taken from the environment variable set by run_pair.sh.
# If the variable is missing we fall back to DEFAULT_PREFIX.
SAVE_PREFIX     = os.getenv("SAVE_PATH", DEFAULT_PREFIX)
SAVE_PATH       = f"{SAVE_PREFIX}.json"

def get_iou(img, q, idx, default=-1.0):# -> Any | Any:
    try:
        r = requests.post(SEG_URL,
                          json={"image_path": str(img), "query": q, 'idx': idx},
                          timeout=120)
        return r.json().get("iou", default)
    except Exception as e:
        print(f"[get_iou] request failed: {e}", file=sys.stderr)
        return default

def next_query(orig_q, prev_q, iou):
    r = requests.post(LLM_URL,
                      json={"orig_query": orig_q, "previous_query": prev_q, "iou": iou},
                      timeout=120)
    return r.json()["adversarial_query"]

def attack_one(image_path, orig_q, idx):
    history, q = [], orig_q
    for step in range(MAX_STEPS):
        cur_iou = get_iou(image_path, q, idx)
        # if cur_iou == -1:
        #     raise ValueError(f'!!!cur_iou={cur_iou}')
        history.append({"step": step, "query": q, "iou": cur_iou})
        if cur_iou <= SUCCESS_IOU_THR: break
        q = next_query(orig_q, q, cur_iou)
    return history

def samples(dataset: str = "reason_test"):
    """Yield (image_path, question) pairs for the selected dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset split to use. Supported values are listed in
        SUPPORTED_DATASETS ("reason_test", "llmseg_test").
    """

    if dataset == "reason_test":
        for fp in pathlib.Path(DATA_DIR).glob("*.json"):
            obj = json.loads(fp.read_text())
            yield fp.with_suffix(".jpg"), obj["text"][0], 0 # HERE ALWAYS 0 instead of idx

    elif dataset == "llmseg_test":
        # Path to the consolidated annotations JSON file.
        data_json_path = os.path.join(BASE_IMAGE_DIR, "llm-seg40k", "validation.json")

        if not os.path.exists(data_json_path):
            raise FileNotFoundError(
                f"Expected consolidated annotations at {data_json_path}. "
                "Please adjust BASE_IMAGE_DIR if your dataset is elsewhere."
            )

        with open(data_json_path, "r", encoding="utf-8") as f:
            full_json = json.load(f)

        ##############################
        img_dir = os.path.join(BASE_IMAGE_DIR, "llm-seg40k", "train2017")
        # Fallback in case the images are directly under llm-seg40k/
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(BASE_IMAGE_DIR, "llm-seg40k")
        ##############################
        
        # Iterate through all image names and yield their corresponding question.
        # for img_name, info in full_json.items():
        #     yield img_name, info["qa_pairs"][0]["question"]
        for idx, (img_name, info) in enumerate(full_json.items()):
            full_img_path = os.path.join(img_dir, img_name)   # <-- build full path
            if not os.path.exists(full_img_path):
                raise FileNotFoundError(
                    f"Image file {full_img_path} does not exist. "
                    "Check BASE_IMAGE_DIR or img_dir."
                )
            yield full_img_path, info["qa_pairs"][0]["question"], idx #HERE IDX!!

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {SUPPORTED_DATASETS}")

def main(max_samples: int, dataset: str = "reason_test", filter_questions: Optional[Set[str]] = None):
    # If the save file already exists, load it and remember which images were processed
    processed_imgs = set()
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as prev:
            for line in prev:
                try:
                    obj = json.loads(line)
                    processed_imgs.add(obj.get("image"))
                except Exception:
                    # Ignore malformed lines
                    continue
    print(f"Loaded {len(processed_imgs)} images from {SAVE_PATH}")
    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        for img, q, idx in tqdm.tqdm(iterable=samples(dataset)):
            # If a filter list is provided, skip questions not in the list
            if filter_questions is not None and q not in filter_questions:
                continue
            # Skip if we've already processed this image
            if str(img) in processed_imgs:
                continue
            if len(processed_imgs) >= max_samples:
                break
            hist = attack_one(img, q, idx)
            f.write(json.dumps({"image": str(img),
                                "history": hist}, ensure_ascii=False) + "\n")
            f.flush()
            processed_imgs.add(str(img))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument(
        "--dataset",
        type=str,
        default="llmseg_test",
        choices=SUPPORTED_DATASETS,
        help="Dataset split to run PAIR attack on."
    )
    p.add_argument("--questions_file", type=str, default=None, help="Path to a file containing one question per line to filter which samples to process.")
    args = p.parse_args()
    # Load question filter list if provided
    filter_set = None
    if args.questions_file is not None:
        with open(args.questions_file, "r", encoding="utf-8") as qf:
            filter_set = {line.strip() for line in qf if line.strip()}
            print(f"Loaded {len(filter_set)} questions from {args.questions_file}")
    main(args.max_samples, args.dataset, filter_set)