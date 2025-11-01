import json, shutil
import os, torch, sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from lisa.utils.dataset import collate_fn
from lisa.model.llava import conversation as conversation_lib
from lisa.utils.data_processing import get_mask_from_json, get_mask_from_rle_json

from rtext.sentence_optimization.model_setup import setup_model, setup_tokenizer
from rtext.sentence_optimization.inference import inference
from hydra.utils import get_class
from omegaconf import OmegaConf
import uvicorn
import hydra
from hydra.utils import instantiate, get_class
import click
from PIL import Image  
from rtext.sentence_optimization.utils import calculate_iou
import shlex

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)  


def prepare_tmp_files(src_img_path: Path, new_query: str) -> Path:
    """
    1. Remove any JPG / JSON in tmp/ that is not the requested pair.
    2. Ensure the requested <name>.jpg / <name>.json are present there
       (copy if needed).
    3. Rewrite the JSON so that  data["text"] == [new_query].
    4. Return the *image* path inside tmp/.
    """
    if 'reason' in str(src_img_path):
        src_json_path = src_img_path.with_suffix(".json")

        dst_img_path  = TMP_DIR / src_img_path.name
        dst_json_path = TMP_DIR / src_json_path.name

        # --- 1) purge foreign files in tmp/ ------------------------------
        for f in TMP_DIR.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"} and \
            f.name not in {dst_img_path.name, dst_json_path.name}:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[warn] could not delete {f}: {e}")

        # --- 2) copy if missing -----------------------------------------
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        if not dst_json_path.exists():
            shutil.copy2(src_json_path, dst_json_path)

        # --- 3) patch the JSON text field -------------------------------
        try:
            with open(dst_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:                         # malformed or empty
            print('Everything is broken!')
            sys.exit(1)

        data["text"] = [new_query]                # <-- inject query
        with open(dst_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return dst_img_path
    else:
        target_dir = src_img_path.parents[1]  # This will be .../llm-seg40k
        src_json_path = target_dir / 'validation.json'

        dst_img_path  = TMP_DIR / src_img_path.name
        dst_json_path = TMP_DIR / src_json_path.name

        # --- 1) purge foreign files in tmp/ ------------------------------
        for f in TMP_DIR.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"} and \
            f.name not in {dst_img_path.name, dst_json_path.name}:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[warn] could not delete {f}: {e}")

        # --- 2) copy if missing -----------------------------------------
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        if not dst_json_path.exists():
            shutil.copy2(src_json_path, dst_json_path)

        # --- 3) patch the JSON text field -------------------------------
        try:
            with open(dst_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:                         # malformed or empty
            print('Everything is broken!')
            sys.exit(1)

        data[f"{src_img_path.name}"]['qa_pairs'][0]['question'] = new_query                # <-- inject query
        with open(dst_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return dst_img_path

def load_config(overrides, config_path='configs', config_name='config'):
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg


def get_sample(dataset, tokenizer, indx=0, req_indx=0):
    sample = collate_fn(
        [dataset[indx]], 
        tokenizer)
    
    sample['input_ids'] = sample['input_ids'][req_indx].unsqueeze(0)
    sample['labels'] = sample['labels'][req_indx].unsqueeze(0)
    sample['attention_masks'] = sample['attention_masks'][req_indx].unsqueeze(0)
    sample['conversation_list'] = [sample['conversation_list'][req_indx]]
    if sample['masks_list'][req_indx].shape[0] > 1:
        sample['masks_list'] = [sample['masks_list'][req_indx][req_indx].unsqueeze(0)]
    else:
        sample['masks_list'] = [sample['masks_list'][req_indx]]
    sample['offset'] = torch.tensor([0, len(sample['conversation_list'])])
    return sample


# mask to physical GPUs 2-3
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"#"2,3"#
device = "cuda"      # inside this process gpu-0==real-gpu-2

cfg = tokenizer = model = None
# cfg = load_config(overrides=overrides)

# conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]
# tokenizer = setup_tokenizer(cfg)

# model = setup_model(cfg, tokenizer, device=device)
# ---------------- global placeholders ------------------------------

def init_model(overrides):                                         # NEW
    """Load Hydra cfg, tokenizer and model once at start-up."""
    global cfg, tokenizer, model
    cfg = load_config(overrides, '../configs')
    conversation_lib.default_conversation = \
        conversation_lib.conv_templates[cfg.conv_type]
    tokenizer = setup_tokenizer(cfg)
    model     = setup_model(cfg, tokenizer, device=device)
    return cfg, tokenizer, model


class Req(BaseModel):
    image_path: str
    query:      str
    idx:        int

class Resp(BaseModel):
    iou: float

app = FastAPI()

@app.post("/get_iou", response_model=Resp)
@torch.no_grad()
def get_iou(req: Req):
    global cfg, tokenizer, model
    # prepare tmp/ and get the path to the copied image
    tmp_img_path = prepare_tmp_files(Path(req.image_path), req.query)

    # re-point Hydra dataset to tmp/
    cfg.dataset['base_image_dir'] = 'tmp'
    cfg.dataset['val_dataset']    = tmp_img_path.name

    dataset = get_class(cfg.dataset_class)(
        **cfg.dataset,
        vision_tower=cfg.vision_tower,
        tokenizer=tokenizer,
    )

    print(f'len(dataset) = {len(dataset)}')
    sample = get_sample(dataset, tokenizer)#, req.idx)
    output = inference(sample, model)
    pred_mask = (output['pred_masks'][0] > 0).int()
    gt_mask = output['gt_masks'][0].int()
    
    iou = calculate_iou(gt_mask, pred_mask)

    # pred_png = TMP_DIR / f"{tmp_img_path.stem}_pred.png"
    # gt_png   = TMP_DIR / f"{tmp_img_path.stem}_gt.png"

    # Image.fromarray(pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255)\
    #      .save(pred_png)
    # Image.fromarray(gt_mask.squeeze().cpu().numpy().astype(np.uint8) * 255)\
    #      .save(gt_png)

    return {"iou": iou}


# ---------------- CLI wrapper  -------------------------------------  NEW
@click.command(context_settings=dict(ignore_unknown_options=True,
                                     allow_extra_args=True))
@click.option('--port', default=8000, show_default=True, help='REST port')
@click.option('--log-level', default='info', show_default=True)
@click.argument('overrides', nargs=-1, type=click.UNPROCESSED)
def cli(port, log_level, overrides):
    """
    Starts the segmentation REST service.

    Positional arguments after the options are forwarded verbatim to
    Hydra as configuration overrides, e.g.

        seg_service.py 'model.name=ResNet' 'dataset.foo=bar'
    """
    init_model(overrides=overrides)
    # print(get_iou({'image_path': 'dataset/reason_seg/ReasonSeg/test/241852123_1c8229b3e7_o.jpg',
    #                'query': 'What is the object that the person in the picture is holding onto while walking his dog?'},
    #                cfg, tokenizer, model))
    # uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(
        app,                        # ← pass the *object*, not "module:app"
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=False,               # ← make sure no worker-respawn occurs
        workers=1                   # (default);  keep single process
    )

if __name__ == "__main__":                                             # NEW
    cli()