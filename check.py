from functools import partial
from hydra.utils import instantiate, get_class
from pathlib import Path
import click
import torch
import tqdm
import hydra
import json
import random
import numpy as np
import os

from lisa_v0.utils.data_processing import get_mask_from_rle_json
from rtext.sentence_optimization.model_setup import setup_model, setup_tokenizer
from rtext.sentence_optimization.utils import calculate_iou
from lisa.model.llava import conversation as conversation_lib
from lisa.utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    AverageMeter, 
    Summary, 
    dict_to_cuda,
    intersectionAndUnionGPU
)
from lisa_v0.utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)


def setup_deterministic(seed):
    """
    set every seed
    """
    torch.set_printoptions(precision=16)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AdversarialDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that applies adversarial text modifications to an existing LISA dataset.
    """
    def __init__(self, base_dataset, adv_samples, data_type, segmod):
        self.base_dataset = base_dataset
        self.adv_samples = adv_samples
        self.data_type = data_type
        self.segmod = segmod
        json_path = 'dataset/llm-seg40k/validation.json'
        with open(json_path, 'r') as f:
            self.json_file = json.load(f)

        
    def __len__(self):
        return len(self.adv_samples)
    
    def __getitem__(self, idx):
        sample_idx, adv_text, *_ = self.adv_samples[idx]
        
        image_path, image, image_clip, _, masks, labels, resize, questions, sampled_classes, inference = self.base_dataset[sample_idx]
        
        if self.data_type == "reason_seg":
            with open(image_path.replace('.jpg', '.json'), 'rb') as f:
                image_info = json.load(f)
                is_sentence = image_info['is_sentence']
        elif self.data_type == "refer_seg":
            is_sentence = False
            if masks.shape[0] > 1:
                masks = masks[0].unsqueeze(0)
        elif self.data_type == "llmseg":
            if masks.shape[0] > 1:
                masks = masks[0].unsqueeze(0)
            *_, is_sentence = get_mask_from_rle_json(self.json_file[image_path.split('/')[-1]])
        
        new_conversations = []
        conv = conversation_lib.default_conversation.copy()

        conv.messages = []
        if self.segmod == 'lisa-13b-v0' or self.segmod == 'lisa-13b-v0-exp':
            image_token_len = 256
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            )
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    replace_token + " {} Please output segmentation mask.".format(adv_text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    replace_token + " What is {} in this image? Please output segmentation mask.".format(adv_text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            new_conversations.append(conv.get_prompt())
        else:
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(adv_text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN + "\n What is {} in this image? Please output segmentation mask.".format(adv_text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            new_conversations.append(conv.get_prompt())
        
        return (
            image_path,
            image,
            image_clip,
            new_conversations,
            masks,
            labels,
            resize,
            questions,
            sampled_classes,
            inference,
        )


def parse_adversarial_csv(csv_path):
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    adv_samples = []
    
    for _, row in df.iterrows():
        sample_idx = int(row['sample_idx'])
        adv_text = row['adv_text']
        orig_text = row['orig_text']
        expected_iou = float(row['iou'])
        
        adv_samples.append((sample_idx, adv_text, orig_text, expected_iou))
    
    return adv_samples


def create_adversarial_dataset(base_dataset, csv_path, data_type, segmod):
    adv_samples = parse_adversarial_csv(csv_path)
    dataset = AdversarialDataset(base_dataset, adv_samples, data_type, segmod)
    print(f"Created adversarial dataset with {len(adv_samples)} samples")
    return dataset


def load_config(overrides, config_path='configs', config_name='config'):
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg


def update_csv_with_results(csv_path, reproduced_iou):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['reproduced_iou'] = reproduced_iou    
    output_path = str(csv_path).replace('.csv', '_reproduced.csv')
    df.to_csv(output_path, index=False)
    print(f"Updated {csv_path} with reproduced IoU values")

    df['error'] = abs(df['reproduced_iou'] - df['iou'])
    with open(output_path, 'a') as f:
        f.write(f"# Average IoU reproduction error: {df['error'].mean():.6f}\n")
    return df


def run_inference(args, input_dict, model_engine):
    torch.cuda.empty_cache()

    input_dict = dict_to_cuda(input_dict)
    if args.precision == "fp16":
        input_dict["images"] = input_dict["images"].half()
        input_dict["images_clip"] = input_dict["images_clip"].half()
    elif args.precision == "bf16":
        input_dict["images"] = input_dict["images"].bfloat16()
        input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
    else:
        input_dict["images"] = input_dict["images"].float()
        input_dict["images_clip"] = input_dict["images_clip"].float()

    with torch.no_grad():
        output_dict = model_engine(**input_dict)
    
    return output_dict


def compute_iou_metrics(pred_masks, gt_masks):
    assert len(pred_masks) == 1 and  len(gt_masks) == 1

    pred_mask = (pred_masks[0] > 0).int()
    gt_mask = gt_masks[0].int()

    intersection, union, acc_iou = 0.0, 0.0, 0.0

    for mask_i, output_i in zip(gt_mask, pred_mask):
        intersection_i, union_i, _ = intersectionAndUnionGPU(
            output_i.contiguous().clone(), mask_i.contiguous().int(), 2, ignore_index=255   # iou for background and foreground
        )
        intersection += intersection_i
        union += union_i
        acc_iou += intersection_i / (union_i + 1e-5)
        acc_iou[union_i == 0] += 1.0  # no-object target
    
    intersection = intersection.cpu().numpy()
    union = union.cpu().numpy()
    acc_iou = acc_iou.cpu().numpy() / gt_mask.shape[0]

    return intersection, union, acc_iou


def validate(args, val_loader, model_engine):
    results = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        output_dict = run_inference(args, input_dict, model_engine)

        pred_masks = output_dict["pred_masks"]
        gt_masks = output_dict["gt_masks"]

        intersection, union, acc_iou = compute_iou_metrics(pred_masks, gt_masks)

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=gt_masks[0].shape[0])

        # results.append(intersection[1] / union[1] if union[1] != 0 else 0)
        pred_mask = (pred_masks[0] > 0).int()
        gt_mask = gt_masks[0].int()
        results.append(calculate_iou(gt_mask, pred_mask))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return results


@click.command()
@click.option('-o', '--outdir', default='experiments')
@click.option('-e', '--exp_id', type=str, default=None)
@click.option('-c', '--check_path', type=str, default=None)
@click.option('-d', '--dataset', type=str, default='reason_test')
@click.option('-g', '--device', type=str, default='cuda')
@click.option('-s', '--segmod', type=str, default='lisa-13b-v0')
def cli(outdir, exp_id, check_path, dataset='reason_test', segmod='lisa-13b-v0', device='cuda'):

    setup_deterministic(42)
    
    cfg = load_config(overrides=('attacker=gumbel', 'dataset={}'.format(dataset), 'segmod@_global_={}'.format(segmod)))

    if 'refcocog' in dataset:
        data_type = "refer_seg"
    elif 'reason' in dataset:
        data_type = "reason_seg"
    else:
        data_type = "llmseg"
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]
    tokenizer = setup_tokenizer(cfg)

    model = setup_model(cfg, tokenizer, device=device)

    dataset = get_class(cfg.dataset_class)(
        **cfg.dataset,
        vision_tower=cfg.vision_tower,
        tokenizer=tokenizer,
    )

    if check_path is not None:
        dataset = create_adversarial_dataset(dataset, Path(outdir) / exp_id / check_path, data_type, segmod)

    if 'v0' in segmod:
        from lisa_v0.utils.dataset import collate_fn
    else:
        from lisa.utils.dataset import collate_fn
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
        ),
    )

    results = validate(cfg, dataloader, model)

    if check_path is not None:
        update_csv_with_results(Path(outdir) / exp_id / check_path, results)


if __name__=='__main__':
    cli()
