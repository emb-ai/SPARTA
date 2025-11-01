import transformers
import torch
from ..new_lisa import LisaModelEmbedInput
from ..new_lisa_v0 import LisaModelEmbedInputv0
from ..new_gsva import LisaGSVAEmbedInput
from lisa.utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig
from typing import Any
import glob
import os

DEFAULT_DEVICE = 'cuda'
DEFAULT_TORCH_DTYPE = torch.half


def setup_tokenizer(cfg):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        model_max_length=cfg.model_max_length,
        padding_side=cfg.padding_side,
        use_fast=cfg.use_fast,
        cache_dir=cfg.cache_dir,
        legacy=cfg.legacy,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    if cfg.name in ["lisa-13b-v0", "lisa-13b-v0-exp"]:
        _ = tokenizer.add_tokens("[SEG]")

    elif "gsva" in cfg.name:
        _ = tokenizer.add_tokens("[SEG]")
        _ = tokenizer.add_tokens("[REJ]")
    
    if cfg.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    return tokenizer
   

def setup_model(
    cfg,
    tokenizer: AutoTokenizer,
    device: str = DEFAULT_DEVICE,
    torch_dtype: torch.dtype = DEFAULT_TORCH_DTYPE
    ) -> Any:

    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    if cfg.name in ["lisa-7b-v1", "lisa-7b-v1-exp", "lisa-13b-v1", "lisa-13b-v1-exp", "lisa++"]:
        model = LisaModelEmbedInput.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            seg_token_idx=seg_token_idx,
            out_dim=cfg.out_dim,
            ce_loss_weight=cfg.ce_loss_weight,
            dice_loss_weight=cfg.dice_loss_weight,
            bce_loss_weight=cfg.bce_loss_weight,
            vision_tower=cfg.vision_tower,
            use_mm_start_end=cfg.use_mm_start_end,
            num_image_tokens=cfg.num_image_tokens,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            attn_implementation=cfg.get("attn_implementation", None)
        )
        model.num_image_tokens = cfg.num_image_tokens
        model.insert_tokens = cfg.num_image_tokens - 1

        configure_special_tokens(model, tokenizer)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
        initialize_vision_modules(model, torch_dtype)
        
        freeze_layers(model)

        model.resize_token_embeddings(len(tokenizer))


    elif cfg.name in ["lisa-13b-v0", "lisa-13b-v0-exp"]:
        model = LisaModelEmbedInputv0(
            seg_token_idx=seg_token_idx,
            tokenizer=tokenizer,
            llm_version=cfg.pretrained_model_name_or_path,
            lora_r=cfg.lora_r,
            precision=cfg.precision,
            device=device,
        )
        model.num_image_tokens = cfg.num_image_tokens
        model.insert_tokens = 0
        user_name, model_name = cfg.pretrained_model_name_or_path.split("/")
        cache_dir = "{}/.cache/huggingface/hub/models--{}--{}".format(os.environ['HOME'], user_name, model_name)
        model1_dir = glob.glob("{}/snapshots/*/pytorch_model-visual_model.bin".format(cache_dir))
        model2_dir = glob.glob("{}/snapshots/*/pytorch_model-text_hidden_fcs.bin".format(cache_dir))
        model1_dir = ["/".join(x.split("/")[:-1]) for x in model1_dir]
        model2_dir = ["/".join(x.split("/")[:-1]) for x in model2_dir]
        model_dir = list(set(model1_dir).intersection(set(model2_dir)))[0]
        weight = {}
        visual_model_weight = torch.load(
            os.path.join(model_dir, "pytorch_model-visual_model.bin")
        )
        text_hidden_fcs_weight = torch.load(
            os.path.join(model_dir, "pytorch_model-text_hidden_fcs.bin")
        )
        weight.update(visual_model_weight)
        weight.update(text_hidden_fcs_weight)
        model.load_state_dict(weight, strict=False)


    elif "gsva" in cfg.name:
        rej_token_idx = tokenizer("[REJ]", add_special_tokens=False).input_ids[0]

        model = LisaGSVAEmbedInput.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            segmentation_model_path=cfg.segmentation_model_path,
            train_mask_decoder=cfg.train_mask_decoder,
            seg_token_idx=seg_token_idx,
            rej_token_idx=rej_token_idx,
            out_dim=cfg.out_dim,
            ce_loss_weight=cfg.ce_loss_weight,
            dice_loss_weight=cfg.dice_loss_weight,
            bce_loss_weight=cfg.bce_loss_weight,
            vision_tower=cfg.vision_tower,
            use_mm_start_end=cfg.use_mm_start_end,
            num_image_tokens=cfg.num_image_tokens,
            torch_dtype=torch_dtype, 
            tokenizer=tokenizer,
        )
        model.num_image_tokens = cfg.num_image_tokens
        model.insert_tokens = cfg.num_image_tokens - 1

        configure_special_tokens(model, tokenizer)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
        initialize_vision_modules(model, torch_dtype)

        model.get_model().init_seg_and_proj(model.get_model().config)
        
        freeze_layers(model)

        model = lora_setup(model, cfg)

        model.resize_token_embeddings(len(tokenizer))

        state_dict = torch.load(cfg.weight, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    else:
        raise ValueError(f"Model {cfg.name} not found")

    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    return model


def configure_special_tokens(model: Any, tokenizer: AutoTokenizer) -> None:
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    

def initialize_vision_modules(model: Any, torch_dtype: torch.dtype) -> None:
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    
    for param in vision_tower.parameters():
        param.requires_grad = False
        
        
def freeze_layers(model: Any) -> None:
    for param in model.get_model().mm_projector.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False


def lora_setup(model: Any, cfg: Any) -> None:
    def find_linear_layers(model, lora_target_modules):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (
                isinstance(module, cls)
                and all(
                    [
                        x not in name
                        for x in [
                            "visual_model",
                            "vision_tower",
                            "mm_projector",
                            "text_hidden_fcs",
                        ]
                    ]
                )
                and any([x in name for x in lora_target_modules])
            ):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))
    lora_target_modules = find_linear_layers(model, cfg.lora_target_modules.split(","))
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model