from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from lisa.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from lisa.model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from .new_llama import LlavaLlamaForCausalLMInputEmbed
from lisa.model.segment_anything import build_sam_vit_h

from lisa.model.LISA import LisaModel, sigmoid_ce_loss, dice_loss

class LisaModelEmbedInput(LlavaLlamaForCausalLMInputEmbed):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        print(kwargs)
        print(config)
        # if not hasattr(config, "train_mask_decoder"):
        if True:
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", config.vision_tower
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.num_image_tokens = kwargs.pop("num_image_tokens")  # upgrade for LISA++

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inputs_embeds = None,
        inference: bool = False,
        **kwargs,
    ):
        assert inference == True, 'model should be in inference mode'
        
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], self.num_image_tokens - 1)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        length = input_ids.shape[0]
        assert images_clip.shape[0] == 1
        images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

        output_hidden_states = []
        output = super().forward(
            images=images_clip_extend,
            attention_mask=attention_masks,
            inputs_embeds=inputs_embeds,  # need embeddings to be optimized!       
            input_ids=input_ids,
            output_hidden_states=True,
        )
        output_hidden_states.append(output.hidden_states)
        torch.cuda.empty_cache()

        output_hidden_states_list = []
        output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
        output_hidden_states_list.append(output_hidden_states_level)
        output_hidden_states = output_hidden_states_list

        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            if pred_mask[:, 0].shape[0] == 0:
                pred_masks.append(torch.zeros(pred_mask[:, 0].shape[1:]).unsqueeze(0).cuda())
                continue
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            # assert (
            #     gt_mask.shape[0] == pred_mask.shape[0]
            # ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
            #     gt_mask.shape, pred_mask.shape
            # )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        return dict(
            mask_loss=mask_loss,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            hidden_states=output_hidden_states,
            logits=output.logits,
        )
