import torch
import torch.nn.functional as F
import typing as t
from utils import (
    initialize_log_coeffs, 
    cut_original_context,
    select_tokens_for_attack,
    bert_score, 
    log_perplexity, 
    cosine_sim,
    img_loss, 
    dice_loss,
    calculate_iou
)
from inference import inference
from abc import ABC, abstractmethod
from easydict import EasyDict as edict
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from fairseq2.nn.incremental_state import IncrementalStateBag
from types import MethodType
import numpy as np
import json
import re
from lisa.utils.utils import dict_to_cuda
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import spacy_alignments as tokenizations


class AttackerBase(ABC):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        self.kwargs = edict(kwargs)
        self.tokenizer = tokenizer
        self.model = model
        self._post_init()
    
    
    def _post_init(self):        
        with torch.no_grad():
            self.embeddings = self.model.get_input_embeddings().weight.float()[:self.tokenizer.vocab_size] # without image tokens
    
    def _init_output(self, sample) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        # Crutch FIX of the cases where there are not only 0 and 1
        gen_output.gt_mask[gen_output.gt_mask != 1] = 0
        
        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=False
        )
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        return gen_output
    
    def loss(self, inf_output, gen_output, probs):
        # Segmentation loss
        if torch.all(inf_output['pred_masks'][0] == 0) or torch.all(torch.isnan(inf_output['pred_masks'][0])): 
            adv_loss = torch.tensor(0.0, device=self.kwargs.device, requires_grad=True)
        else:
            adv_loss = self.kwargs.dice_weight * dice_loss(inf_output['pred_masks'][0], gen_output.gt_mask) + \
                self.kwargs.bce_weight * img_loss(inf_output['pred_masks'][0], gen_output.gt_mask)
        
        # Token Similarity constraint
        sim_loss = self.kwargs.sim_weight * (1 - bert_score(                
            gen_output.orig_output['hidden_states'][self.kwargs.bert_layer_num],    
            inf_output['hidden_states'][self.kwargs.bert_layer_num],
            weights=self.kwargs.ref_weights
        ).mean())

        # Perplexity constraint
        pred = inf_output['logits'][0, \
                gen_output.start_adv_token + self.model.insert_tokens : \
                gen_output.start_adv_token + self.model.insert_tokens + probs.shape[1]].unsqueeze(0)    
        perp_loss = self.kwargs.perp_weight * log_perplexity(pred, probs)
            
        return adv_loss + sim_loss + perp_loss
    
    @property
    def num_iters(self):
        return self.kwargs.num_iters
    
    @property
    def num_samples(self):
        return self.kwargs.num_samples
    
    @property
    def prompt_handling(self):
        return self.kwargs.prompt_handling
    
    @property
    def crop_attack(self):
        return self.kwargs.crop_attack
    
    @property
    def null_mask(self):
        return self.kwargs.null_mask
    
    @property
    def lr(self):
        return self.kwargs.lr
    
    def __len__(self):
        return self.num_iters
    
    @abstractmethod
    def __call__(self, sample) -> t.Generator:
        pass
    
    @abstractmethod
    def generate(self, gen_output) -> list[str]:
        pass
    

class AttackerGumbel(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        
    def _init_output(self, sample) -> edict:
        prerproc = super()._init_output(sample)
        
        prerproc.log_coeffs = initialize_log_coeffs(prerproc.input_ids, self.tokenizer.vocab_size)  # without image tokens
        return prerproc
        
    def input_embeds(self, gen_output, with_probs=False, hard=False):
        probs = F.gumbel_softmax(gen_output.log_coeffs, hard=hard, tau=self.temperature) # B x T x V, random sampling
        inputs_embeds = probs @ self.embeddings
        if with_probs:
            return inputs_embeds, probs
        return inputs_embeds
    
    def __call__(self, 
                 sample) -> t.Generator[edict, torch.Any, None]:
        
        gen_output = self._init_output(sample)
        gen_output.log_coeffs.requires_grad = True
        
        optimizer = torch.optim.Adam([gen_output.log_coeffs], lr=self.lr)
        
        for _ in range(self.num_iters):
            optimizer.zero_grad()

            inputs_embeds, probs = self.input_embeds(gen_output, with_probs=True)

            output = inference(
                gen_output.sample,
                self.model, 
                gen_output.original_context,
                adv_embeds=inputs_embeds.squeeze(0),
                adv_ids=probs.argmax(dim=-1).squeeze(0),
                device=self.kwargs.device
            )
            
            total_loss = self.loss(output, gen_output, probs)
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(gen_output.log_coeffs, max_norm=1.0)

            optimizer.step()
            
            yield gen_output
            
        return None
    
    def generate(self, gen_output):
        
        def _sample():
            probs = F.gumbel_softmax(gen_output.log_coeffs, hard=True, tau=self.temperature)
            adv_ids = probs.argmax(dim=-1).squeeze(0).cpu().tolist()
            adv_text = self.tokenizer.decode(adv_ids)
            return adv_text
        
        with torch.no_grad():
            return [_sample() for _ in range(self.num_samples)]
        
    @property
    def temperature(self):
        return self.kwargs.temperature


class AttackerGumbelOriginal(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        
    def _init_output(self, sample) -> edict:
        prerproc = super()._init_output(sample)
        
        prerproc.log_coeffs = initialize_log_coeffs(prerproc.input_ids, self.tokenizer.vocab_size)  # without image tokens
        return prerproc
        
    def input_embeds(self, gen_output, with_probs=False, hard=False):
        probs = F.gumbel_softmax(gen_output.log_coeffs, hard=hard, tau=self.temperature) # B x T x V, random sampling
        inputs_embeds = probs @ self.embeddings
        if with_probs:
            return inputs_embeds, probs
        return inputs_embeds
    
    def __call__(self, 
                 sample) -> t.Generator[edict, torch.Any, None]:
        
        gen_output = self._init_output(sample)
        gen_output.log_coeffs.requires_grad = True
        
        optimizer = torch.optim.Adam([gen_output.log_coeffs], lr=self.lr)
        
        for _ in range(self.num_iters):
            optimizer.zero_grad()

            inputs_embeds, probs = self.input_embeds(gen_output, with_probs=True)

            output = inference(
                gen_output.sample,
                self.model, 
                gen_output.original_context,
                adv_embeds=inputs_embeds.squeeze(0),
                adv_ids=probs.argmax(dim=-1).squeeze(0),
                device=self.kwargs.device
            )
            
            total_loss = self.loss(output, gen_output, probs)
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(gen_output.log_coeffs, max_norm=1.0)

            optimizer.step()

        yield gen_output
            
        return None
    
    def generate(self, gen_output):
        
        def _sample():
            probs = F.gumbel_softmax(gen_output.log_coeffs, hard=True, tau=self.temperature)
            adv_ids = probs.argmax(dim=-1).squeeze(0).cpu().tolist()
            adv_text = self.tokenizer.decode(adv_ids)
            return adv_text
        
        with torch.no_grad():
            return [_sample() for _ in range(self.num_samples)]
        
    @property
    def temperature(self):
        return self.kwargs.temperature


class AttackerARCA(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.accept_toks = self.filter_toks(self.tokenizer, n_total_toks=self.tokenizer.vocab_size)
        self.accept_toks = torch.tensor(self.accept_toks, device=self.kwargs.device)
        
    def _init_output(self, sample) -> edict:
        return super()._init_output(sample)
        
    def input_embeds(self, gen_output, with_probs=False, hard=False):
        probs = F.gumbel_softmax(gen_output.log_coeffs, hard=hard, tau=self.temperature) # B x T x V, random sampling
        inputs_embeds = probs @ self.embeddings
        if with_probs:
            return inputs_embeds, probs
        return inputs_embeds

    @staticmethod
    def filter_toks(tokenizer, n_total_toks=50257):
        import re
        english_pattern = re.compile(r'^[a-zA-Z\s\.,!?\'\"-]+$')
        toks = []
        for i in range(n_total_toks):
            token_str = tokenizer.decode([i])
            if english_pattern.match(token_str):
                toks.append(i)
        return np.array(toks)
    
    def __call__(self, 
                 sample) -> t.Generator[edict, torch.Any, None]:
        from tqdm import tqdm
        gen_output = self._init_output(sample)
        gen_output.gt_mask = gen_output.gt_mask.repeat(self.top_k, 1, 1)
        
        input_ids = gen_output.input_ids.repeat(self.top_k, 1)
        length = gen_output.input_ids.shape[0]

        input_embeds = torch.zeros(self.top_k, length, self.embeddings.shape[1], device=self.kwargs.device)
        for i in range(length):
            input_embeds[:, i] = self.embeddings[input_ids[:, i]]
        
        for _ in tqdm(range(self.num_iters)):
            for tok_id in range(length):
                input_embeds = input_embeds.detach()
                new_indices = self.accept_toks[torch.randint(0, len(self.accept_toks), (self.top_k,), device=input_ids.device)]
                input_embeds[:, tok_id, :] = self.embeddings[new_indices, :] 
                input_ids[:, tok_id] = new_indices

                if input_embeds.requires_grad:
                    input_embeds.grad.zero_()
                input_embeds.requires_grad = True
                input_embeds.retain_grad()
                
                output = self.inference(
                    gen_output.sample,
                    self.model, 
                    gen_output.original_context,
                    adv_embeds=input_embeds,
                    adv_ids=input_ids,
                    device=self.kwargs.device
                )
                probs = F.one_hot(input_ids, num_classes=output['logits'].shape[-1])

                loss = self.loss(output, gen_output, probs).mean()
                loss.backward(retain_graph=True)
                
                scores = torch.matmul(self.embeddings[self.accept_toks], input_embeds.grad[:, tok_id, :].mean(dim=0))
                sorted_toks = self.accept_toks[scores.argsort()]

                with torch.no_grad():
                    input_embeds[:, tok_id, :] = self.embeddings[sorted_toks[:self.top_k], :]                
                    input_ids[:, tok_id] = sorted_toks[:self.top_k]
                    
                    output = self.inference(
                        gen_output.sample,
                        self.model, 
                        gen_output.original_context,
                        adv_embeds=input_embeds,
                        adv_ids=input_ids,
                        device=self.kwargs.device
                    )
                    loss = self.loss(output, gen_output, probs)
                    best_idx = sorted_toks[loss.argmin()]
                    
                    input_ids[:, tok_id] = best_idx
                    input_embeds[:, tok_id, :] = self.embeddings[best_idx].repeat(self.top_k, 1)
                    gen_output.input_ids[tok_id] = best_idx
                    print(gen_output.input_ids)

            yield gen_output
            
        return None
    
    def loss(self, inf_output, gen_output, probs):
        adv_loss = self.kwargs.dice_weight * self.dice_loss(torch.stack(inf_output['pred_masks']).squeeze(1), gen_output.gt_mask) + \
            self.kwargs.bce_weight * self.img_loss(torch.stack(inf_output['pred_masks']).squeeze(1), gen_output.gt_mask)
        
        sim_loss = self.kwargs.sim_weight * (1 - bert_score(                
            gen_output.orig_output['hidden_states'][self.kwargs.bert_layer_num],    
            inf_output['hidden_states'][self.kwargs.bert_layer_num],
            weights=self.kwargs.ref_weights
        ))
        pred = inf_output['logits'][:, 
            gen_output.start_adv_token + self.model.insert_tokens : 
            gen_output.start_adv_token + self.model.insert_tokens + probs.shape[1]]
        perp_loss = self.kwargs.perp_weight * self.log_perplexity(pred, probs)
            
        return adv_loss + sim_loss + perp_loss
    
    @staticmethod
    def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        scale=1000,
        eps=1e-6,
        ):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        numerator = 2 * (inputs / scale * targets).sum(-1)
        denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        return loss
    
    @staticmethod
    def img_loss(inputs: torch.Tensor, targets: torch.Tensor):
        loss = F.binary_cross_entropy(inputs.sigmoid(), targets.to(inputs.device), reduction='none')
        return loss.sum(dim=(1, 2)) / (inputs.shape[1] * inputs.shape[2])

    @staticmethod
    def log_perplexity(logits, coeffs):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_coeffs = coeffs[:, 1:, :].contiguous()
        shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
        return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean(-1)

    def inference(
        self,
        sample: dict, 
        model: torch.nn.Module, 
        original_context: dict = None, 
        adv_embeds: torch.Tensor = None, 
        adv_ids: torch.Tensor = None,
        device='cuda'
        ) -> dict:
        sample = dict_to_cuda(sample, device=self.kwargs.device)
        
        new_input_ids_seqs = []
        for i in range(len(adv_ids)):
            new_input_ids_seqs.append(torch.cat([
                original_context['prefix_ids'],
                torch.tensor(adv_ids[i], dtype=torch.long, device=device),
                original_context['suffix_ids']
            ], dim=0).int())
        input_ids = pad_sequence(new_input_ids_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id).long()

        new_attention_masks_seqs = []
        for i in range(len(adv_ids)):
            new_attention_masks_seqs.append(torch.cat([
                original_context['prefix_attn'],
                torch.ones(len(adv_ids[i]), dtype=torch.bool, device=device),
                original_context['suffix_attn']
            ], dim=0).bool())
        attention_masks = pad_sequence(new_attention_masks_seqs, batch_first=True, padding_value=False).bool()

        new_input_embeds_seqs = []
        for i in range(len(adv_embeds)):
            new_input_embeds_seqs.append(torch.cat([
                original_context['prefix_embeds'],
                adv_embeds[i].to(device),
                original_context['suffix_embeds']
            ], dim=0))
        inputs_embeds = pad_sequence(new_input_embeds_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id).half()
        
        with torch.amp.autocast('cuda'):
            batch_size = input_ids.shape[0]
            output_dict = model(
                images=sample['images'].expand(batch_size, -1, -1, -1).half(),
                images_clip=sample['images_clip'].half(),
                input_ids=input_ids,
                attention_masks=attention_masks,
                inputs_embeds=inputs_embeds,
                masks_list=sample['masks_list'] * batch_size,
                label_list=sample['label_list'] * batch_size,
                resize_list=sample['resize_list'] * batch_size,
                labels = self.adjust_tensor_size(
                    sample['labels'], 
                    input_ids.size()[1]
                ),
                offset=torch.arange(batch_size + 1, device=device),
                inference=True
            )
        return output_dict

    @staticmethod
    def adjust_tensor_size(
        tensor: torch.Tensor, 
        target_size: int, 
        pad_value: int = -100
        ) -> torch.Tensor:
        current_size = tensor.size(-1)
        if current_size > target_size:
            truncated_tensor = tensor[:, -target_size:]
        else:
            padding_size = target_size - current_size
            truncated_tensor = torch.nn.functional.pad(tensor, (padding_size, 0), value=pad_value)
        return truncated_tensor
    
    def generate(self, gen_output):
        return [self.tokenizer.decode(gen_output.input_ids)]

    @property
    def top_k(self):
        return self.kwargs.top_k
        
    @property
    def temperature(self):
        return self.kwargs.temperature


class AttackerSonar(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()


    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()
    

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))


    def _init_output(self, sample, crop_attack: bool = False, first_part: bool = True) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        # Crutch FIX of the cases where there are not only 0 and 1
        gen_output.gt_mask[gen_output.gt_mask != 1] = 0

        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=crop_attack,
            first_part=first_part
        )
        
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        return gen_output
            
    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:

        gen_outputs = [
            self._init_output(sample, crop_attack=False, first_part=False),
            self._init_output(sample, crop_attack=True, first_part=True),
            self._init_output(sample, crop_attack=True, first_part=False),
        ]
        try:
            for gen_output in gen_outputs:

                with torch.no_grad():
                    gen_output.embedding = self.sonar_t2v.predict(
                        [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
                    )

                gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

                optimizer = torch.optim.Adam([gen_output.embedding], lr=self.lr)
                
                for _ in range(self.num_iters):
                    optimizer.zero_grad()

                    gen_output.text, sonar_probs = self.sonar_v2t.predict(
                        gen_output.embedding, 
                        target_lang="eng_Latn", 
                        max_seq_len=gen_output.max_adv_len
                    )
                    sonar_probs = sonar_probs.to(self.kwargs.device)[:-1]   # remove <eos>

                    lisa_probs = self.transform_probs(sonar_probs)

                    input_embeds = lisa_probs @ self.embeddings

                    output = inference(
                        gen_output.sample,
                        self.model, 
                        gen_output.original_context,
                        adv_embeds=input_embeds,
                        adv_ids=lisa_probs.argmax(dim=-1),
                        device=self.kwargs.device
                    )
                    
                    total_loss = self.loss(output, gen_output, lisa_probs.unsqueeze(0))
                    total_loss.backward()

                    optimizer.step()
                    
                    yield gen_output
        except Exception as e:
            print(gen_output.text)
            print(e)

        return None

    def transform_probs(self, sonar_probs):
        lisa_probs = torch.zeros(sonar_probs.shape[0], self.tokenizer.vocab_size, device=self.kwargs.device)

        start_tokens = []
        for word in str(self.sonar_decoder(sonar_probs.argmax(dim=1))).split():
            start_tokens.append(self.sonar_tokenizer(word)[1])
        
        lisa_start = 0
        zero_padding = {}
        
        for s_token in sonar_probs.argmax(dim=1):
            subword = self.sonar_decoder(s_token.unsqueeze(0))
            
            if s_token in start_tokens:
                l_tokens = self.tokenizer.encode(str(subword), add_special_tokens=False)
            elif str(subword) == '!':
                l_tokens = self.tokenizer.encode('_' + str(subword), add_special_tokens=False)[1:]
            else:
                l_tokens = self.tokenizer.encode('!' + str(subword), add_special_tokens=False)[1:]
            
            if len(l_tokens) == 0:
                lisa_probs[..., 1] += sonar_probs[..., s_token.item()]
            else:
                lisa_probs[..., l_tokens[0]] += sonar_probs[..., s_token.item()]
                    
            if len(l_tokens) > 1:
                for j, add_token in enumerate(l_tokens[1:]):
                    zero_padding[lisa_start + 1 + j] = \
                        torch.nn.functional.one_hot(
                            torch.tensor(add_token, device=self.kwargs.device), num_classes=self.embeddings.shape[0]
                        ).float().unsqueeze(1).T 

            lisa_start += len(l_tokens)
    
        if len(zero_padding):
            for idx, vs in zero_padding.items():
                left = lisa_probs[:idx] 
                right = lisa_probs[idx:]
                lisa_probs = torch.cat((left, vs, right), dim=0)

        return lisa_probs

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text[0]]


class AttackerSonarPerplexity(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.target_text_encoder = self.sonar_v2t.tokenizer.create_encoder(task='translation', lang="eng_Latn", mode='target')
        self.sonar_v2t.model.decoder.decoder_frontend.embed.forward = MethodType(
            lambda _self, x: x @ _self.weight,
            self.sonar_v2t.model.decoder.decoder_frontend.embed
        )
        self.prefill_len = 2
        self.max_seq_len = 100
        self.min_seq_len = 1
        self.unk_penalty = 100
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()

        self.sonar_to_lisa_transfer_matrix = torch.load(f'/home/jovyan/zinkovich/ref-seg-text-break/notebooks/meta_tokenize/sparse_Ts2l.pt', map_location=self.device)

        self.sonar_perplexity = []
        self.total_loss = []
        self.adv_loss = []


    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))


    def _init_output(self, sample, crop_attack: bool = False, first_part: bool = True) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        # Crutch FIX of the cases where there are not only 0 and 1
        gen_output.gt_mask[gen_output.gt_mask != 1] = 0

        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=crop_attack,
            first_part=first_part
        )
        
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        return gen_output

    def sonar_predict(self, vector):
        
        seqs = torch.nn.functional.one_hot(self.target_text_encoder.prefix_indices, num_classes=self.sonar_v2t.model.target_vocab_info.size).float().cuda()
        seqs = seqs.unsqueeze(0)
        encoder_output, encoder_padding_mask = self.sonar_v2t.model.encode(
            vector, None
        )
        is_eos = False
        while not is_eos:
            decoder_output, decoder_padding_mask = self.sonar_v2t.model.decode(
                seqs,
                None,
                encoder_output,
                encoder_padding_mask,
            )
            model_output = self.sonar_v2t.model.project(decoder_output, decoder_padding_mask)
            logits = model_output.logits[:, -1, :]
            if self.kwargs.get('gumbel_temperature', 0.0) > 0.0:
                probs = torch.nn.functional.gumbel_softmax(logits, tau=self.kwargs.gumbel_temperature, hard=False, dim=-1)
            else:
                lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
                probs = torch.exp(lprobs)

            if probs.argmax(dim=-1) == self.sonar_v2t.model.target_vocab_info.eos_idx:
                is_eos = True
            else:
                seqs = torch.cat([seqs, probs.unsqueeze(1)], dim=1)

            if seqs.size(1) == self.max_seq_len:
                break

        return seqs[:, len(self.target_text_encoder.prefix_indices):]
    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:

        gen_outputs = [
            self._init_output(sample, crop_attack=False, first_part=False),
            # self._init_output(sample, crop_attack=True, first_part=True),
            # self._init_output(sample, crop_attack=True, first_part=False),
        ]
        try:
            import torch
            torch.autograd.set_detect_anomaly(True)
            for gen_output in gen_outputs:

                with torch.no_grad():
                    gen_output.embedding = self.sonar_t2v.predict(
                        [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
                    )

                gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

                optimizer = torch.optim.Adam([gen_output.embedding], lr=self.lr)
                
                for _ in range(self.num_iters):
                    optimizer.zero_grad()

                    sonar_probs = self.sonar_predict(gen_output.embedding)
                    sonar_ids = sonar_probs.argmax(dim=-1).squeeze(0)

                    gen_output.text = str(self.sonar_decoder(sonar_ids))

                    sonar_probs = sonar_probs.to(self.kwargs.device).squeeze(0)
                    # sonar_ids = sonar_ids
                    lisa_probs = self.transform_probs(sonar_probs, sonar_ids, gen_output.text)
                    print('lisa gets this text: ', self.tokenizer.decode(lisa_probs.argmax(-1).squeeze()))
                    input_embeds = lisa_probs @ self.embeddings

                    output = inference(
                        gen_output.sample,
                        self.model, 
                        gen_output.original_context,
                        adv_embeds=input_embeds,
                        adv_ids=lisa_probs.argmax(dim=-1),
                        device=self.kwargs.device
                    )
                    
                    total_loss = self.loss(output, gen_output, lisa_probs.unsqueeze(0), sonar_probs)
                    total_loss.backward()

                    optimizer.step()
                    
                    yield gen_output
        except Exception as e:
            raise e
            print(gen_output.text)
            print(e)

        for loss_name, loss_values in zip(
            ['Sonar Perplexity', 'Total Loss', 'Adv Loss'],
            [self.sonar_perplexity, self.total_loss, self.adv_loss]
        ):
            plt.plot(torch.arange(len(loss_values)), loss_values, label=loss_name)
            plt.plot(torch.arange(len(loss_values))[5:-4], np.convolve(loss_values, np.ones(10)/10, mode='valid'), label=f'Smoothed {loss_name}')
            plt.legend()
            plt.savefig(f'anvika_debug/{self.tokenizer.decode(gen_output.orig_output.input_ids)[:15]}_{loss_name}.png')
            plt.close()
            loss_values.clear()

        return None

    def loss(self, inf_output, gen_output, probs, sonar_probs):
        if torch.all(inf_output['pred_masks'][0] == 0) or torch.all(torch.isnan(inf_output['pred_masks'][0])): 
            adv_loss = torch.tensor(0.0, device=self.kwargs.device, requires_grad=True)
        else:
            adv_loss = self.kwargs.dice_weight * dice_loss(inf_output['pred_masks'][0], gen_output.gt_mask) + \
                self.kwargs.bce_weight * img_loss(inf_output['pred_masks'][0], gen_output.gt_mask)

        # sim_loss = self.kwargs.sim_weight * (1 - bert_score(                
        #     gen_output.orig_output['hidden_states'][self.kwargs.bert_layer_num],    
        #     inf_output['hidden_states'][self.kwargs.bert_layer_num],
        #     weights=self.kwargs.ref_weights
        # ).mean())

        # pred = inf_output['logits'][0, \
        #         gen_output.start_adv_token + self.model.insert_tokens : \
        #         gen_output.start_adv_token + self.model.insert_tokens + probs.shape[1]].unsqueeze(0)    
        # perp_loss = self.kwargs.perp_weight * log_perplexity(pred, probs)

        tokens = sonar_probs.argmax(-1)
        entropy_loss = -self.kwargs.entropy_weight * sonar_probs[np.arange(len(tokens)), tokens].log().sum()
        total_loss = adv_loss + entropy_loss # + sim_loss + perp_loss
        self.sonar_perplexity.append(entropy_loss.item())
        self.total_loss.append(total_loss.item())
        self.adv_loss.append(adv_loss.item())
            
        return total_loss

    def transform_probs(self, sonar_probs, sonar_ids, text):
        sonar_probs = sonar_probs.max(dim=-1).values

        lisa_tokens = self.tokenizer.tokenize(text)
        lisa_ids = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids
        lisa_ids = lisa_ids.cuda().squeeze(0)
        sonar_tokens = [self.sonar_t2v.tokenizer.model.index_to_token(sonar_id) for sonar_id in sonar_ids]

        a2b, _ = tokenizations.get_alignments(lisa_tokens, sonar_tokens)

        pre_lisa_probs = torch.empty(0, self.tokenizer.vocab_size, device=self.kwargs.device)

        for i in range(len(a2b)):
            for j in range(len(a2b[i])):
                pre_lisa_probs = torchsonar_probs[a2b[i][j]]
            pre_lisa_probs[i, :] /= len(a2b[i])
        # print('pre matrix multiplication', pre_lisa_probs)
        
        lisa_soft_probs = pre_lisa_probs @ self.sonar_to_lisa_transfer_matrix

        # print('post matrix multiplication', lisa_probs)

        lisa_soft_probs = lisa_soft_probs / (lisa_soft_probs.sum(dim=-1, keepdim=True) + 1e-10)

        lisa_probs = torch.nn.functional.one_hot(lisa_ids, num_classes=self.tokenizer.vocab_size) + lisa_soft_probs - lisa_soft_probs.detach()

        print('lisa probs', lisa_probs)


        return lisa_probs
    

    def transform_probs_transfer_matrix(self, sonar_probs, sonar_ids, text):

        lisa_tokens = self.tokenizer.tokenize(text)
        lisa_ids = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids
        lisa_ids = lisa_ids.cuda().squeeze(0)
        print('lisa ids', lisa_ids)
        sonar_tokens = [self.sonar_t2v.tokenizer.model.index_to_token(sonar_id) for sonar_id in sonar_ids]
        # print('lisa tokens', lisa_tokens)
        # print('sonar tokens', sonar_tokens)
        a2b, _ = tokenizations.get_alignments(lisa_tokens, sonar_tokens)
        # print('alignment', a2b)

        pre_lisa_probs = torch.zeros(len(lisa_tokens), sonar_probs.shape[1], device=self.kwargs.device)

        for i in range(len(a2b)):
            for j in range(len(a2b[i])):
                pre_lisa_probs[i, :] += sonar_probs[a2b[i][j], :]
            pre_lisa_probs[i, :] /= len(a2b[i])
        # print('pre matrix multiplication', pre_lisa_probs)
        
        lisa_soft_probs = pre_lisa_probs @ self.sonar_to_lisa_transfer_matrix

        # print('post matrix multiplication', lisa_probs)

        lisa_soft_probs = lisa_soft_probs / (lisa_soft_probs.sum(dim=-1, keepdim=True) + 1e-10)

        lisa_probs = torch.nn.functional.one_hot(lisa_ids, num_classes=self.tokenizer.vocab_size) + lisa_soft_probs - lisa_soft_probs.detach()

        print('lisa probs', lisa_probs)


        return lisa_probs

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text]
    
    # def inference(
    #     self,
    #     sample: dict, 
    #     model: torch.nn.Module, 
    #     original_context: dict = None, 
    #     adv_embeds: torch.Tensor = None, 
    #     device='cuda'
    #     ) -> dict:
    #     """
    #     Perform inference with adversarial embeddings or adversarial IDs.
    #     """
    #     sample = dict_to_cuda(sample, device=self.kwargs.device)

    #     new_input_ids_seqs = []
    #     for i in range(len(adv_ids)):
    #         new_input_ids_seqs.append(torch.cat([
    #             original_context['prefix_ids'],
    #             torch.tensor(adv_ids[i], dtype=torch.long, device=device),
    #             original_context['suffix_ids']
    #         ], dim=0).int())
    #     input_ids = pad_sequence(new_input_ids_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id).long()

    #     new_attention_masks_seqs = []
    #     for i in range(len(adv_ids)):
    #         new_attention_masks_seqs.append(torch.cat([
    #             original_context['prefix_attn'],
    #             torch.ones(len(adv_ids[i]), dtype=torch.bool, device=device),
    #             original_context['suffix_attn']
    #         ], dim=0).bool())

    #     with torch.amp.autocast('cuda'):
    #         output_dict = model(
    #             images=sample['images'].expand(self.kwargs.sample_size, -1, -1, -1).half(),
    #             images_clip=sample['images_clip'].half(),
    #             input_ids = input_ids,
    #             attention_masks = pad_sequence(new_attention_masks_seqs, batch_first=True, padding_value=False).bool(),
    #             masks_list = sample['masks_list'] * self.kwargs.sample_size,
    #             label_list = sample['label_list'] * self.kwargs.sample_size,
    #             resize_list = sample['resize_list'] * self.kwargs.sample_size,
    #             labels = AttackerSonarReinforceWithBaseline.adjust_tensor_size(
    #                 sample['labels'], 
    #                 input_ids.size()[1]
    #             ),
    #             offset = torch.arange(self.kwargs.sample_size + 1, device=device),
    #             inference = True
    #         )
    #     return output_dict


class AttackerSonarPerplexityNewMT(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.target_text_encoder = self.sonar_v2t.tokenizer.create_encoder(task='translation', lang="eng_Latn", mode='target')
        self.sonar_v2t.model.decoder.decoder_frontend.embed.forward = MethodType(
            lambda self, x: x @ self.weight,
            self.sonar_v2t.model.decoder.decoder_frontend.embed
        )
        self.prefill_len = 2
        self.max_seq_len = 100
        self.min_seq_len = 1
        self.unk_penalty = 100
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()
        self.sonar_perplexity = []
        self.total_loss = []
        self.adv_loss = []

        self.l2s_mapping = self.get_mapping()

    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))


    def _init_output(self, sample, crop_attack: bool = False, first_part: bool = True) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        # Crutch FIX of the cases where there are not only 0 and 1
        gen_output.gt_mask[gen_output.gt_mask != 1] = 0

        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=crop_attack,
            first_part=first_part
        )
        
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        return gen_output

    def sonar_predict(self, vector):
        
        seqs = torch.nn.functional.one_hot(self.target_text_encoder.prefix_indices, num_classes=self.sonar_v2t.model.target_vocab_info.size).float().cuda()
        seqs = seqs.unsqueeze(0)
        encoder_output, encoder_padding_mask = self.sonar_v2t.model.encode(
            vector, None
        )
        is_eos = False
        while not is_eos:
            decoder_output, decoder_padding_mask = self.sonar_v2t.model.decode(
                seqs,
                None,
                encoder_output,
                encoder_padding_mask,
            )
            model_output = self.sonar_v2t.model.project(decoder_output, decoder_padding_mask)
            logits = model_output.logits[:, -1, :]
            lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
            probs = torch.exp(lprobs)
            if probs.argmax(dim=-1) == self.sonar_v2t.model.target_vocab_info.eos_idx:
                is_eos = True
            else:
                seqs = torch.cat([seqs, probs.unsqueeze(1)], dim=1)

            if seqs.size(1) == self.max_seq_len:
                break

        return seqs
    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:

        gen_outputs = [
            self._init_output(sample, crop_attack=False, first_part=False),
            # self._init_output(sample, crop_attack=True, first_part=True),
            # self._init_output(sample, crop_attack=True, first_part=False),
        ]
        try:
            import torch
            torch.autograd.set_detect_anomaly(True)
            for gen_output in gen_outputs:

                with torch.no_grad():
                    gen_output.embedding = self.sonar_t2v.predict(
                        [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
                    )

                gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

                optimizer = torch.optim.Adam([gen_output.embedding], lr=self.lr)
                
                for _ in range(self.num_iters):
                    optimizer.zero_grad()

                    sonar_probs = self.sonar_predict(gen_output.embedding)

                    gen_output.text = str(self.sonar_decoder(sonar_probs.argmax(dim=-1).squeeze(0)))

                    sonar_probs = sonar_probs.to(self.kwargs.device).squeeze(0)[:-1]   # remove <eos>

                    lisa_probs = self.transform_probs(sonar_probs)

                    input_embeds = lisa_probs @ self.embeddings

                    output = inference(
                        gen_output.sample,
                        self.model, 
                        gen_output.original_context,
                        adv_embeds=input_embeds,
                        adv_ids=lisa_probs.argmax(dim=-1),
                        device=self.kwargs.device
                    )
                    
                    total_loss = self.loss(output, gen_output, lisa_probs.unsqueeze(0), sonar_probs)
                    total_loss.backward()

                    optimizer.step()
                    
                    yield gen_output
        except Exception as e:
            print(gen_output.text)
            print(e)

        for loss_name, loss_values in zip(
            ['Sonar Perplexity', 'Total Loss', 'Adv Loss'],
            [self.sonar_perplexity, self.total_loss, self.adv_loss]
        ):
            plt.plot(torch.arange(len(loss_values)), loss_values, label=loss_name)
            plt.plot(torch.arange(len(loss_values))[5:-4], np.convolve(loss_values, np.ones(10)/10, mode='valid'), label=f'Smoothed {loss_name}')
            plt.legend()
            plt.savefig(f'anvika_debug/{self.tokenizer.decode(gen_output.orig_output.input_ids)[:15]}_{loss_name}.png')
            plt.close()
            loss_values.clear()

        return None

    def loss(self, inf_output, gen_output, probs, sonar_probs):
        if torch.all(inf_output['pred_masks'][0] == 0) or torch.all(torch.isnan(inf_output['pred_masks'][0])): 
            adv_loss = torch.tensor(0.0, device=self.kwargs.device, requires_grad=True)
        else:
            adv_loss = self.kwargs.dice_weight * dice_loss(inf_output['pred_masks'][0], gen_output.gt_mask) + \
                self.kwargs.bce_weight * img_loss(inf_output['pred_masks'][0], gen_output.gt_mask)

        sim_loss = self.kwargs.sim_weight * (1 - bert_score(                
            gen_output.orig_output['hidden_states'][self.kwargs.bert_layer_num],    
            inf_output['hidden_states'][self.kwargs.bert_layer_num],
            weights=self.kwargs.ref_weights
        ).mean())

        pred = inf_output['logits'][0, \
                gen_output.start_adv_token + self.model.insert_tokens : \
                gen_output.start_adv_token + self.model.insert_tokens + probs.shape[1]].unsqueeze(0)    
        perp_loss = self.kwargs.perp_weight * log_perplexity(pred, probs)

        tokens = sonar_probs.argmax(-1)
        entropy_loss = -self.kwargs.entropy_weight * sonar_probs[np.arange(len(tokens)), tokens].log().sum()

        sense_loss = self.kwargs.sense_weight * (1 - cosine_sim(                     
            gen_output.orig_output['hidden_states'][0].sum(dim=1),
            inf_output['hidden_states'][0].sum(dim=1)
        ))

        self.sonar_perplexity.append(entropy_loss.item())
        self.total_loss.append((adv_loss + sim_loss + perp_loss + entropy_loss).item())
        self.adv_loss.append(adv_loss.item())
            
        return adv_loss + sim_loss + perp_loss + entropy_loss + sense_loss  

    def transform_probs(self, sonar_probs):
        
        def get_tokens(probs):
            sonar_tokens = probs.argmax(dim=-1).tolist()
            text = self.sonar_decode(sonar_tokens)
            lisa_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return sonar_tokens, lisa_tokens

        sonar_tokens, lisa_tokens = get_tokens(sonar_probs)

        sonar_subwords = [self.sonar_decode([token]) for token in sonar_tokens]
        lisa_subwords = [self.tokenizer.decode([token]) for token in lisa_tokens]

        deleted, inserted, mapping = self.meta_tokenize(sonar_subwords, lisa_subwords, lisa_tokens)

        mask = torch.ones(len(sonar_tokens), dtype=torch.bool)
        mask[deleted] = False
        sonar_probs_filtered = sonar_probs[mask]

        lisa_probs = torch.zeros(len(sonar_probs_filtered), self.tokenizer.vocab_size, device=self.kwargs.device)

        for _, pair in self.l2s_mapping.items():
            lisa_probs[..., pair['lisa_token']] = sonar_probs_filtered[..., pair['sonar_token']]

        for l_idx, s_idx in mapping.items():
            try:
                lisa_probs[..., lisa_tokens[l_idx]] = sonar_probs_filtered[..., sonar_tokens[s_idx]]
            except Exception as e:
                print(f"An error occurred: {e}")
                breakpoint()

        lisa_probs = self.insert_onehots(lisa_probs, inserted)

        # print('sonar text: \t', self.sonar_decode(sonar_probs.argmax(dim=-1)))
        # print('lisa text: \t', self.tokenizer.decode(lisa_probs.argmax(dim=-1).tolist()))

        return lisa_probs


    def meta_tokenize(self, sonar_subwords, lisa_subwords, lisa_tokens):
        s_idx, l_idx = 0, 0
        lisa_insert = {}
        sonar_delete = []
        mapping = {}

        while s_idx != len(sonar_subwords) or l_idx != len(lisa_subwords):
    
            if sonar_subwords[s_idx] == '':
                mapping[l_idx] = s_idx
                sonar_delete.append(s_idx)
                s_idx += 1
                continue

            mapping[l_idx] = s_idx

            s_subword = sonar_subwords[s_idx]
            l_subword = lisa_subwords[l_idx]
            
            while len(s_subword) != len(l_subword):
                
                if len(s_subword) < len(l_subword):
                    s_idx += 1
                    s_subword += sonar_subwords[s_idx]
                    sonar_delete.append(s_idx)
                else:
                    l_idx += 1
                    l_subword += lisa_subwords[l_idx]
                    lisa_insert[l_idx] = lisa_tokens[l_idx]

            l_idx += 1
            s_idx += 1

        return sonar_delete, lisa_insert, mapping


    def insert_onehots(self, probs, onehots):
        for idx, value in onehots.items():
            left = probs[:idx] 
            right = probs[idx:]
            probs = torch.cat((
                left, 
                torch.nn.functional.one_hot(
                    torch.tensor(value, device=self.device), num_classes=self.embeddings.shape[0]
                ).float().unsqueeze(1).T, 
                right
            ), dim=0)
        return probs

    
    def get_mapping(self, special_symbol='~', gap_symbol='▁'):
        import re
        def is_english_only(text):
            return bool(re.match('^[a-zA-Z]+$', text))
        
        def is_punctuation(text, punctuation = ['.', ',', '!', '?', ':', ';', '(', ')']):
            return text in punctuation

        n_special_tokens = list(np.arange(0, 260)) + [32000, 32001, 32002, 32003]
        matches = {}
        n_lisa_tokens = 0
        for lisa_subword, i in self.tokenizer.get_vocab().items():

            if i not in n_special_tokens and \
                (is_english_only(lisa_subword.replace(gap_symbol, '')) or is_punctuation(lisa_subword.replace(gap_symbol, ''))):

                if lisa_subword.startswith(gap_symbol):
                    sonar_tokens = self.sonar_tokenize(lisa_subword)
                else:
                    sonar_tokens = self.sonar_tokenize(special_symbol + lisa_subword)[1:]

                if len(sonar_tokens) == 1:
                    matches[lisa_subword] = {
                        'sonar_token': sonar_tokens[0],
                        'lisa_token' : i
                    }
                n_lisa_tokens += 1
        print(f'matches: {len(matches)} out of {n_lisa_tokens} (lisa english only), p = {len(matches)/n_lisa_tokens * 100:.0f}%')
        return matches

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text]
    

class AttackerSonarDummy(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()


    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()
    

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))


    def _init_output(self, sample, crop_attack: bool = False, first_part: bool = True) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=crop_attack,
            first_part=first_part
        )
        
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        return gen_output
            
    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:

        gen_outputs = [
            self._init_output(sample, crop_attack=False, first_part=False),
            self._init_output(sample, crop_attack=True, first_part=True),
            self._init_output(sample, crop_attack=True, first_part=False),
        ]
        try:
            for gen_output in gen_outputs:

                with torch.no_grad():
                    gen_output.embedding = self.sonar_t2v.predict(
                        [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
                    )

                gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

                optimizer = torch.optim.Adam([gen_output.embedding], lr=self.lr)
                
                for _ in range(self.num_iters):
                    optimizer.zero_grad()

                    gen_output.text, sonar_probs = self.sonar_v2t.predict(
                        gen_output.embedding, 
                        target_lang="eng_Latn", 
                        max_seq_len=gen_output.max_adv_len
                    )
                    sonar_probs = sonar_probs.to(self.kwargs.device)[:-1]   # remove <eos>

                    lisa_probs = self.transform_probs(sonar_probs)

                    input_embeds = lisa_probs @ self.embeddings

                    output = inference(
                        gen_output.sample,
                        self.model, 
                        gen_output.original_context,
                        adv_embeds=input_embeds,
                        adv_ids=lisa_probs.argmax(dim=-1),
                        device=self.kwargs.device
                    )
                    
                    total_loss = self.loss(output, gen_output, lisa_probs.unsqueeze(0))
                    total_loss.backward()

                    optimizer.step()
                    
                    yield gen_output
        except Exception as e:
            print(gen_output.text)
            print(e)

        return None

    def transform_probs(self, sonar_probs):
        lisa_probs = torch.zeros(sonar_probs.shape[0], self.tokenizer.vocab_size, device=self.kwargs.device)

        start_tokens = []
        for word in str(self.sonar_decoder(sonar_probs.argmax(dim=1))).split():
            start_tokens.append(self.sonar_tokenizer(word)[1])
        
        for s_token in sonar_probs.argmax(dim=1):
            subword = self.sonar_decoder(s_token.unsqueeze(0))
            
            if s_token in start_tokens:
                l_tokens = self.tokenizer.encode(str(subword), add_special_tokens=False)
            elif str(subword) == '!':
                l_tokens = self.tokenizer.encode('_' + str(subword), add_special_tokens=False)[1:]
            else:
                l_tokens = self.tokenizer.encode('!' + str(subword), add_special_tokens=False)[1:]
            
            if len(l_tokens) == 0:
                lisa_probs[..., 1] += sonar_probs[..., s_token.item()]
            else:
                lisa_probs[..., l_tokens[0]] += sonar_probs[..., s_token.item()]
               
        return lisa_probs

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text[0]]
    

class AttackerSonarNoise(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()


    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()
    

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))


    def _init_output(self, sample, crop_attack: bool = False, first_part: bool = True) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=crop_attack,
            first_part=first_part
        )
        
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        return gen_output
            
    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:

        gen_outputs = [
            self._init_output(sample, crop_attack=False, first_part=False),
            self._init_output(sample, crop_attack=True, first_part=True),
            self._init_output(sample, crop_attack=True, first_part=False),
        ]
        try:
            for gen_output in gen_outputs:

                with torch.no_grad():
                    gen_output.embedding = self.sonar_t2v.predict(
                        [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
                    )

                gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

                optimizer = torch.optim.Adam([gen_output.embedding], lr=self.lr)
                
                for _ in range(self.num_iters):
                    optimizer.zero_grad()

                    gen_output.text, sonar_probs = self.sonar_v2t.predict(
                        gen_output.embedding, 
                        target_lang="eng_Latn", 
                        max_seq_len=gen_output.max_adv_len
                    )
                    sonar_probs = sonar_probs.to(self.kwargs.device)[:-1]   # remove <eos>

                    lisa_probs = self.transform_probs(sonar_probs)

                    input_embeds = lisa_probs @ self.embeddings

                    output = inference(
                        gen_output.sample,
                        self.model, 
                        gen_output.original_context,
                        adv_embeds=input_embeds,
                        adv_ids=lisa_probs.argmax(dim=-1),
                        device=self.kwargs.device
                    )
                    
                    total_loss = self.loss(output, gen_output, lisa_probs.unsqueeze(0))
                    total_loss.backward()

                    optimizer.step()
                    
                    yield gen_output
        except Exception as e:
            print(gen_output.text)
            print(e)

        return None

    def transform_probs(self, sonar_probs):
        lisa_probs = torch.zeros(sonar_probs.shape[0], self.tokenizer.vocab_size, device=self.kwargs.device)

        start_tokens = []
        for word in str(self.sonar_decoder(sonar_probs.argmax(dim=1))).split():
            start_tokens.append(self.sonar_tokenizer(word)[1])
        
        lisa_start = 0
        zero_padding = {}
        
        for s_token in sonar_probs.argmax(dim=1):
            subword = self.sonar_decoder(s_token.unsqueeze(0))
            
            if s_token in start_tokens:
                l_tokens = self.tokenizer.encode(str(subword), add_special_tokens=False)
            elif str(subword) == '!':
                l_tokens = self.tokenizer.encode('_' + str(subword), add_special_tokens=False)[1:]
            else:
                l_tokens = self.tokenizer.encode('!' + str(subword), add_special_tokens=False)[1:]
            
            if len(l_tokens) == 0:
                lisa_probs[..., 1] += sonar_probs[..., s_token.item()]
            else:
                lisa_probs[..., l_tokens[0]] += sonar_probs[..., s_token.item()]
                    
            if len(l_tokens) > 1:
                for j, _ in enumerate(l_tokens[1:]):
                    zero_padding[lisa_start + 1 + j] = \
                        torch.zeros((1, self.embeddings.shape[0]), device=self.kwargs.device)

            lisa_start += len(l_tokens)
    
        if len(zero_padding):
            for idx, _ in zero_padding.items():
                left = lisa_probs[:idx] 
                right = lisa_probs[idx:]
                lisa_probs = torch.cat((left, lisa_probs.mean(dim=0).unsqueeze(0), right), dim=0)

        return lisa_probs

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text[0]]
    

class AttackerSonarCV(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()


    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()
    

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))

    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:

        gen_output = self._init_output(sample)

        with torch.no_grad():
            gen_output.embedding = self.sonar_t2v.predict(
                [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
            )

        gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([gen_output.embedding], lr=self.lr)
        
        for _ in range(self.num_iters):
            try:
                optimizer.zero_grad()

                gen_output.text, sonar_probs = self.sonar_v2t.predict(      # here is the problem, "try" here 
                    gen_output.embedding, 
                    target_lang="eng_Latn", 
                    max_seq_len=gen_output.max_adv_len
                )
                sonar_probs = sonar_probs.to(self.kwargs.device)[:-1]   # remove <eos>

                lisa_probs = self.transform_probs(sonar_probs)

                input_embeds = lisa_probs @ self.embeddings

                output = inference(
                    gen_output.sample,
                    self.model, 
                    gen_output.original_context,
                    adv_embeds=input_embeds,
                    adv_ids=lisa_probs.argmax(dim=-1),
                    device=self.kwargs.device
                )
                
                total_loss = self.loss(output, gen_output, lisa_probs.unsqueeze(0))
                total_loss.backward()

                optimizer.step()

                yield gen_output

            except Exception as e:
                print(gen_output.text)
                print(e)
                break
                
        return None

    def transform_probs(self, sonar_probs):
        lisa_probs = torch.zeros(sonar_probs.shape[0], self.tokenizer.vocab_size, device=self.kwargs.device)

        start_tokens = []
        for word in str(self.sonar_decoder(sonar_probs.argmax(dim=1))).split():
            start_tokens.append(self.sonar_tokenizer(word)[1])
        
        lisa_start = 0
        zero_padding = {}
        
        for s_token in sonar_probs.argmax(dim=1):
            subword = self.sonar_decoder(s_token.unsqueeze(0))
            
            if s_token in start_tokens:
                l_tokens = self.tokenizer.encode(str(subword), add_special_tokens=False)
            elif str(subword) == '!':
                l_tokens = self.tokenizer.encode('_' + str(subword), add_special_tokens=False)[1:]
            else:
                l_tokens = self.tokenizer.encode('!' + str(subword), add_special_tokens=False)[1:]
            
            if len(l_tokens) == 0:
                lisa_probs[..., 1] += sonar_probs[..., s_token.item()]
            else:
                lisa_probs[..., l_tokens[0]] += sonar_probs[..., s_token.item()]
                    
            if len(l_tokens) > 1:
                for j, add_token in enumerate(l_tokens[1:]):
                    zero_padding[lisa_start + 1 + j] = \
                        torch.zeros((1, self.embeddings.shape[0]), device=self.kwargs.device)

            lisa_start += len(l_tokens)
    
        if len(zero_padding):
            for idx, vs in zero_padding.items():
                left = lisa_probs[:idx] 
                right = lisa_probs[idx:]
                lisa_probs = torch.cat((left, vs, right), dim=0)

        return lisa_probs

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text[0]]
    

class AttackerSonarReinforceWithBaseline(AttackerBase):
    
    def __init__(
        self,
        tokenizer,
        model,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            **kwargs,
        )
        self.device = torch.device(self.kwargs.device)
        
        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=self.device)
        
        self.sonar_tokenizer = self.sonar_t2v.tokenizer.create_encoder(lang='eng_Latn')
        self.sonar_vocab_size = self.sonar_t2v.tokenizer.vocab_info.size
        self.sonar_decoder = self.sonar_t2v.tokenizer.create_decoder()
        self.get_orig_only = True

        self.adv_loss = []
        self.sim_loss = []
        self.sonar_perplexity = []
        self.reward = []
        self.policy_loss = []
        self.value_loss = []
        self.total_loss = []
        if self.kwargs.get('compile', False):
            self.model = torch.compile(self.model, mode='max-autotune')
            self.sonar_v2t = torch.compile(self.sonar_v2t, mode='max-autotune')
            # self.sonar_t2v = torch.compile(self.sonar_t2v, mode='max-autotune')


    @torch.no_grad
    def sonar_tokenize(self, text):
        return self.sonar_t2v.tokenizer.create_encoder(lang="eng_Latn")(text)[1:-1].tolist()
    

    def sonar_decode(self, tokens):
        return str(self.sonar_t2v.tokenizer.create_decoder()(torch.tensor(tokens)))
    
    def _init_output(self, sample, crop_attack: bool = False, first_part: bool = True) -> edict:
        gen_output = edict()
        gen_output.sample = sample
        
        gen_output.sample['gt_mask'] = gen_output.sample['masks_list']
        gen_output.gt_mask = gen_output.sample['gt_mask'][0].to(self.kwargs.device)
        
        gen_output.gt_mask[gen_output.gt_mask == 255] = 1
        gen_output.gt_mask = 1 - gen_output.gt_mask

        # Crutch FIX of the cases where there are not only 0 and 1
        gen_output.gt_mask[gen_output.gt_mask != 1] = 0

        gen_output.start_adv_token, gen_output.end_adv_token = select_tokens_for_attack(
            sample['conversation_list'][0],
            sample['input_ids'][0].tolist(),
            tokenizer=self.tokenizer,
            crop_attack=crop_attack,
            first_part=first_part
        )
        
        gen_output.original_context = cut_original_context(
            gen_output.sample,
            model=self.model,
            num_image_tokens=self.model.num_image_tokens,
            start_adv_token=gen_output.start_adv_token,
            end_adv_token=gen_output.end_adv_token,
            device=self.kwargs.device
        )

        gen_output.max_adv_len = max(10, int((gen_output.end_adv_token - gen_output.start_adv_token) * 2))

        with torch.no_grad():
            gen_output.orig_output = inference(
                gen_output.sample, 
                self.model, 
                gen_output.original_context, 
                device=self.kwargs.device
            )
            gen_output.input_ids = gen_output.sample['input_ids'][0, gen_output.start_adv_token:gen_output.end_adv_token]
            gen_output.orig_output.input_ids = gen_output.input_ids.clone()

        gen_output.text = ["WARNING, FOR SOME REASON ATTACK STEP FAILED"]

        return gen_output

    
    def policy(self, embedding, log_scale, action=None):
        distribution = torch.distributions.Normal(embedding.squeeze(0), scale=torch.log1p(torch.exp(log_scale)))
        distribution = torch.distributions.Independent(distribution, 1)
        if action is None:
            action = distribution.sample(torch.Size([self.kwargs.sample_size]))
        log_prob = distribution.log_prob(action)
        return action, log_prob
    

    def loss(self, inf_output, gen_output, log_prob, value, sonar_probs):
        with torch.no_grad():
            # Segmentation loss
            if torch.all(inf_output['pred_masks'][0] == 0) or torch.all(torch.isnan(inf_output['pred_masks'][0])): 
                adv_loss = torch.tensor(0.0, device=self.kwargs.device, requires_grad=True)
            else:
                ious = []
                for i in range(len(inf_output['pred_masks'])):
                    pred_mask = (inf_output['pred_masks'][i] > 0).int()
                    ious.append(
                        calculate_iou(pred_mask, gen_output.gt_mask.int())
                    )
                adv_loss = torch.tensor(ious, device=self.kwargs.device, dtype=torch.float32) * self.kwargs.adv_weight
            
            # sonar perplexity
            sonar_perplexity = torch.tensor(sonar_probs, device=self.kwargs.device) * self.kwargs.sonar_perplexity_weight
            reward = (-adv_loss + sonar_perplexity) / (self.kwargs.adv_weight + self.kwargs.sonar_perplexity_weight)

        # sonar embedding distance
        sim_loss = torch.nn.functional.mse_loss(gen_output.embedding, self.initial_embedding, reduction='mean') * self.kwargs.sim_weight
        diff = log_prob - self.old_log_prob.detach()
        diff = torch.clamp(diff, min=-100, max=100)
        ratio = torch.exp(diff)
        advantage = reward - value.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # print("reward: ", reward)
        # print("ratio: ", ratio)
        # print("advantage: ", advantage)
        policy_loss = -torch.mean(torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.kwargs.clip_ratio, 1 + self.kwargs.clip_ratio) * advantage))
        # policy_loss = -torch.mean((reward - value.detach()) * log_prob)
        value_loss = torch.nn.functional.mse_loss(reward.view(self.kwargs.sample_size), value.view(1).expand(self.kwargs.sample_size), reduction='mean')
        self.adv_loss.append(adv_loss.mean().item())
        self.sim_loss.append(sim_loss.item())
        self.sonar_perplexity.append(sonar_perplexity.mean().item())
        self.reward.append(reward.mean().item())
        self.policy_loss.append(policy_loss.item())
        self.value_loss.append(value_loss.item())
        self.total_loss.append(policy_loss.item() + value_loss.item() + sim_loss.item())
        return policy_loss + value_loss + sim_loss
    
    def __call__(self, sample) -> t.Generator[edict, torch.Any, None]:
        if self.get_orig_only:
            self.get_orig_only = False
            gen_output = self._init_output(sample, crop_attack=False, first_part=False)
            yield gen_output
            self.get_orig_only = True
        else:
            gen_outputs = [
                self._init_output(sample, crop_attack=False, first_part=False),
                # self._init_output(sample, crop_attack=True, first_part=True),
                # self._init_output(sample, crop_attack=True, first_part=False),
            ]
            
            try:
                for gen_output in gen_outputs:

                    with torch.inference_mode():
                        gen_output.embedding = self.sonar_t2v.predict(
                            [self.tokenizer.decode(gen_output.input_ids)], source_lang="eng_Latn"
                        )
                    self.initial_embedding = gen_output.embedding.clone().detach()

                    gen_output.embedding = gen_output.embedding.clone().requires_grad_(True)

                    state_value = torch.tensor(0, device=self.kwargs.device, requires_grad=True, dtype=torch.float32)
                    noise_scale = self.kwargs.noise_scale + np.log(-np.expm1(-self.kwargs.noise_scale))
                    log_scale = torch.tensor(noise_scale, device=self.kwargs.device, dtype=torch.float32, requires_grad=True)
                    optimizer = torch.optim.Adam([
                        {'params': [gen_output.embedding], 'lr': self.lr},
                        {'params': [state_value], 'lr': self.kwargs.value_lr},
                        {'params': [log_scale], 'lr': self.kwargs.log_scale_lr}
                    ])

                    
                    for iteration in range(self.num_iters):
                        with torch.inference_mode():
                            action, self.old_log_prob = self.policy(gen_output.embedding, log_scale)
                            # print("action: ", action.shape, action[:2, :10])
                            gen_output.text, sonar_probs = self.sonar_v2t.predict(
                                action, 
                                target_lang="eng_Latn", 
                                max_seq_len=gen_output.max_adv_len
                            )
                            # print(gen_output.text)
                            adv_ids = self.tokenizer(gen_output.text, add_special_tokens=False)['input_ids']
                            output = self.inference(
                                gen_output.sample,
                                self.model, 
                                gen_output.original_context,
                                adv_embeds=None,
                                adv_ids=adv_ids,
                                device=self.kwargs.device
                            )
                        for _ in range(self.kwargs.policy_update_interval):
                            optimizer.zero_grad()
                            _, log_prob = self.policy(gen_output.embedding, log_scale, action)
                            total_loss = self.loss(output, gen_output, log_prob, state_value, sonar_probs) #/ self.kwargs.policy_update_interval
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(gen_output.embedding.grad, 1.0)
                            optimizer.step()
                        # print("GRAD:")
                        # print(state_value.grad)
                        # print(log_scale.grad)
                        # print(gen_output.embedding.grad.norm())

                        with torch.inference_mode():
                            # print("PARAMETERS: ")
                            # print("Scale: ", torch.log1p(torch.exp(log_scale.detach())))
                            # print("--------------------------------")
                            gen_output.text, sonar_probs = self.sonar_v2t.predict(
                                gen_output.embedding, 
                                target_lang="eng_Latn", 
                                max_seq_len=gen_output.max_adv_len
                            )

                        yield gen_output
                        
            except Exception as e:
                print(e)
                print(gen_output.text)
        
        for loss_name, loss_values in zip(
            ['Adv Loss', 'Sim Loss', 'Sonar Perplexity', 'Reward', 'Policy Loss', 'Value Loss', 'Total Loss'],
            [self.adv_loss, self.sim_loss, self.sonar_perplexity, self.reward, self.policy_loss, self.value_loss, self.total_loss]
        ):
            plt.plot(torch.arange(len(loss_values)), loss_values, label=loss_name)
            plt.plot(torch.arange(len(loss_values))[5:-4], np.convolve(loss_values, np.ones(10)/10, mode='valid'), label=f'Smoothed {loss_name}')
            plt.legend()
            plt.savefig(f'andrew_debug/{sample["image_paths"][0].split("/")[-1].split(".")[0]}_{loss_name}.png')
            plt.close()
            loss_values.clear()
        return None

    @torch.no_grad
    def generate(self, gen_output):
        return [gen_output.text[0]]

    def inference(
        self,
        sample: dict, 
        model: torch.nn.Module, 
        original_context: dict = None, 
        adv_embeds: torch.Tensor = None, 
        adv_ids: torch.Tensor = None,
        device='cuda'
        ) -> dict:
        """
        Perform inference with adversarial embeddings or adversarial IDs.
        """
        sample = dict_to_cuda(sample, device=self.kwargs.device)

        new_input_ids_seqs = []
        for i in range(len(adv_ids)):
            new_input_ids_seqs.append(torch.cat([
                original_context['prefix_ids'],
                torch.tensor(adv_ids[i], dtype=torch.long, device=device),
                original_context['suffix_ids']
            ], dim=0).int())
        input_ids = pad_sequence(new_input_ids_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id).long()

        new_attention_masks_seqs = []
        for i in range(len(adv_ids)):
            new_attention_masks_seqs.append(torch.cat([
                original_context['prefix_attn'],
                torch.ones(len(adv_ids[i]), dtype=torch.bool, device=device),
                original_context['suffix_attn']
            ], dim=0).bool())

        with torch.amp.autocast('cuda'):
            output_dict = model(
                images=sample['images'].expand(self.kwargs.sample_size, -1, -1, -1).half(),
                images_clip=sample['images_clip'].half(),
                input_ids = input_ids,
                attention_masks = pad_sequence(new_attention_masks_seqs, batch_first=True, padding_value=False).bool(),
                masks_list=sample['masks_list'] * self.kwargs.sample_size,
                label_list = sample['label_list'] * self.kwargs.sample_size,
                resize_list = sample['resize_list'] * self.kwargs.sample_size,
                labels = AttackerSonarReinforceWithBaseline.adjust_tensor_size(
                    sample['labels'], 
                    input_ids.size()[1]
                ),
                offset = torch.arange(self.kwargs.sample_size + 1, device=device),
                inference = True
            )
        # import json
        # # print(output_dict)
        # with open(f'andrew_debug/{sample["image_paths"][0].split("/")[-1].split(".")[0]}_output_dict.json', 'w') as f:
        #     try:
        #         json.dump({k: v for k, v in output_dict.items() if isinstance(v, (dict, list, str, int, float, bool))}, f)
        #     except Exception as e:
        #         pass
        # import sys
        # sys.exit()
        return output_dict


    @staticmethod
    def adjust_tensor_size(
        tensor: torch.Tensor, 
        target_size: int, 
        pad_value: int = -100
        ) -> torch.Tensor:
        """
        Adjust the size of a tensor to match the target size, padding or truncating as necessary.
        """
        current_size = tensor.size(-1)
        
        if current_size > target_size:
            truncated_tensor = tensor[:, -target_size:]
        else:
            padding_size = target_size - current_size
            # padding = torch.full((padding_size,), pad_value)
            # truncated_tensor = torch.cat((padding, tensor.cpu()), dim=0)
            truncated_tensor = torch.nn.functional.pad(tensor, (padding_size, 0), value=pad_value)
        return truncated_tensor

if __name__ == "__main__":
    # from fairseq2.nn.incremental_state import IncrementalStateBag
    # sonar_t2v = TextToEmbeddingModelPipeline(
    #     encoder="text_sonar_basic_encoder", 
    #     tokenizer="text_sonar_basic_encoder",
    #     device=torch.device("cuda")
    # )
    # vector = sonar_t2v.predict(["the region exhibiting unusual color"], source_lang="eng_Latn", batch_size=1)
    # sonar_v2t = EmbeddingToTextModelPipeline(
    #     decoder="text_sonar_basic_decoder", 
    #     tokenizer="text_sonar_basic_encoder",
    #     device=torch.device("cuda")
    # )

    # from fairseq2.nn.transformer.multihead_attention import FullAttentionState, StandardMultiheadAttention
    # class FullAttentionStateDifferentiable(FullAttentionState):
    #     def __init__(self, k, v, max_seq_len):
    #         batch_size, num_heads, seq_len, head_dim = k.shape

    #         self.k = k.new_empty((batch_size, num_heads, 0, head_dim))
    #         self.v = v.new_empty((batch_size, num_heads, 0, head_dim))

    #         self.k = torch.cat([self.k, k], dim=2)
    #         self.v = torch.cat([self.v, v], dim=2)

    #         self.seq_len = seq_len

    #     def append(self, k, v):

    #         self.k = torch.cat([self.k, k], dim=2)
    #         self.v = torch.cat([self.v, v], dim=2)

    #         self.seq_len += 1


    #     def get(self):
    #         k = self.k
    #         v = self.v

    #         return k, v

    #     def reorder(self, new_order):
    #         self.k = self.k.index_select(0, new_order)
    #         self.v = self.v.index_select(0, new_order)

    # for submodule in sonar_v2t.modules():
    #     if isinstance(submodule, StandardMultiheadAttention):
    #         submodule.state_factory = FullAttentionStateDifferentiable

    # from fairseq2.generation.step_processor import StepProcessor
    # class GumbelProcessor(StepProcessor):
    #     def __init__(self, tau=1.0):
    #         self.tau = tau
    #         self.topk = 20

    #     def __call__(self, seqs, lprobs, lprob):
    #         indices = torch.topk(lprobs, k=self.topk, dim=-1).indices
    #         maskall = torch.full_like(lprobs, dtype=torch.float32, fill_value=-torch.inf)
    #         maskall[..., indices] = lprobs[..., indices]
    #         lprobs = maskall
    #         lprobs.data = (lprobs - torch.log(-torch.log(torch.rand_like(lprobs)))) / self.tau

    # # breakpoint()
    # # vector = vector.expand(10, -1)
    # # with torch.inference_mode():
    # #     output = sonar_v2t.predict(vector, "eng_Latn", max_seq_len=100, temperature=1.0, step_processors=[GumbelProcessor(tau=2.0)])
    # # print(output)
    # # exit()
    # vector = vector.detach().clone()
    # vector.requires_grad = True
    # vector = vector.expand(10, -1)
    # # vector = vector + torch.randn_like(vector) * 0.005
    # target_text_encoder = sonar_v2t.tokenizer.create_encoder(task='translation', lang="eng_Latn", mode='target')
    # text_decoder = sonar_v2t.tokenizer.create_decoder()
    # prompt_seqs = torch.nn.functional.one_hot(target_text_encoder.prefix_indices, num_classes=sonar_v2t.model.target_vocab_info.size).float().cuda()
    # prompt_seqs = prompt_seqs.unsqueeze(0)
    # # print(target_text_encoder.prefix_indices, prompt_seqs.shape)
    # prompt_padding_mask = None
    # source_padding_mask = None
    # encoder_output, encoder_padding_mask = sonar_v2t.model.encode(
    #         vector, None
    # )
    
    # max_seq_len = 20
    # min_seq_len = 1
    # unk_penalty = 100
    # state_bag = IncrementalStateBag(max_seq_len)
    
    # padding_mask = prompt_padding_mask

    # #prefill
    # prefill_len = 2
    
    # sonar_v2t.model.decoder.decoder_frontend.embed.forward = MethodType(
    #     lambda self, x: x @ self.weight,
    #     sonar_v2t.model.decoder.decoder_frontend.embed
    # )
    # seqs = prompt_seqs.expand(10, -1, -1)

    # decoder_output, decoder_padding_mask = sonar_v2t.model.decode(
    #         seqs[:, :prefill_len - 1],
    #         None,
    #         encoder_output,
    #         encoder_padding_mask,
    #         state_bag=state_bag,
    #     )
    # model_output = sonar_v2t.model.project(decoder_output, decoder_padding_mask)
    # state_bag.increment_step_nr(prefill_len - 1)
    # # state_bag.reorder(torch.tensor([0], device='cuda'))
    # # logits = model_output.logits
    # # lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32) # ?

    # is_eos = False
    # while not is_eos:
    #     decoder_output, decoder_padding_mask = sonar_v2t.model.decode(
    #         seqs[:, -1:],
    #         None,
    #         encoder_output,
    #         encoder_padding_mask,
    #         state_bag=state_bag,
    #     )
    #     model_output = sonar_v2t.model.project(decoder_output, decoder_padding_mask)
    #     state_bag.increment_step_nr()
    #     # state_bag.reorder(torch.tensor([0], device='cuda'))
    #     logits = model_output.logits[:, -1, :]
    #     top_logit = torch.kthvalue(logits, logits.size(-1) - 10, dim=-1, keepdim=True).values
    #     logits = torch.where(logits >= top_logit, logits, torch.full_like(logits, -torch.inf))
    #     probs = torch.softmax(logits / 2, dim=-1)
    #     probs = torch.nn.functional.one_hot(torch.multinomial(probs, num_samples=1), num_classes=sonar_v2t.model.target_vocab_info.size)
        # mask = probs > 0.5
        # top_logit = torch.kthvalue(logits, logits.size(-1) - 100, dim=-1, keepdim=True).values
        # logits = torch.where(logits >= top_logit, logits, torch.full_like(logits, -torch.inf))
        # probs = torch.nn.functional.gumbel_softmax(logits, tau=10000.0, hard=False, dim=-1)
        # print(probs)
        # lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        # lprobs = lprobs.squeeze(1)
        # mask = torch.ones_like(lprobs, dtype=torch.bool)
        # good_ids = torch.load('/home/jovyan/zinkovich/ref-seg-text-break/notebooks/meta_tokenize/matched_sonar_ids.pt')
        # mask[:, good_ids] = False
        # mask[:, [3]] = False
        # lprobs = lprobs.masked_fill(mask, -torch.inf)
        # lprobs = torch.scatter_add(lprobs, dim=-1, index=sonar_v2t.model.target_vocab_info.unk_idx, src=-unk_penalty)
        # lprobs = torch.scatter(lprobs, dim=-1, index=sonar_v2t.model.target_vocab_info.pad_idx, src=-torch.inf)
        # if seqs.size(1) < min_seq_len:
        #     lprobs = torch.scatter(lprobs, dim=-1, index=sonar_v2t.model.target_vocab_info.eos_idx, src=-torch.inf)
        # probs = torch.exp(lprobs)
    #     if False and (probs.argmax(dim=-1) == sonar_v2t.model.target_vocab_info.eos_idx):
    #         is_eos = True
    #     else:
    #         seqs = torch.cat([seqs, probs], dim=1)
    #     if seqs.size(1) == max_seq_len:
    #         break; # ?
    # for i in range(seqs.size(0)):
    #     print(str(text_decoder(seqs[i].argmax(dim=-1).squeeze(0))))
    # print([state_bag._module_states.values()])
    # log_perplexity = -torch.mean(seqs.max(dim=-1).values)
    # log_perplexity.backward()
    # print(vector.grad)
    
    # #     print(model_output)
    # #     break

    from sonar.models.sonar_text import load_sonar_tokenizer
    sonar_tokenizer = load_sonar_tokenizer('text_sonar_basic_encoder', progress=True)
    sonar_tokenize_encoder = sonar_tokenizer.create_encoder(task='translation', lang="eng_Latn", mode='source')
    # sonar_token_list = []
    # for i in range(sonar_tokenizer.vocab_info.size):
    #     sonar_token_list.append(str(sonar_tokenizer.model.index_to_token(i)))

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"

    def setup_tokenizer(cfg):
        tokenizer =  AutoTokenizer.from_pretrained(
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

    config_path = "/home/jovyan/zinkovich/ref-seg-text-break/configs/segmod/lisa-13b-v1-exp.yaml"
    cfg = OmegaConf.load(config_path)

    lisa_tokenizer = setup_tokenizer(cfg)
    # lisa_token_list = list(lisa_tokenizer.get_vocab().keys())
    # with open(f'/home/jovyan/zinkovich/ref-seg-text-break/andrew_debug/lisa_token_list.txt', 'w') as f:
    #     for token in lisa_token_list:
    #         f.write(token + '\n')
    # with open(f'/home/jovyan/zinkovich/ref-seg-text-break/andrew_debug/sonar_token_list.txt', 'w') as f:
    #     for token in sonar_token_list:
    #         f.write(token + '\n')

    # print(lisa_tokenizer.tokenize("Hello,world!"))
    
    # print(len(sonar_token_list), len(lisa_token_list), len(sonar_token_list) * len(lisa_token_list))
    import spacy_alignments as tokenizations
    from pylcs import lcs_string_length
    from tqdm import tqdm

    def ratio(token_a, token_b):
        intersection = lcs_string_length(token_a, token_b)
        return intersection / (len(token_a) + len(token_b) - intersection)

    from datasets import load_dataset
    import re

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    from collections import defaultdict
    matrix = defaultdict(lambda: defaultdict(float))
    # matrix = torch.sparse_coo_tensor(size=(lisa_tokenizer.vocab_size, sonar_tokenizer.vocab_info.size), dtype=torch.float32)

    for text in tqdm(ds['train']['text']):
        # print(text)
        text = re.sub(r'[^\x00-\x7f]',r'', text)
        sonar_subwords = list(map(str, sonar_tokenize_encoder.encode_as_tokens(text)[1:-1]))
        lisa_subwords = lisa_tokenizer.tokenize(text, add_special_tokens=False)[:-1]
        alignment, _ = tokenizations.get_alignments(lisa_subwords, sonar_subwords)
        # print(sonar_subwords, lisa_subwords, alignment)
        for i in range(len(lisa_subwords)):
            for j in alignment[i]:
                matrix[lisa_tokenizer.convert_tokens_to_ids(lisa_subwords[i])][sonar_tokenizer.model.token_to_index(sonar_subwords[j])] += ratio(lisa_subwords[i], sonar_subwords[j])
    

    exact_match = {}

    for i in range(sonar_tokenizer.vocab_info.size):
        subword = sonar_tokenizer.model.index_to_token(i)
        lisa_ids = lisa_tokenizer.convert_tokens_to_ids(subword)
        if lisa_ids != 0:
            exact_match[i] = lisa_ids
    
    for i in exact_match:
        del matrix[exact_match[i]]

    indices = []
    values = []
    for i in matrix:
        for j in matrix[i]:
            indices.append([i, j])
            values.append(matrix[i][j])
    
    for i in exact_match:
        indices.append([exact_match[i], i])
        values.append(1.0)

    sparse_matrix = torch.sparse_coo_tensor(
        indices = torch.tensor(indices).T,
        values = torch.tensor(values),
        size=(lisa_tokenizer.vocab_size, sonar_tokenizer.vocab_info.size),
        dtype=torch.float32
        )
    
    sparse_matrix = sparse_matrix / sparse_matrix.sum(dim=1, keepdim=True)
    torch.save(sparse_matrix, f'/home/jovyan/zinkovich/ref-seg-text-break/andrew_debug/lisa_sonar_transfer_matrix.pt')
            

