from rtext.sentence_optimization.utils import calculate_iou
from rtext.sentence_optimization.train import AttackerBase
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
import torch.nn.functional as F
import torch


class MetricsCalculator:
    def __init__(self, attacker: AttackerBase, embed_name="microsoft/mpnet-base"):
        self.attacker = attacker
        # self.tokenizer = AutoTokenizer.from_pretrained(embed_name)
        # self.embed_model = AutoModel.from_pretrained(embed_name)
        
    @torch.no_grad()
    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
    @torch.no_grad()
    def cos_sim(self, orig_text, adv_text, **kwargs):
        return F.cosine_similarity(
            self._get_embedding(orig_text), 
            self._get_embedding(adv_text), 
        dim=-1).item()
    
    @staticmethod
    def iou(gt_mask, pred_mask, **kwargs):
        return calculate_iou(gt_mask, pred_mask)
