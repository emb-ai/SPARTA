import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bert_score import BERTScorer
import warnings
import os
import json


class TextSimilarityEvaluator:

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        print("Loading BERTScore model...")
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=self.device)
        
        print("Loading BLEURT model...")
        self.bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
        self.bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512").to(self.device)
            
        self.smoothing = SmoothingFunction().method1
    

    def calculate_bleu4(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        if len(references) != len(candidates):
            raise ValueError(f"Number of references ({len(references)}) and candidates ({len(candidates)}) must match")
            
        bleu4_scores = []
        tokenized_refs = []
        tokenized_cands = []
        
        for ref, cand in zip(references, candidates):
            ref_tokens = nltk.word_tokenize(ref.lower())
            cand_tokens = nltk.word_tokenize(cand.lower())
            
            tokenized_refs.append([ref_tokens]) 
            tokenized_cands.append(cand_tokens)
            
            bleu4_scores.append(
                sentence_bleu([ref_tokens], cand_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25), 
                smoothing_function=self.smoothing)
            )
        
        corpus_bleu4 = corpus_bleu(
            tokenized_refs, tokenized_cands, 
            weights=(0.25, 0.25, 0.25, 0.25), 
            smoothing_function=self.smoothing
        )
        return {
            'bleu4_avg': np.mean(bleu4_scores),
            'bleu4_corpus': corpus_bleu4
        }
    

    def calculate_rougeL(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        rougeL_p, rougeL_r, rougeL_f = [], [], []
        
        for ref, cand in zip(references, candidates):
            scores = self.rouge_scorer.score(ref, cand)
            
            rougeL_p.append(scores['rougeL'].precision)
            rougeL_r.append(scores['rougeL'].recall)
            rougeL_f.append(scores['rougeL'].fmeasure)
        
        result = {
            'rougeL_precision': np.mean(rougeL_p),
            'rougeL_recall': np.mean(rougeL_r),
            'rougeL_f1': np.mean(rougeL_f)
        }
        return result
    

    def calculate_bertscore(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        P, R, F1 = self.bert_scorer.score(candidates, references)
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    

    def calculate_bleurt(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        bleurt_scores = []
        batch_size = 8
        for i in range(0, len(references), batch_size):
            batch_refs = references[i:i+batch_size]
            batch_cands = candidates[i:i+batch_size]
            
            inputs = self.bleurt_tokenizer(batch_refs, batch_cands, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                scores = self.bleurt_model(**inputs).logits
            
            bleurt_scores.extend(scores.cpu().numpy().tolist())
        
        return {'bleurt': np.mean(bleurt_scores)}


    def visualize_results(self, results: Dict[str, Any], output_dir: str = "./results") -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_to_plot = {
            'BLEU-4 (corpus)': results['bleu4_corpus'],
            'ROUGE-L (F1)': results['rougeL_f1'],
            'BERTScore (F1)': results['bertscore_f1'],
            'BLEURT': results['bleurt']
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.title('Key Similarity Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1) 
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "key_metrics.png"))
        plt.close()
    

    def evaluate_all(self, references: List[str], candidates: List[str]) -> Dict[str, Any]:

        if len(references) != len(candidates):
            raise ValueError(f"Number of references ({len(references)}) and candidates ({len(candidates)}) must match")
        
        results = {}
        
        print("Calculating BLEU-4 score...")
        bleu_scores = self.calculate_bleu4(references, candidates)
        results.update(bleu_scores)
        
        print("Calculating ROUGE-L scores...")
        rouge_scores = self.calculate_rougeL(references, candidates)
        results.update(rouge_scores)
        
        print("Calculating BERTScore...")
        bertscore_scores = self.calculate_bertscore(references, candidates)
        results.update(bertscore_scores)
        
        print("Calculating BLEURT score...")
        bleurt_scores = self.calculate_bleurt(references, candidates)
        results.update(bleurt_scores)
        
        return results
    

def calculate_text_similarity(original_texts: List[str], 
                              restored_texts: List[str],
                              output_dir: str = "./results",
                              visualize: bool = True) -> Dict[str, Any]:
    if len(original_texts) != len(restored_texts):
        raise ValueError(f"Number of original texts ({len(original_texts)}) and restored texts ({len(restored_texts)}) must match")
    
    if len(original_texts) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    print(f"Evaluating {len(original_texts)} text pairs...")
    
    evaluator = TextSimilarityEvaluator()
    
    results = evaluator.evaluate_all(original_texts, restored_texts)
    
    if visualize:
        evaluator.visualize_results(results, output_dir=output_dir)
    
    print("\nResults Summary:")
    print(f"BLEU-4 (corpus): {results['bleu4_corpus']:.4f}")
    print(f"ROUGE-L (F1): {results['rougeL_f1']:.4f}")
    print(f"BERTScore (F1): {results['bertscore_f1']:.4f}")
    print(f"BLEURT: {results['bleurt']:.4f}")
    
    return results