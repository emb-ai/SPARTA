from __future__ import annotations

import argparse
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Protocol, Any
from tqdm import tqdm
from datetime import datetime

import pandas as pd

from texts_sim import calculate_text_similarity


class TextEmbedder(Protocol):
    """Minimal interface every embedder must implement."""

    def encode(self, text: str) -> Any:
        """Return an intermediate representation (e.g. latent, audio, image)."""

    def decode(self, latent: Any) -> str:
        """Decode the representation back into human-readable text."""



def load_dataset(dataset_path: str) -> List[str]:
    """Load raw text dataset."""
    texts = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.json'):
            with open(os.path.join(dataset_path, filename), 'r') as file:
                data = json.load(file)
                file_texts = data.get('text', [])[0]
                texts.append(file_texts)
    return texts


class SonarEmbedder: 
    def __init__(self):
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

        self.sonar_t2v = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder",
            device='cuda'
        )
        self.sonar_v2t = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device='cuda'
        )

    def encode(self, text: str) -> str:
        return self.sonar_t2v.predict([text], source_lang="eng_Latn")

    def decode(self, latent: str) -> str:
        return self.sonar_v2t.predict(latent, target_lang="eng_Latn", max_seq_len=512)[0]


class CLIPEmbedder: 
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    @torch.no_grad()
    def encode(self, text: str) -> str:
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def decode(self, latent: str) -> str:
        return self.sonar_v2t.predict(latent, target_lang="eng_Latn", max_seq_len=512)[0]




EMBEDDERS: Dict[str, TextEmbedder] = {
    "sonar": SonarEmbedder(),
    # "clip": CLIPEmbedder(),
}


def evaluate_embedder(name: str, embedder: TextEmbedder, texts: List[str]) -> Dict[str, float]:
    """Encode → decode the whole corpus and compute similarity metrics."""

    latents = [embedder.encode(t) for t in tqdm(texts)]
    restored = [embedder.decode(l) for l in tqdm(latents)]

    metrics = calculate_text_similarity(texts, restored, visualize=False)

    metrics["samples"] = len(texts)
    metrics["embedder"] = name
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple text embedders.")
    parser.add_argument("--dataset_path", default="../../dataset/reason_seg/ReasonSeg/train", help="Path to input dataset file.")
    parser.add_argument(
        "--output_dir",
        default="results/embedder_eval",
        help="Base directory where a timestamped sub-directory will be created to store results.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    print(f"Loading dataset from {args.dataset_path} …")
    texts = load_dataset(args.dataset_path)
    print(f"▶ Loaded {len(texts):,} samples.")

    # 2. Evaluate each embedder
    all_results: List[Dict[str, float]] = []
    for name, embedder in EMBEDDERS.items():
        print(f"\n▶ Evaluating embedder: {name}")
        metrics = evaluate_embedder(name, embedder, texts)
        all_results.append(metrics)

    # 3. Aggregate + save
    df = pd.DataFrame(all_results).set_index("embedder")
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path)

    print("\nEvaluation complete. Results:")
    print(df.round(4))
    print(f"\nCSV summary saved to {csv_path.absolute()}")


if __name__ == "__main__":
    main()
