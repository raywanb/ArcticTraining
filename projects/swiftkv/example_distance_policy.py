#!/usr/bin/env python3
"""
Simple example: Using the "distance" policy for most dissimilar pairs.

This policy selects the most dissimilar layer pairs directly based on 
distance metrics (no evaluation needed - fastest option).
"""

from projects.swiftkv.models.qwen3 import Qwen3SwiftKVForCausalLM
from projects.swiftkv.strategy_search import run_strategy_search
from datasets import load_dataset

# Step 1: Load calibration data
print("Loading calibration data...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

texts = []
for sample in dataset:
    text = sample["text"].strip()
    if len(text) > 100 and not text.startswith("="):
        texts.append(text)
    if len(texts) >= 50:
        break

print(f"Loaded {len(texts)} calibration samples\n")

# Step 2: Run strategy search with "distance" policy
sharing_map, diagnostics = run_strategy_search(
    model_class=Qwen3SwiftKVForCausalLM,  # Your SwiftKV model class
    model_name="Qwen/Qwen3-8B",            # Model name/path
    calibration_texts=texts,
    target_shared_layers=12,               # Number of layers to share
    policy="distance",                      # <-- Use most dissimilar pairs
    distance_metric="euclidean",           # Options: "euclidean", "cosine", "manhattan"
    batch_size=1,
    max_length=512,
    device="cuda:0",
)

# Step 3: Results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Sharing map: {sharing_map}")
print(f"Similarity: {diagnostics['similarity']:.6f}")
print(f"Memory reduction: {diagnostics['memory_reduction_pct']:.2f}%")
print("="*80)


















