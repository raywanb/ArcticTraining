from projects.swiftkv.models.qwen3 import Qwen3SwiftKVForCausalLM
from projects.swiftkv.strategy_search import run_strategy_search
from datasets import load_dataset

# Load real calibration data from wikitext
print("Loading calibration data from wikitext-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Filter for substantial text samples (avoid headers and short lines)
texts = []
for sample in dataset:
    text = sample["text"].strip()
    # Keep samples with at least 100 characters (roughly 20-30 tokens)
    if len(text) > 100 and not text.startswith("="):  # Skip wiki headers
        texts.append(text)
    if len(texts) >= 50:  # Use 50 calibration samples
        break

print(f"Selected {len(texts)} calibration samples")
print(f"Sample text (first 200 chars): {texts[0][:200]}...\n")

# ============================================================================
# Policy 1: "distance" - Most dissimilar pairs (fastest, no evaluation)
# ============================================================================
print("\n" + "="*80)
print("POLICY: distance (most dissimilar pairs)")
print("="*80)
sharing_map_distance, info_distance = run_strategy_search(
    model_class=Qwen3SwiftKVForCausalLM,
    model_name="Qwen/Qwen3-8B",
    calibration_texts=texts,
    target_shared_layers=12,
    policy="distance",  # <-- New policy: directly uses most dissimilar pairs
    distance_metric="euclidean",  # or "cosine", "manhattan"
    batch_size=1,
    max_length=512,
    device="cuda:0",
)
print("\nDistance Policy Results:")
print(f"Sharing map: {sharing_map_distance}")
print(f"Similarity: {info_distance['similarity']:.6f}")
print(f"Memory reduction: {info_distance['memory_reduction_pct']:.2f}%\n")

# ============================================================================
# Policy 2: "greedy" - Adaptive best-first search (slower, evaluates quality)
# ============================================================================
print("\n" + "="*80)
print("POLICY: greedy (adaptive best-first search)")
print("="*80)
sharing_map_greedy, info_greedy = run_strategy_search(
    model_class=Qwen3SwiftKVForCausalLM,
    model_name="Qwen/Qwen3-8B",
    calibration_texts=texts,
    target_shared_layers=12,
    similarity_threshold=0.96,
    policy="greedy",
    distance_metric="euclidean",
    top_k=5,  # Evaluate top 5 candidates per round
    batch_size=1,
    max_length=512,
    device="cuda:0",
)
print("\nGreedy Policy Results:")
print(f"Sharing map: {sharing_map_greedy}")
print(f"Similarity: {info_greedy['similarity']:.6f}")
print(f"Memory reduction: {info_greedy['memory_reduction_pct']:.2f}%\n")

# ============================================================================
# Policy 3: "cluster" - Hierarchical clustering (fast heuristic)
# ============================================================================
print("\n" + "="*80)
print("POLICY: cluster (hierarchical clustering)")
print("="*80)
sharing_map_cluster, info_cluster = run_strategy_search(
    model_class=Qwen3SwiftKVForCausalLM,
    model_name="Qwen/Qwen3-8B",
    calibration_texts=texts,
    target_shared_layers=12,
    policy="cluster",
    distance_metric="euclidean",
    linkage_method="average",  # or "ward", "complete", "single"
    batch_size=1,
    max_length=512,
    device="cuda:0",
)
print("\nCluster Policy Results:")
print(f"Sharing map: {sharing_map_cluster}")
print(f"Similarity: {info_cluster['similarity']:.6f}")
print(f"Memory reduction: {info_cluster['memory_reduction_pct']:.2f}%\n")
