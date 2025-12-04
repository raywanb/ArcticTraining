#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SwiftKV Strategy Search (compact)
- Policy "greedy": Adaptive best-first search with similarity evaluation
- Policy "cluster": Hierarchical clustering heuristic
- Policy "distance": Most dissimilar pairs (no evaluation, fastest)
"""

from typing import Dict, List, Tuple, Optional, Iterable
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ------------------------------ Data -----------------------------------------

class CalibrationDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i],
                       max_length=self.max_length,
                       truncation=True,
                       padding="max_length",
                       return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def make_loader(texts: List[str], tokenizer, batch_size: int, max_length: int) -> DataLoader:
    return DataLoader(CalibrationDataset(texts, tokenizer, max_length),
                      batch_size=batch_size, shuffle=False)


# ------------------------------ Utils ----------------------------------------

def resolve_device_map(device: str):
    """
    Returns a device_map suitable for HF .from_pretrained.
    - "cuda:0" -> {"": "cuda:0"}
    - "cuda" / "cpu" / "auto" -> "cuda"/"cpu"/"auto"
    """
    if isinstance(device, str) and device.startswith("cuda:"):
        return {"": device}
    return device


def _running_mean_update(mean: torch.Tensor, count: int, batch_sum: torch.Tensor, n_new: int):
    if count == 0:
        return (batch_sum / n_new), n_new
    total = count + n_new
    mean = (mean * count + batch_sum) / total
    return mean, total


# ------------------------------ KV stats -------------------------------------

@torch.no_grad()
def collect_kv_means_streaming(model, loader: DataLoader, device: str = "cuda") -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Streaming per-layer means for key/value (on CPU, float32).
    Returns {layer: {"k": mean_k, "v": mean_v}}.
    
    NOTE: This uses means as a lossy compression. For more accurate similarity,
    use collect_layer_similarity_direct() which computes token-by-token similarity.
    
    Why means? They're computationally efficient - we only need to store/compare
    one vector per layer instead of millions of KV vectors. However, they lose
    variance/distribution information. Two layers with similar means can have
    very different distributions.
    """
    model.eval()
    dev = next(model.parameters()).device if device == "auto" else torch.device(device)
    num_layers = model.config.num_hidden_layers

    # stats: {layer: {"k": (mean, n), "v": (mean, n)}}
    stats = {i: {"k": (None, 0), "v": (None, 0)} for i in range(num_layers)}

    batch_count = 0
    for batch in loader:
        batch_count += 1
        input_ids = batch["input_ids"].to(dev)
        attn = batch["attention_mask"].to(dev)
        
        # Count actual (non-padding) tokens
        actual_tokens = attn.sum().item()
        total_tokens = attn.numel()
        
        print(f"  Batch {batch_count}: shape={input_ids.shape}, "
              f"actual_tokens={int(actual_tokens)}/{total_tokens} "
              f"({actual_tokens/total_tokens*100:.1f}% non-padding)")
        
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)

        pkv = getattr(out, "past_key_values", None)
        if pkv is None:  # some models require generate(...) for caches
            continue

        for i in range(min(num_layers, len(pkv))):
            k, v = pkv[i]  # shapes: [B, H, S, D]
            if batch_count == 1 and i == 0:
                print(f"  KV cache shape for layer 0: {k.shape} = [batch, heads, seq_len, head_dim]")
            
            # Reshape attention mask to match KV shape: [B, H, S, 1] for broadcasting
            # attn shape: [B, S], expand to [B, 1, S, 1] then broadcast to [B, H, S, 1]
            B, H, S, D = k.shape
            attn_expanded = attn.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
            attn_mask = attn_expanded.expand(B, H, S, 1)  # [B, H, S, 1]
            
            # Apply mask: zero out padding positions
            k_masked = k * attn_mask.float()
            v_masked = v * attn_mask.float()
            
            # Flatten and compute sums (padding positions are now zero)
            kf = k_masked.reshape(-1, D).float()  # [-1, D]
            vf = v_masked.reshape(-1, D).float()
            
            # Count actual (non-padding) tokens: sum over all dimensions except head_dim
            n = attn.sum().item() * H  # actual_tokens * num_heads
            
            ksum, vsum = kf.sum(0).cpu(), vf.sum(0).cpu()
            
            km, kn = stats[i]["k"]
            vm, vn = stats[i]["v"]
            km, kn = _running_mean_update(km, kn, ksum, n) if km is not None else (ksum / n, n)
            vm, vn = _running_mean_update(vm, vn, vsum, n) if vm is not None else (vsum / n, n)
            stats[i]["k"], stats[i]["v"] = (km, kn), (vm, vn)

    # finalize (drop layers with no stats)
    means = {}
    total_vectors = 0
    for i in range(num_layers):
        km, kn = stats[i]["k"]
        vm, vn = stats[i]["v"]
        if km is not None and vm is not None and kn > 0 and vn > 0:
            means[i] = {"k": km.contiguous(), "v": vm.contiguous()}
            total_vectors = max(total_vectors, kn)  # Use max across layers for summary
    
    if total_vectors > 0:
        print(f"  Total KV vectors averaged: ~{total_vectors:.0f} per layer (actual_tokens*heads, excluding padding)\n")
    else:
        print(f"  Warning: No valid KV statistics collected\n")
    return means


@torch.no_grad()
def collect_layer_similarity_direct(model, loader: DataLoader, 
                                    layer_i: int, layer_j: int,
                                    device: str = "cuda",
                                    metric: str = "cosine") -> float:
    """
    Direct similarity computation between two layers WITHOUT using means.
    Computes similarity for each token position, then averages.
    
    Returns average cosine similarity between layer_i and layer_j KV vectors.
    """
    model.eval()
    dev = next(model.parameters()).device if device == "auto" else torch.device(device)
    
    similarities = []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(dev)
        attn = batch["attention_mask"].to(dev)
        
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)
        pkv = getattr(out, "past_key_values", None)
        if pkv is None:
            continue
        
        if layer_i >= len(pkv) or layer_j >= len(pkv):
            continue
            
        ki, vi = pkv[layer_i]  # [B, H, S, D]
        kj, vj = pkv[layer_j]
        
        B, H, S, D = ki.shape
        
        # Reshape attention mask
        attn_expanded = attn.unsqueeze(1).unsqueeze(-1).expand(B, H, S, 1)
        attn_mask = attn_expanded.float()
        
        # Flatten: [B*H*S, D]
        ki_flat = ki.reshape(-1, D).float()
        vi_flat = vi.reshape(-1, D).float()
        kj_flat = kj.reshape(-1, D).float()
        vj_flat = vj.reshape(-1, D).float()
        mask_flat = attn_mask.reshape(-1, 1).float()
        
        # Mask out padding tokens
        ki_flat = ki_flat * mask_flat
        vi_flat = vi_flat * mask_flat
        kj_flat = kj_flat * mask_flat
        vj_flat = vj_flat * mask_flat
        
        # Compute similarity: concatenate k+v for each layer
        kv_i = torch.cat([ki_flat, vi_flat], dim=-1)  # [B*H*S, 2*D]
        kv_j = torch.cat([kj_flat, vj_flat], dim=-1)
        
        if metric == "cosine":
            # Normalize each vector
            kv_i_norm = F.normalize(kv_i, dim=-1)
            kv_j_norm = F.normalize(kv_j, dim=-1)
            # Cosine similarity per token
            sim_per_token = (kv_i_norm * kv_j_norm).sum(dim=-1)  # [B*H*S]
            # Only count non-padding tokens
            valid_mask = mask_flat.squeeze(-1) > 0
            if valid_mask.sum() > 0:
                avg_sim = sim_per_token[valid_mask].mean().item()
                similarities.append(avg_sim)
        else:
            raise ValueError(f"Direct similarity only supports 'cosine' metric")
    
    return float(np.mean(similarities)) if similarities else 0.0


def pairwise_distance_matrix(kv_means: Dict[int, Dict[str, torch.Tensor]],
                             num_layers: int,
                             metric: str = "euclidean") -> np.ndarray:
    """
    Build full [L, L] distance matrix from mean K/V vectors (concat(k, v)).
    Missing layers get large imputed distance to avoid accidental merges.
    """
    L = num_layers
    mat = np.zeros((L, L), dtype=np.float32)
    vecs = {}

    for i, kv in kv_means.items():
        vec = torch.cat([kv["k"], kv["v"]], dim=0)  # [2D]
        # normalize once for cosine distance to be stable
        if metric == "cosine":
            vec = F.normalize(vec, dim=0)
        vecs[i] = vec.cpu()

    # max penalty for missing entries
    big = 0.0
    # precompute for speed
    for i in range(L):
        for j in range(i + 1, L):
            if i in vecs and j in vecs:
                vi, vj = vecs[i], vecs[j]
                if metric == "euclidean":
                    d = torch.norm(vi - vj, p=2).item()
                elif metric == "manhattan":
                    d = torch.norm(vi - vj, p=1).item()
                elif metric == "cosine":
                    d = 1.0 - torch.sum(vi * vj).item()
                else:
                    raise ValueError(f"Unknown distance metric: {metric}")
                mat[i, j] = mat[j, i] = d
                big = max(big, d)
            else:
                mat[i, j] = mat[j, i] = np.nan

    # impute NaNs with a conservative big distance
    if np.isnan(mat).any():
        fill = big if big > 0 else 1.0
        mat = np.where(np.isnan(mat), fill, mat)
    return mat


def pairwise_distance_matrix_direct(model, loader: DataLoader,
                                     num_layers: int,
                                     metric: str = "cosine",
                                     device: str = "cuda") -> np.ndarray:
    """
    Build full [L, L] distance matrix using DIRECT similarity computation.
    Computes token-by-token similarity between layers (more accurate than means).
    
    NOTE: This is slower but more accurate than pairwise_distance_matrix().
    Only supports 'cosine' metric currently.
    """
    if metric != "cosine":
        raise ValueError(f"Direct similarity only supports 'cosine' metric, got '{metric}'")
    
    L = num_layers
    mat = np.zeros((L, L), dtype=np.float32)
    
    print(f"Computing direct similarity matrix ({L}x{L})...")
    print("This may take longer than mean-based approach, but is more accurate.\n")
    
    total_pairs = L * (L - 1) // 2
    computed = 0
    
    for i in range(L):
        for j in range(i + 1, L):
            computed += 1
            if computed % 10 == 0 or computed == total_pairs:
                print(f"  Progress: {computed}/{total_pairs} pairs ({computed*100//total_pairs}%)", end='\r')
            
            # Compute similarity (returns value between -1 and 1)
            sim = collect_layer_similarity_direct(model, loader, i, j, device=device, metric=metric)
            # Convert similarity to distance (1 - similarity for cosine)
            d = 1.0 - sim
            mat[i, j] = mat[j, i] = d
    
    print(f"\n  Completed: {computed}/{total_pairs} pairs computed")
    return mat


def rank_layer_pairs(dist: np.ndarray) -> List[Tuple[int, int]]:
    """Pairs ranked by **descending** distance (Algorithm 1)."""
    L = dist.shape[0]
    pairs = [ (i, j, dist[i, j]) for i in range(L) for j in range(i+1, L) ]
    pairs.sort(key=lambda x: x[2], reverse=False)
    return [(i, j) for i, j, _ in pairs]


# ------------------------------ Sharing & Eval --------------------------------

def apply_kv_sharing(model_class, base_config, model_name: str,
                     sharing_map: Dict[int, int],
                     device: str = "cuda",
                     torch_dtype=torch.float16):
    # Use model_class's config class (e.g., Qwen3SwiftKVConfig) to ensure SwiftKV parameters work
    config_class = getattr(model_class, 'config_class', base_config.__class__)
    cfg = config_class(**base_config.to_dict(),
                       swiftkv=True,
                       kv_sharing_map=sharing_map)
    return model_class.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch_dtype,
        device_map=resolve_device_map(device)
    )


@torch.no_grad()
def hidden_similarity(model_a, model_b, loader: DataLoader,
                      device: str = "cuda",
                      metric: str = "cosine") -> float:
    dev = next(model_a.parameters()).device if device == "auto" else torch.device(device)
    sims = []
    model_a.eval(); model_b.eval()
    for batch in loader:
        ids, attn = batch["input_ids"].to(dev), batch["attention_mask"].to(dev)
        oa = model_a(input_ids=ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
        ob = model_b(input_ids=ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
        ha = oa.hidden_states[-1].float().reshape(-1, oa.hidden_states[-1].shape[-1])
        hb = ob.hidden_states[-1].float().reshape(-1, ob.hidden_states[-1].shape[-1])

        if metric == "cosine":
            sims.append(F.cosine_similarity(ha, hb, dim=-1).mean().item())
        elif metric == "mse":
            sims.append(-F.mse_loss(ha, hb).item())  # higher is better
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    return float(np.mean(sims)) if sims else 0.0


# ------------------------------ Policies -------------------------------------

def choose_sharing_via_clustering(dist: np.ndarray,
                                  target_shared_layers: int,
                                  linkage_method: str = "average") -> Dict[int, int]:
    """
    Cluster layers, assign earliest index in each cluster as producer, others as consumers.
    """
    L = dist.shape[0]
    num_clusters = max(1, L - target_shared_layers)
    if linkage_method == "ward":
        # Ward assumes Euclidean distance
        # If the matrix didn't come from Euclidean, results may be invalid.
        pass

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, num_clusters, criterion="maxclust")  # 1..K

    clusters: Dict[int, List[int]] = {}
    for layer, cid in enumerate(labels):
        clusters.setdefault(cid, []).append(layer)

    sharing: Dict[int, int] = {}
    for cid, layers in clusters.items():
        layers.sort()
        if len(layers) <= 1:
            continue
        prod = layers[0]
        for cons in layers[1:]:
            sharing[cons] = prod
    return sharing


def choose_sharing_via_distance(dist: np.ndarray,
                                target_shared_layers: int) -> Dict[int, int]:
    """
    Direct distance-based selection: pick the most dissimilar pairs (non-intuitive but might work).
    No evaluation needed - purely based on distance metric.
    """
    L = dist.shape[0]
    sharing: Dict[int, int] = {}
    
    # Get all pairs ranked by distance (descending = most dissimilar first)
    all_pairs = rank_layer_pairs(dist)
    
    print(f"\nSelecting {target_shared_layers} most dissimilar pairs...")
    print("Pairs sorted by distance (most dissimilar first):\n")
    
    selected_count = 0
    for i, j in all_pairs:
        if selected_count >= target_shared_layers:
            break
            
        producer, consumer = (i, j) if i < j else (j, i)
        distance = dist[i, j]
        
        # Skip if consumer already assigned
        if consumer in sharing:
            continue
            
        # Skip if consumer is already producing for others
        if consumer in sharing.values():
            continue
            
        # Skip if producer is already consuming from another layer
        if producer in sharing:
            continue
        
        # Accept this pair
        sharing[consumer] = producer
        selected_count += 1
        print(f"  #{selected_count}: Layer {consumer} -> Layer {producer}  |  Distance: {distance:.6f}")
    
    print(f"\nSelected {len(sharing)} pairs (target was {target_shared_layers})")
    return sharing


def choose_sharing_via_greedy(dist: np.ndarray,
                              base_model, config, model_class, model_name,
                              loader: DataLoader,
                              device: str,
                              torch_dtype,
                              sim_thresh: float,
                              sim_metric: str,
                              top_k: int = 5) -> Dict[int, int]:
    """
    Adaptive best-first greedy search:
    1. Get top K pairs by distance
    2. Evaluate each against base model
    3. Pick the best one (highest similarity)
    4. Add to sharing map and repeat
    """
    sharing: Dict[int, int] = {}
    
    print(f"\nStarting adaptive best-first search with threshold: {sim_thresh}")
    print(f"Similarity metric: {sim_metric}")
    print(f"Top-K candidates per round: {top_k}\n")
    
    round_num = 0
    total_evaluated = 0
    
    while True:
        round_num += 1
        print("="*80)
        print(f"ROUND {round_num} - Current sharing: {len(sharing)} layers")
        print("="*80)
        
        # Get all valid pairs (not already used)
        all_pairs = rank_layer_pairs(dist)
        valid_pairs = []
        
        # Get set of layers that are already producers (cannot become consumers)
        existing_producers = set(sharing.values())
        
        for i, j in all_pairs:
            producer, consumer = (i, j) if i < j else (j, i)
            # Skip if:
            # 1. Consumer already consuming from another layer (is a key in sharing)
            # 2. Consumer already producing for another layer (is a value in sharing)
            # 3. Producer is consuming from another layer (is a key in sharing)
            # Note: A producer CAN produce for multiple consumers (no check for that)
            if (consumer not in sharing and 
                consumer not in existing_producers and
                producer not in sharing):
                valid_pairs.append((producer, consumer, dist[i, j]))
        
        if not valid_pairs:
            print("No more valid pairs to evaluate.\n")
            break
        
        # Take top K by distance
        candidates = valid_pairs[:min(top_k, len(valid_pairs))]
        
        print(f"\nTop {len(candidates)} candidate pairs by distance:")
        for idx, (prod, cons, d) in enumerate(candidates):
            print(f"  #{idx+1}: Layer {cons} -> Layer {prod}  |  Distance: {d:.6f}")
        print()
        
        # Evaluate each candidate
        best_pair = None
        best_sim = -float('inf')
        results = []
        
        for idx, (producer, consumer, d) in enumerate(candidates):
            total_evaluated += 1
            candidate_map = dict(sharing)
            candidate_map[consumer] = producer
            
            print(f"[{idx+1}/{len(candidates)}] Testing pair ({consumer} -> {producer}), distance={d:.6f}")
            
            cand_model = apply_kv_sharing(model_class, config, model_name, candidate_map,
                                          device=device, torch_dtype=torch_dtype)
            sim = hidden_similarity(base_model, cand_model, loader, device=device, metric=sim_metric)
            del cand_model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            print(f"    Similarity: {sim:.6f}")
            results.append((producer, consumer, d, sim))
            
            if sim > best_sim:
                best_sim = sim
                best_pair = (producer, consumer)
        
        # Show results sorted by similarity
        print("\n" + "-"*80)
        print("Results for this round (sorted by similarity):")
        results.sort(key=lambda x: x[3], reverse=True)
        for idx, (prod, cons, d, sim) in enumerate(results):
            marker = "★ BEST" if (prod, cons) == best_pair else ""
            status = "✓" if sim >= sim_thresh else "✗"
            print(f"  {status} ({cons} -> {prod}): sim={sim:.6f}, dist={d:.6f} {marker}")
        print("-"*80)
        
        # Accept best if it meets threshold
        if best_sim >= sim_thresh:
            producer, consumer = best_pair
            sharing[consumer] = producer
            print(f"\n✓ ACCEPTED: Layer {consumer} -> Layer {producer} with similarity {best_sim:.6f}")
            print(f"Updated sharing map: {sharing}\n")
        else:
            print(f"\n✗ STOPPED: Best similarity {best_sim:.6f} < threshold {sim_thresh:.6f}")
            print("No more pairs meet the threshold.\n")
            break
    
    print("="*80)
    print(f"Adaptive search completed:")
    print(f"  - Rounds: {round_num}")
    print(f"  - Total pairs evaluated: {total_evaluated}")
    print(f"  - Pairs accepted: {len(sharing)}")
    print(f"  - Final sharing map: {sharing}")
    print("="*80 + "\n")
    
    return sharing


# ------------------------------ Orchestrator ----------------------------------

def run_strategy_search(
    model_class,
    model_name: str,
    calibration_texts: List[str],
    target_shared_layers: int,
    similarity_threshold: float = 0.95,
    policy: str = "greedy",              # "greedy" (adaptive best-first), "cluster", or "distance" (most dissimilar)
    distance_metric: str = "euclidean",  # "euclidean" | "cosine" | "manhattan"
    use_direct_similarity: bool = False, # If True, use token-by-token similarity instead of means
    linkage_method: str = "average",     # for clustering
    top_k: int = 5,                      # for greedy: number of candidates per round
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda",
    load_dtype = torch.float16,
    sim_metric: str = "cosine",
    
) -> Tuple[Dict[int, int], Dict[str, float]]:
    """
    Returns (sharing_map, diagnostics)
    """
    assert target_shared_layers >= 0, "target_shared_layers must be >= 0"

    print("\n" + "="*80)
    print(f"SwiftKV Strategy Search")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Policy: {policy}")
    print(f"Target shared layers: {target_shared_layers}")
    print(f"Distance metric: {distance_metric}")
    print(f"Similarity method: {'DIRECT (token-by-token)' if use_direct_similarity else 'MEAN-BASED (faster)'}")
    if policy == "greedy":
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Similarity metric: {sim_metric}")
        print(f"Top-K candidates per round: {top_k}")
    elif policy == "distance":
        print(f"Strategy: Most dissimilar pairs (no evaluation)")
    print("="*80 + "\n")

    # tokenizer + data
    print("Loading tokenizer and preparing calibration data...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    loader = make_loader(calibration_texts, tok, batch_size, max_length)
    print(f"  - Calibration samples: {len(calibration_texts)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max length: {max_length}\n")

    # base model + config
    print("Loading base model...")
    base_cfg = AutoConfig.from_pretrained(model_name)
    base_model = model_class.from_pretrained(
        model_name,
        torch_dtype=load_dtype,
        device_map=resolve_device_map(device)
    )
    base_model.eval()
    L = base_model.config.num_hidden_layers
    print(f"  - Total layers: {L}\n")

    # Compute distance matrix using chosen method
    if use_direct_similarity:
        # Direct similarity: more accurate but slower
        if distance_metric != "cosine":
            print(f"Warning: Direct similarity only supports 'cosine' metric. Switching to 'cosine'.")
            distance_metric = "cosine"
        print("Computing pairwise distance matrix using DIRECT similarity...")
        dist = pairwise_distance_matrix_direct(base_model, loader, num_layers=L, 
                                               metric=distance_metric, device=device)
    else:
        # Mean-based: faster but less accurate
        print("Collecting KV statistics (mean-based)...")
        kv_means = collect_kv_means_streaming(base_model, loader, device=device)
        print(f"  - Collected statistics for {len(kv_means)} layers\n")
        print("Computing pairwise distance matrix from means...")
        dist = pairwise_distance_matrix(kv_means, num_layers=L, metric=distance_metric)
    
    print(f"  - Distance matrix shape: {dist.shape}")
    print(f"  - Distance range: [{dist.min():.6f}, {dist.max():.6f}]\n")

    # choose policy
    if policy == "cluster":
        sharing = choose_sharing_via_clustering(dist, target_shared_layers, linkage_method)
        # Optional: if clustering yields more/less consumers than target, that's fine; we report actual.
    elif policy == "greedy":
        sharing = choose_sharing_via_greedy(dist,
                                            base_model, base_cfg, model_class, model_name,
                                            loader, device, load_dtype,
                                            similarity_threshold, sim_metric, top_k)
    elif policy == "distance":
        sharing = choose_sharing_via_distance(dist, target_shared_layers)
    else:
        raise ValueError("policy must be 'greedy', 'cluster', or 'distance'")

    # evaluate final strategy (both paths)
    print("Evaluating final strategy...")
    cand = apply_kv_sharing(model_class, base_cfg, model_name, sharing,
                            device=device, torch_dtype=load_dtype)
    sim = hidden_similarity(base_model, cand, loader, device=device, metric=sim_metric)
    del cand
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    diagnostics = {
        "similarity": float(sim),
        "layers_total": float(L),
        "layers_shared": float(len(sharing)),
        "memory_reduction_pct": (len(sharing) / L * 100.0) if L > 0 else 0.0
    }
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Sharing map: {sharing}")
    print(f"Total layers: {L}")
    print(f"Shared layers: {len(sharing)}")
    print(f"Producer layers: {L - len(sharing)}")
    print(f"Memory reduction: {diagnostics['memory_reduction_pct']:.2f}%")
    print(f"Final similarity: {sim:.6f}")
    print("="*80 + "\n")
    
    return sharing, diagnostics


# ------------------------------ CLI -------------------------------------------

def get_model_class(model_type: str):
    """Get the SwiftKV model class based on model type."""
    model_type = model_type.lower()
    if model_type == "qwen3":
        from projects.swiftkv.models.qwen3 import Qwen3SwiftKVForCausalLM
        return Qwen3SwiftKVForCausalLM
    elif model_type == "qwen2":
        from projects.swiftkv.models.qwen2 import Qwen2SwiftKVForCausalLM
        return Qwen2SwiftKVForCausalLM
    elif model_type == "llama":
        from projects.swiftkv.models.llama import LlamaSwiftKVForCausalLM
        return LlamaSwiftKVForCausalLM
    elif model_type == "deepseek_v2" or model_type == "deepseek":
        from projects.swiftkv.models.deepseek_v2 import DeepseekV2SwiftKVForCausalLM
        return DeepseekV2SwiftKVForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: qwen3, qwen2, llama, deepseek_v2")


def load_calibration_texts(source: str, num_samples: int = 50) -> List[str]:
    """Load calibration texts from a file or dataset."""
    if source.startswith("datasets:"):
        # Load from HuggingFace datasets
        # Format: "datasets:dataset_name:config:split" or "datasets:dataset_name:split"
        from datasets import load_dataset
        parts = source[9:].split(":")  # Remove "datasets:" prefix
        
        if len(parts) == 2:
            dataset_name, split = parts
            config = None
        elif len(parts) == 3:
            dataset_name, config, split = parts
        else:
            raise ValueError(f"Invalid dataset format: {source}. Use 'datasets:dataset_name:split' or 'datasets:dataset_name:config:split'")
        
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        texts = []
        for sample in dataset:
            text = sample.get("text", "").strip()
            if len(text) > 100 and not text.startswith("="):
                texts.append(text)
            if len(texts) >= num_samples:
                break
        return texts
    else:
        # Load from file (one text per line or JSON)
        if source.endswith(".json"):
            with open(source, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [str(item) for item in data[:num_samples]]
                elif isinstance(data, dict) and "texts" in data:
                    texts = data["texts"][:num_samples]
                else:
                    raise ValueError(f"JSON file must contain a list or dict with 'texts' key")
        else:
            with open(source, "r") as f:
                texts = [line.strip() for line in f if line.strip()][:num_samples]
        return texts


def main():
    parser = argparse.ArgumentParser(
        description="SwiftKV Strategy Search - Find optimal KV sharing map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["qwen3", "qwen2", "llama", "deepseek_v2"],
        help="Type of model architecture"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen3-8B')"
    )
    
    # Data arguments
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="Path to calibration data file (JSON or text) or 'datasets:dataset_name:config:split' (e.g., 'datasets:wikitext:wikitext-2-raw-v1:train')"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of calibration samples to use"
    )
    
    # Search arguments
    parser.add_argument(
        "--target_shared_layers",
        type=int,
        required=True,
        help="Target number of layers to share KV caches"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="distance",
        choices=["distance", "greedy", "cluster"],
        help="Search policy: 'distance' (most dissimilar pairs, fastest), 'greedy' (best-first with evaluation), 'cluster' (hierarchical clustering)"
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "manhattan"],
        help="Distance metric for computing layer similarity"
    )
    parser.add_argument(
        "--use_direct_similarity",
        action="store_true",
        help="Use direct token-by-token similarity instead of mean-based approach (more accurate but slower). Only supports cosine metric."
    )
    
    # Greedy policy arguments
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for greedy policy (only used with --policy greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K candidates per round for greedy policy (only used with --policy greedy)"
    )
    parser.add_argument(
        "--sim_metric",
        type=str,
        default="cosine",
        choices=["cosine", "mse"],
        help="Similarity metric for evaluation (only used with --policy greedy)"
    )
    
    # Cluster policy arguments
    parser.add_argument(
        "--linkage_method",
        type=str,
        default="average",
        choices=["average", "ward", "complete", "single"],
        help="Linkage method for clustering (only used with --policy cluster)"
    )
    
    # Training/Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for model loading"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON (sharing map and diagnostics)"
    )
    
    args = parser.parse_args()
    
    # Convert torch_dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Get model class
    model_class = get_model_class(args.model_type)
    
    # Load calibration texts
    print(f"Loading calibration data from: {args.calibration_data}")
    calibration_texts = load_calibration_texts(args.calibration_data, args.num_samples)
    print(f"Loaded {len(calibration_texts)} calibration samples\n")
    
    # Run strategy search
    sharing_map, diagnostics = run_strategy_search(
        model_class=model_class,
        model_name=args.model_name,
        calibration_texts=calibration_texts,
        target_shared_layers=args.target_shared_layers,
        similarity_threshold=args.similarity_threshold,
        policy=args.policy,
        distance_metric=args.distance_metric,
        use_direct_similarity=args.use_direct_similarity,
        linkage_method=args.linkage_method,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        load_dtype=torch_dtype,
        sim_metric=args.sim_metric,
    )
    
    # Save results if output path provided
    if args.output:
        results = {
            "sharing_map": sharing_map,
            "diagnostics": diagnostics,
            "config": {
                "model_type": args.model_type,
                "model_name": args.model_name,
                "policy": args.policy,
                "target_shared_layers": args.target_shared_layers,
                "distance_metric": args.distance_metric,
            }
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
