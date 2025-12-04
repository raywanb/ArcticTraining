#!/usr/bin/env python3
"""
Analyze and visualize KV cache distances between layers.

This script computes pairwise distances between all layers' KV caches
and provides visualizations to understand which layers are most similar.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import json
import argparse
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from projects.swiftkv.strategy_search import (
    CalibrationDataset,
    collect_kv_means_streaming,
    pairwise_distance_matrix,
    rank_layer_pairs,
    make_loader
)


def create_distance_matrix(
    distances: Dict[Tuple[int, int], float],
    num_layers: int
) -> np.ndarray:
    """
    Convert pairwise distances dict to a full distance matrix.
    
    Args:
        distances: Dictionary of pairwise distances
        num_layers: Total number of layers
    
    Returns:
        Symmetric distance matrix of shape [num_layers, num_layers]
    """
    matrix = np.zeros((num_layers, num_layers))
    
    for (i, j), dist in distances.items():
        matrix[i, j] = dist
        matrix[j, i] = dist  # Symmetric
    
    return matrix


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    output_file: str = "kv_distance_heatmap.png",
    title: str = "KV Cache Distance Between Layers"
):
    """
    Create a heatmap visualization of layer distances.
    
    Args:
        distance_matrix: Distance matrix [num_layers, num_layers]
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Use a diverging colormap where high distance = red, low = blue
    sns.heatmap(
        distance_matrix,
        cmap='RdYlBu_r',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Distance'},
        xticklabels=5,
        yticklabels=5
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to {output_file}")
    plt.close()


def plot_distance_distribution(
    distances: Dict[Tuple[int, int], float],
    output_file: str = "distance_distribution.png"
):
    """
    Plot distribution of pairwise distances.
    
    Args:
        distances: Dictionary of pairwise distances
        output_file: Path to save the plot
    """
    dist_values = list(distances.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(dist_values, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(dist_values), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(dist_values):.4f}')
    plt.axvline(np.median(dist_values), color='g', linestyle='--',
                linewidth=2, label=f'Median: {np.median(dist_values):.4f}')
    
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of KV Cache Distances', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Distribution plot saved to {output_file}")
    plt.close()


def plot_layer_similarity_network(
    distances: Dict[Tuple[int, int], float],
    num_layers: int,
    top_k: int = 50,
    output_file: str = "similarity_network.png"
):
    """
    Plot top-k most similar layer pairs as a network.
    
    Args:
        distances: Dictionary of pairwise distances
        num_layers: Total number of layers
        top_k: Number of top similar pairs to show
        output_file: Path to save the plot
    """
    # Sort by distance (ascending for most similar)
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1])[:top_k]
    
    plt.figure(figsize=(14, 10))
    
    # Create positions for layers in a circle
    angles = np.linspace(0, 2*np.pi, num_layers, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Normalize distances for line width
    max_dist = max(dist for _, dist in sorted_pairs)
    
    # Draw connections
    for (i, j), dist in sorted_pairs:
        # Line width inversely proportional to distance
        linewidth = 2 * (1 - dist / max_dist)
        alpha = 0.3 * (1 - dist / max_dist)
        plt.plot([x[i], x[j]], [y[i], y[j]], 
                'b-', linewidth=linewidth, alpha=alpha)
    
    # Draw layer nodes
    plt.scatter(x, y, s=100, c='red', zorder=5, edgecolors='black', linewidths=1)
    
    # Label layers
    for i in range(num_layers):
        if i % 2 == 0:  # Label every other layer to avoid clutter
            offset = 1.1
            plt.text(x[i]*offset, y[i]*offset, str(i), 
                    ha='center', va='center', fontsize=8)
    
    plt.title(f'Top {top_k} Most Similar Layer Pairs', 
             fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Network plot saved to {output_file}")
    plt.close()


def find_optimal_sharing_candidates(
    distances: Dict[Tuple[int, int], float],
    num_layers: int,
    top_k: int = 10
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Find the most promising layer pairs for KV sharing.
    
    Layers that are far apart should NOT share KV (they're too different).
    Layers that are close should potentially share KV (they're similar).
    
    Args:
        distances: Dictionary of pairwise distances
        num_layers: Total number of layers
        top_k: Number of top candidates to return
    
    Returns:
        List of (layer_pair, distance) tuples, sorted by distance (ascending)
    """
    # Sort by ascending distance (most similar first)
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1])
    
    return sorted_pairs[:top_k]


def analyze_layer_neighborhoods(
    distance_matrix: np.ndarray,
    layer_idx: int,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Find the k nearest neighbors for a specific layer.
    
    Args:
        distance_matrix: Full distance matrix
        layer_idx: Index of layer to analyze
        top_k: Number of neighbors to return
    
    Returns:
        List of (neighbor_layer, distance) tuples
    """
    distances_from_layer = distance_matrix[layer_idx]
    
    # Get indices of k smallest distances (excluding self)
    neighbor_indices = np.argsort(distances_from_layer)[1:top_k+1]
    
    neighbors = [
        (int(idx), float(distances_from_layer[idx]))
        for idx in neighbor_indices
    ]
    
    return neighbors


def generate_analysis_report(
    distances: Dict[Tuple[int, int], float],
    num_layers: int,
    output_file: str = "kv_analysis_report.txt"
):
    """
    Generate a comprehensive text report of the analysis.
    
    Args:
        distances: Dictionary of pairwise distances
        num_layers: Total number of layers
        output_file: Path to save the report
    """
    dist_values = list(distances.values())
    distance_matrix = create_distance_matrix(distances, num_layers)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("KV CACHE DISTANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Total layer pairs: {len(distances)}\n")
        f.write(f"Distance range: [{min(dist_values):.6f}, {max(dist_values):.6f}]\n")
        f.write(f"Mean distance: {np.mean(dist_values):.6f}\n")
        f.write(f"Median distance: {np.median(dist_values):.6f}\n")
        f.write(f"Std deviation: {np.std(dist_values):.6f}\n\n")
        
        # Top 20 most similar pairs (candidates for sharing)
        f.write("TOP 20 MOST SIMILAR LAYER PAIRS (Best candidates for KV sharing)\n")
        f.write("-" * 80 + "\n")
        most_similar = sorted(distances.items(), key=lambda x: x[1])[:20]
        for rank, ((i, j), dist) in enumerate(most_similar, 1):
            f.write(f"{rank:2d}. Layers {i:2d} <-> {j:2d}: distance = {dist:.6f}\n")
        f.write("\n")
        
        # Top 20 most distant pairs (should NOT share)
        f.write("TOP 20 MOST DISTANT LAYER PAIRS (Should NOT share KV)\n")
        f.write("-" * 80 + "\n")
        most_distant = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:20]
        for rank, ((i, j), dist) in enumerate(most_distant, 1):
            f.write(f"{rank:2d}. Layers {i:2d} <-> {j:2d}: distance = {dist:.6f}\n")
        f.write("\n")
        
        # Per-layer neighborhood analysis
        f.write("PER-LAYER NEIGHBORHOOD ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for layer_idx in range(num_layers):
            neighbors = analyze_layer_neighborhoods(distance_matrix, layer_idx, top_k=3)
            f.write(f"\nLayer {layer_idx:2d} - 3 nearest neighbors:\n")
            for neighbor_idx, dist in neighbors:
                f.write(f"  -> Layer {neighbor_idx:2d}: distance = {dist:.6f}\n")
        f.write("\n")
        
        # Sharing recommendations
        f.write("SHARING RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("Based on similarity analysis, recommended sharing strategies:\n\n")
        
        # Conservative strategy: only very similar layers
        threshold_conservative = np.percentile(dist_values, 10)
        conservative_pairs = [(pair, dist) for pair, dist in distances.items() 
                             if dist <= threshold_conservative]
        f.write(f"Conservative (top 10% similar pairs, threshold={threshold_conservative:.6f}):\n")
        f.write(f"  Could share: {len(conservative_pairs)} layer pairs\n")
        f.write(f"  Potential memory reduction: ~{len(conservative_pairs)}/{num_layers} = "
                f"{len(conservative_pairs)/num_layers*100:.1f}%\n\n")
        
        # Moderate strategy
        threshold_moderate = np.percentile(dist_values, 25)
        moderate_pairs = [(pair, dist) for pair, dist in distances.items() 
                         if dist <= threshold_moderate]
        f.write(f"Moderate (top 25% similar pairs, threshold={threshold_moderate:.6f}):\n")
        f.write(f"  Could share: {len(moderate_pairs)} layer pairs\n")
        f.write(f"  Potential memory reduction: ~{len(moderate_pairs)}/{num_layers} = "
                f"{len(moderate_pairs)/num_layers*100:.1f}%\n\n")
        
        # Aggressive strategy
        threshold_aggressive = np.percentile(dist_values, 50)
        aggressive_pairs = [(pair, dist) for pair, dist in distances.items() 
                           if dist <= threshold_aggressive]
        f.write(f"Aggressive (top 50% similar pairs, threshold={threshold_aggressive:.6f}):\n")
        f.write(f"  Could share: {len(aggressive_pairs)} layer pairs\n")
        f.write(f"  Potential memory reduction: ~{len(aggressive_pairs)}/{num_layers} = "
                f"{len(aggressive_pairs)/num_layers*100:.1f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Note: These are preliminary recommendations based on KV cache similarity.\n")
        f.write("Run full strategy_search with quality validation for production use.\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Analysis report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KV cache distances between layers"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen3-8B',
        help='Model name or path'
    )
    parser.add_argument(
        '--model_class',
        type=str,
        default='qwen3',
        choices=['qwen3', 'llama', 'qwen2', 'deepseek'],
        help='Model architecture'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--distance_metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine', 'manhattan'],
        help='Distance metric'
    )
    
    args = parser.parse_args()
    
    # Import appropriate model class
    if args.model_class == 'qwen3':
        from projects.swiftkv.models.qwen3.modeling_qwen3_swiftkv import Qwen3SwiftKVForCausalLM
        model_class = Qwen3SwiftKVForCausalLM
    elif args.model_class == 'llama':
        from projects.swiftkv.models.llama.modeling_llama_swiftkv import LlamaSwiftKVForCausalLM
        model_class = LlamaSwiftKVForCausalLM
    elif args.model_class == 'qwen2':
        from projects.swiftkv.models.qwen2.modeling_qwen2_swiftkv import Qwen2SwiftKVForCausalLM
        model_class = Qwen2SwiftKVForCausalLM
    else:
        raise ValueError(f"Unsupported model class: {args.model_class}")
    
    print("=" * 80)
    print("KV CACHE DISTANCE ANALYSIS")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Calibration samples: {args.num_samples}")
    print(f"Distance metric: {args.distance_metric}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/6] Loading model and tokenizer...")
    model = model_class.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # Prepare calibration data
    print("\n[2/6] Preparing calibration data...")
    calibration_texts = [
        "The future of artificial intelligence lies in",
        "Once upon a time in a distant galaxy",
        "Climate change is one of the most pressing",
        "Quantum computing has the potential to",
        "The key to successful software development",
        "Natural language processing enables computers",
        "Machine learning algorithms can learn from",
        "The history of human civilization shows",
        "Modern transportation systems rely on",
        "Advances in renewable energy have made",
        "The human brain is incredibly complex",
        "Economic theories help us understand markets",
        "DNA contains the genetic instructions",
        "The internet has revolutionized communication",
        "Space exploration continues to reveal",
        "Educational systems around the world",
        "Neural networks are inspired by biology",
        "Mathematics is the language of science",
        "Cultural diversity enriches our society",
        "Medical research has led to breakthroughs"
    ][:args.num_samples]
    
    dataloader = make_loader(calibration_texts, tokenizer, args.batch_size, args.max_length)
    
    # Compute KV means
    print("\n[3/6] Computing KV cache statistics...")
    kv_means = collect_kv_means_streaming(model, dataloader, device='cuda')
    
    # Create distance matrix
    print("\n[4/6] Computing pairwise distance matrix...")
    distance_matrix = pairwise_distance_matrix(kv_means, num_layers, metric=args.distance_metric)
    
    # Convert to distances dict for compatibility with visualization functions
    distances = {}
    for i in range(num_layers):
        for j in range(i+1, num_layers):
            distances[(i, j)] = distance_matrix[i, j]
    
    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_distance_heatmap(
        distance_matrix,
        os.path.join(args.output_dir, "kv_distance_heatmap.png")
    )
    
    plot_distance_distribution(
        distances,
        os.path.join(args.output_dir, "distance_distribution.png")
    )
    
    plot_layer_similarity_network(
        distances,
        num_layers,
        top_k=min(50, len(distances) // 2),
        output_file=os.path.join(args.output_dir, "similarity_network.png")
    )
    
    # Generate report
    print("\n[6/6] Generating analysis report...")
    generate_analysis_report(
        distances,
        num_layers,
        os.path.join(args.output_dir, "kv_analysis_report.txt")
    )
    
    # Save distances to JSON
    distances_json = {
        f"{i}_{j}": dist for (i, j), dist in distances.items()
    }
    with open(os.path.join(args.output_dir, "kv_distances.json"), 'w') as f:
        json.dump({
            'model': args.model_name,
            'num_layers': num_layers,
            'distance_metric': args.distance_metric,
            'distances': distances_json
        }, f, indent=2)
    print(f"✅ Distances saved to kv_distances.json")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

