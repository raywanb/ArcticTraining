#!/usr/bin/env python3
"""
Validate a SwiftKV sharing strategy by measuring quality and efficiency.

This script evaluates:
1. Output similarity between original and SwiftKV models
2. Memory usage reduction
3. Inference speed comparison
4. Per-layer KV cache statistics
"""

import torch
import time
import json
import argparse
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import torch.nn.functional as F


def measure_memory_usage(model, tokenizer, prompt: str, max_tokens: int = 100):
    """
    Measure peak memory usage during generation.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        prompt: Test prompt
        max_tokens: Number of tokens to generate
    
    Returns:
        Peak memory in GB
    """
    device = next(model.parameters()).device
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get peak memory
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        peak_memory = 0  # Can't measure CPU memory reliably
    
    return peak_memory


def measure_inference_speed(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 50,
    num_runs: int = 3
):
    """
    Measure average inference speed.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        prompts: List of test prompts
        max_tokens: Tokens to generate per prompt
        num_runs: Number of runs for averaging
    
    Returns:
        Dict with timing statistics
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warm up
    inputs = tokenizer(prompts[0], return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, use_cache=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    total_tokens = 0
    
    for run in range(num_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            total_tokens += outputs.shape[1] - inputs['input_ids'].shape[1]
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_tokens': total_tokens,
        'tokens_per_second': total_tokens / sum(times)
    }


def measure_output_quality(
    model_original,
    model_swiftkv,
    tokenizer,
    test_prompts: List[str],
    max_tokens: int = 100
):
    """
    Compare output quality between original and SwiftKV models.
    
    Args:
        model_original: Original model
        model_swiftkv: SwiftKV model
        tokenizer: Tokenizer
        test_prompts: Test prompts
        max_tokens: Tokens to generate
    
    Returns:
        Dict with quality metrics
    """
    device = next(model_original.parameters()).device
    model_original.eval()
    model_swiftkv.eval()
    
    cosine_similarities = []
    exact_matches = 0
    token_overlap_ratios = []
    
    print("\nComparing outputs on test prompts:")
    print("-" * 80)
    
    for prompt in tqdm(test_prompts, desc="Testing prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate from both models
        with torch.no_grad():
            # Original model - get hidden states
            outputs_orig = model_original(
                **inputs,
                output_hidden_states=True,
                use_cache=True
            )
            
            # SwiftKV model - get hidden states
            outputs_swift = model_swiftkv(
                **inputs,
                output_hidden_states=True,
                use_cache=True
            )
            
            # Compare final hidden states
            hidden_orig = outputs_orig.hidden_states[-1]
            hidden_swift = outputs_swift.hidden_states[-1]
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(
                hidden_orig.reshape(-1, hidden_orig.shape[-1]),
                hidden_swift.reshape(-1, hidden_swift.shape[-1]),
                dim=-1
            ).mean().item()
            cosine_similarities.append(cos_sim)
            
            # Also compare generated tokens
            gen_orig = model_original.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_swift = model_swiftkv.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Token-level comparison
            if gen_orig.shape == gen_swift.shape and torch.all(gen_orig == gen_swift):
                exact_matches += 1
            
            # Token overlap ratio
            orig_tokens = set(gen_orig[0].cpu().tolist())
            swift_tokens = set(gen_swift[0].cpu().tolist())
            overlap = len(orig_tokens & swift_tokens) / len(orig_tokens | swift_tokens)
            token_overlap_ratios.append(overlap)
    
    return {
        'mean_cosine_similarity': np.mean(cosine_similarities),
        'std_cosine_similarity': np.std(cosine_similarities),
        'min_cosine_similarity': np.min(cosine_similarities),
        'exact_match_ratio': exact_matches / len(test_prompts),
        'mean_token_overlap': np.mean(token_overlap_ratios),
        'num_test_prompts': len(test_prompts)
    }


def analyze_kv_sharing_pattern(model, sharing_map: Dict[int, int]):
    """
    Analyze the KV sharing pattern in a model.
    
    Args:
        model: SwiftKV model
        sharing_map: Dictionary mapping consumer -> producer layers
    
    Returns:
        Dict with sharing statistics
    """
    num_layers = model.config.num_hidden_layers
    
    # Group by producer
    producer_to_consumers = {}
    for consumer, producer in sharing_map.items():
        if producer not in producer_to_consumers:
            producer_to_consumers[producer] = []
        producer_to_consumers[producer].append(consumer)
    
    # Compute statistics
    num_producers = num_layers - len(sharing_map)
    num_consumers = len(sharing_map)
    sharing_groups = len(producer_to_consumers)
    avg_consumers_per_producer = (
        np.mean([len(consumers) for consumers in producer_to_consumers.values()])
        if producer_to_consumers else 0
    )
    
    return {
        'num_layers': num_layers,
        'num_kv_computing_layers': num_producers,
        'num_kv_sharing_layers': num_consumers,
        'num_sharing_groups': sharing_groups,
        'avg_consumers_per_producer': avg_consumers_per_producer,
        'memory_reduction_ratio': num_consumers / num_layers,
        'producer_to_consumers': producer_to_consumers
    }


def validate_strategy(
    model_class,
    model_name: str,
    sharing_strategy: Dict[int, int],
    test_prompts: List[str],
    device: str = 'auto',
    torch_dtype = torch.float16,
    max_tokens: int = 100
):
    """
    Comprehensive validation of a KV sharing strategy.
    
    Args:
        model_class: SwiftKV model class
        model_name: Model name or path
        sharing_strategy: KV sharing map
        test_prompts: List of test prompts
        device: Device to use
        torch_dtype: Model precision
        max_tokens: Tokens to generate for benchmarks
    
    Returns:
        Dict with validation results
    """
    print("=" * 80)
    print("SWIFTKV STRATEGY VALIDATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Sharing strategy: {len(sharing_strategy)} layers share KV cache")
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original config
    print("\n[2/6] Loading original model...")
    original_config = AutoConfig.from_pretrained(model_name)
    
    model_original = model_class.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device
    )
    model_original.eval()
    
    # Load SwiftKV model
    print("\n[3/6] Loading SwiftKV model with sharing strategy...")
    swiftkv_config = model_class.config_class(
        **original_config.to_dict(),
        swiftkv=True,
        kv_sharing_map=sharing_strategy
    )
    
    model_swiftkv = model_class.from_pretrained(
        model_name,
        config=swiftkv_config,
        torch_dtype=torch_dtype,
        device_map=device
    )
    model_swiftkv.eval()
    
    # Analyze sharing pattern
    print("\n[4/6] Analyzing KV sharing pattern...")
    sharing_analysis = analyze_kv_sharing_pattern(model_swiftkv, sharing_strategy)
    
    print(f"  Total layers: {sharing_analysis['num_layers']}")
    print(f"  KV-computing layers: {sharing_analysis['num_kv_computing_layers']}")
    print(f"  KV-sharing layers: {sharing_analysis['num_kv_sharing_layers']}")
    print(f"  Memory reduction: {sharing_analysis['memory_reduction_ratio']*100:.1f}%")
    
    # Measure memory usage
    print("\n[5/6] Benchmarking memory usage...")
    test_prompt = test_prompts[0] if test_prompts else "The future of AI is"
    
    memory_original = measure_memory_usage(model_original, tokenizer, test_prompt, max_tokens)
    memory_swiftkv = measure_memory_usage(model_swiftkv, tokenizer, test_prompt, max_tokens)
    memory_savings = (memory_original - memory_swiftkv) / memory_original * 100
    
    print(f"  Original model peak memory: {memory_original:.2f} GB")
    print(f"  SwiftKV model peak memory: {memory_swiftkv:.2f} GB")
    print(f"  Memory savings: {memory_savings:.1f}%")
    
    # Measure inference speed
    print("\n[6/6] Benchmarking inference speed...")
    speed_original = measure_inference_speed(model_original, tokenizer, test_prompts[:3], max_tokens=50)
    speed_swiftkv = measure_inference_speed(model_swiftkv, tokenizer, test_prompts[:3], max_tokens=50)
    
    speedup = speed_swiftkv['tokens_per_second'] / speed_original['tokens_per_second']
    
    print(f"  Original model: {speed_original['tokens_per_second']:.2f} tokens/sec")
    print(f"  SwiftKV model: {speed_swiftkv['tokens_per_second']:.2f} tokens/sec")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Measure output quality
    print("\n[7/7] Measuring output quality...")
    quality_metrics = measure_output_quality(
        model_original, model_swiftkv, tokenizer, test_prompts, max_tokens=max_tokens
    )
    
    print(f"  Mean cosine similarity: {quality_metrics['mean_cosine_similarity']:.6f}")
    print(f"  Exact match ratio: {quality_metrics['exact_match_ratio']*100:.1f}%")
    print(f"  Token overlap: {quality_metrics['mean_token_overlap']*100:.1f}%")
    
    # Compile results
    results = {
        'model': model_name,
        'sharing_strategy': sharing_strategy,
        'sharing_analysis': sharing_analysis,
        'memory': {
            'original_gb': memory_original,
            'swiftkv_gb': memory_swiftkv,
            'savings_percent': memory_savings
        },
        'speed': {
            'original_tokens_per_sec': speed_original['tokens_per_second'],
            'swiftkv_tokens_per_sec': speed_swiftkv['tokens_per_second'],
            'speedup': speedup
        },
        'quality': quality_metrics
    }
    
    return results


def print_validation_report(results: Dict, output_file: str = None):
    """
    Print a comprehensive validation report.
    
    Args:
        results: Validation results dictionary
        output_file: Optional file to save report
    """
    report = []
    report.append("=" * 80)
    report.append("SWIFTKV VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"\nModel: {results['model']}")
    report.append(f"Sharing Strategy: {results['sharing_strategy']}")
    
    report.append("\n" + "-" * 80)
    report.append("SHARING PATTERN ANALYSIS")
    report.append("-" * 80)
    sa = results['sharing_analysis']
    report.append(f"Total layers: {sa['num_layers']}")
    report.append(f"KV-computing layers: {sa['num_kv_computing_layers']}")
    report.append(f"KV-sharing layers: {sa['num_kv_sharing_layers']}")
    report.append(f"Sharing groups: {sa['num_sharing_groups']}")
    report.append(f"Memory reduction: {sa['memory_reduction_ratio']*100:.1f}%")
    
    report.append("\nSharing groups:")
    for producer, consumers in sa['producer_to_consumers'].items():
        report.append(f"  Layer {producer} -> Layers {consumers}")
    
    report.append("\n" + "-" * 80)
    report.append("MEMORY USAGE")
    report.append("-" * 80)
    mem = results['memory']
    report.append(f"Original model: {mem['original_gb']:.2f} GB")
    report.append(f"SwiftKV model: {mem['swiftkv_gb']:.2f} GB")
    report.append(f"Savings: {mem['savings_percent']:.1f}%")
    
    report.append("\n" + "-" * 80)
    report.append("INFERENCE SPEED")
    report.append("-" * 80)
    speed = results['speed']
    report.append(f"Original model: {speed['original_tokens_per_sec']:.2f} tokens/sec")
    report.append(f"SwiftKV model: {speed['swiftkv_tokens_per_sec']:.2f} tokens/sec")
    report.append(f"Speedup: {speed['speedup']:.2f}x")
    
    report.append("\n" + "-" * 80)
    report.append("OUTPUT QUALITY")
    report.append("-" * 80)
    qual = results['quality']
    report.append(f"Mean cosine similarity: {qual['mean_cosine_similarity']:.6f}")
    report.append(f"Min cosine similarity: {qual['min_cosine_similarity']:.6f}")
    report.append(f"Std cosine similarity: {qual['std_cosine_similarity']:.6f}")
    report.append(f"Exact match ratio: {qual['exact_match_ratio']*100:.1f}%")
    report.append(f"Token overlap: {qual['mean_token_overlap']*100:.1f}%")
    
    report.append("\n" + "=" * 80)
    report.append("OVERALL ASSESSMENT")
    report.append("=" * 80)
    
    # Quality check
    if qual['mean_cosine_similarity'] >= 0.95:
        quality_status = "✅ EXCELLENT - High similarity maintained"
    elif qual['mean_cosine_similarity'] >= 0.90:
        quality_status = "⚠️  GOOD - Acceptable similarity"
    else:
        quality_status = "❌ POOR - Consider adjusting strategy"
    
    # Memory check
    if mem['savings_percent'] >= 15:
        memory_status = "✅ EXCELLENT - Significant memory reduction"
    elif mem['savings_percent'] >= 10:
        memory_status = "✅ GOOD - Moderate memory reduction"
    else:
        memory_status = "⚠️  MODEST - Limited memory savings"
    
    # Speed check
    if speed['speedup'] >= 1.1:
        speed_status = "✅ FASTER - Performance improved"
    elif speed['speedup'] >= 0.95:
        speed_status = "✅ SIMILAR - Comparable performance"
    else:
        speed_status = "⚠️  SLOWER - Some performance overhead"
    
    report.append(f"\nQuality: {quality_status}")
    report.append(f"Memory: {memory_status}")
    report.append(f"Speed: {speed_status}")
    report.append("\n" + "=" * 80)
    
    # Print to console
    full_report = "\n".join(report)
    print(full_report)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"\n✅ Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate SwiftKV sharing strategy")
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name or path'
    )
    parser.add_argument(
        '--model_class',
        type=str,
        default='qwen3',
        choices=['qwen3', 'llama', 'qwen2'],
        help='Model architecture'
    )
    parser.add_argument(
        '--strategy_file',
        type=str,
        required=True,
        help='JSON file with sharing strategy'
    )
    parser.add_argument(
        '--output_report',
        type=str,
        default='validation_report.txt',
        help='Output file for validation report'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='validation_results.json',
        help='Output file for JSON results'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=100,
        help='Max tokens to generate for testing'
    )
    
    args = parser.parse_args()
    
    # Load strategy
    print(f"Loading sharing strategy from {args.strategy_file}...")
    with open(args.strategy_file, 'r') as f:
        strategy_data = json.load(f)
    
    # Convert string keys to int if needed
    if 'kv_sharing_map' in strategy_data:
        sharing_strategy = {
            int(k): int(v) for k, v in strategy_data['kv_sharing_map'].items()
        }
    else:
        sharing_strategy = {int(k): int(v) for k, v in strategy_data.items()}
    
    # Import model class
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
        raise ValueError(f"Unknown model class: {args.model_class}")
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "Climate change is one of the most",
        "The key to successful software development",
        "Natural language processing enables"
    ]
    
    # Run validation
    results = validate_strategy(
        model_class=model_class,
        model_name=args.model_name,
        sharing_strategy=sharing_strategy,
        test_prompts=test_prompts,
        device='auto',
        torch_dtype=torch.float16,
        max_tokens=args.max_tokens
    )
    
    # Print report
    print_validation_report(results, output_file=args.output_report)
    
    # Save JSON results
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

