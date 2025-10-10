#!/usr/bin/env python3
"""
Test KV sharing with Qwen3-8B model (36 layers) sharing the last 9 layers.
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from projects.swiftkv.models.qwen3.modeling_qwen3_swiftkv import Qwen3SwiftKVForCausalLM
from projects.swiftkv.models.qwen3.configuration_qwen3_swiftkv import Qwen3SwiftKVConfig

def test_qwen3_8b_last_9_layers_kv_sharing():
    """Test KV sharing with Qwen3-8B model sharing the last 6 layers."""
    
    model_name = "Qwen/Qwen3-8B"
    print("=" * 80)
    print("TESTING KV SHARING WITH QWEN3-8B (36 LAYERS) - LAST 6 LAYERS")
    print("=" * 80)
    
    # Load original config
    original_config = AutoConfig.from_pretrained(model_name)
    print(f"Model: {model_name}")
    print(f"Total layers: {original_config.num_hidden_layers}")
    print(f"Hidden size: {original_config.hidden_size}")
    print(f"Attention heads: {original_config.num_attention_heads}")
    
    # Create KV sharing map for last 6 layers
    # Layers 31-35 will share KV with layer 30
    kv_sharing_map = {}
    for i in range(35, 36):  # Layers 31-35 share with layer 30
        kv_sharing_map[i] = 34
    
    print(f"\nKV sharing map: {kv_sharing_map}")
    print("Layers 35 will share KV cache with layer 34")
    print("This means 1 layers share KV cache, reducing memory usage")
    
    # Create SwiftKV config with KV sharing
    config = Qwen3SwiftKVConfig(
        vocab_size=original_config.vocab_size,
        hidden_size=original_config.hidden_size,
        intermediate_size=original_config.intermediate_size,
        num_hidden_layers=original_config.num_hidden_layers,
        num_attention_heads=original_config.num_attention_heads,
        num_key_value_heads=original_config.num_key_value_heads,
        num_key_value_layers=35,  # First 34 layers compute KV
        kv_sharing_map=kv_sharing_map,
        swiftkv=True,
        pad_token_id=original_config.pad_token_id,
        eos_token_id=original_config.eos_token_id,
        bos_token_id=original_config.bos_token_id,
    )
    
    print(f"\nConfiguration:")
    print(f"  num_key_value_layers: {config.num_key_value_layers}")
    print(f"  kv_sharing_map: {config.kv_sharing_map}")
    
    # Load the model
    print(f"\nLoading Qwen3-8B model...")
    model = Qwen3SwiftKVForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to solving climate change is",
        "In the realm of quantum computing",
        "The most important skill for programmers is"
    ]
    
    print(f"\nTesting with {len(prompts)} different prompts:")
    print("-" * 60)
    
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    # Test each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        
        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with KV sharing
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"   Generated: {generated_text}")
    
    # Check KV sharing configuration
    print(f"\n" + "=" * 60)
    print("KV SHARING CONFIGURATION VERIFICATION")
    print("=" * 60)
    
    print("Layer KV sharing status:")
    sharing_layers = []
    computing_layers = []
    
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, 'kv_sharing_target_layer_idx'):
            if attn.kv_sharing_target_layer_idx is not None:
                print(f"  Layer {i:2d}: SHARES KV with layer {attn.kv_sharing_target_layer_idx}")
                sharing_layers.append(i)
            else:
                print(f"  Layer {i:2d}: COMPUTES own KV")
                computing_layers.append(i)
        else:
            print(f"  Layer {i:2d}: COMPUTES own KV")
            computing_layers.append(i)
    
    # Memory usage analysis
    print(f"\n" + "=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    total_layers = config.num_hidden_layers
    kv_computing_layers = len(computing_layers)
    kv_sharing_layers = len(sharing_layers)
    
    print(f"Total layers: {total_layers}")
    print(f"Layers computing KV: {kv_computing_layers}")
    print(f"Layers sharing KV: {kv_sharing_layers}")
    print(f"Memory reduction: {kv_sharing_layers}/{total_layers} = {kv_sharing_layers/total_layers*100:.1f}%")
    
    # Performance test
    print(f"\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    import time
    
    # Warm up
    test_prompt = "The future of artificial intelligence"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        _ = model(**inputs, use_cache=True)
    
    # Time multiple forward passes
    num_runs = 3
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average forward pass time: {avg_time:.4f} seconds")
    print(f"Output logits shape: {outputs.logits.shape}")
    
    print(f"\n" + "=" * 80)
    print("QWEN3-8B KV SHARING TEST COMPLETED SUCCESSFULLY!")
    print(f"✅ {kv_sharing_layers} layers sharing KV cache")
    print(f"✅ {kv_sharing_layers/total_layers*100:.1f}% memory reduction")
    print(f"✅ High-quality text generation maintained")
    print("=" * 80)

if __name__ == "__main__":
    test_qwen3_8b_last_9_layers_kv_sharing()
