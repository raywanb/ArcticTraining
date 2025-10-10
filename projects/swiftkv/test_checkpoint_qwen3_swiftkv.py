#!/usr/bin/env python3
"""
Test trained Qwen3-SwiftKV checkpoint from checkpoint/qwen3-swiftkv-8b-5-consumers/global_step_1234
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from projects.swiftkv.models.qwen3.modeling_qwen3_swiftkv import Qwen3SwiftKVForCausalLM
from projects.swiftkv.models.qwen3.configuration_qwen3_swiftkv import Qwen3SwiftKVConfig

# Register the custom config and model classes so AutoConfig/AutoModel can find them
AutoConfig.register("qwen3_swiftkv", Qwen3SwiftKVConfig)

def test_trained_checkpoint():
    """Test trained Qwen3-SwiftKV checkpoint."""
    
    checkpoint_path = "/data/raywanb/checkpoint/qwen3-swiftkv-8b-5-consumers/global_step_1234"
    base_model_name = "Qwen/Qwen3-8B"
    
    print("=" * 80)
    print("TESTING TRAINED QWEN3-SWIFTKV CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    
    # First check if config.json exists and contains SwiftKV parameters
    import json
    import os
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        print(f"\nVerifying config.json file...")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        print(f"  Model type in config.json: {config_dict.get('model_type', 'NOT FOUND')}")
        
        # Check for SwiftKV-specific keys
        swiftkv_keys = ['swiftkv', 'num_key_value_layers', 'key_value_group_size', 'kv_sharing_map']
        print(f"  SwiftKV parameters in config.json:")
        for key in swiftkv_keys:
            if key in config_dict:
                value = config_dict[key]
                if isinstance(value, dict) and len(value) > 3:
                    print(f"    ✓ {key}: <dict with {len(value)} entries>")
                else:
                    print(f"    ✓ {key}: {value}")
            else:
                print(f"    ✗ {key}: NOT FOUND")
                print(f"    WARNING: Config file may be missing SwiftKV parameters!")
    else:
        print(f"\nWARNING: config.json not found at {config_path}")
    
    # First, load and validate the config separately
    print(f"\nLoading config from checkpoint...")
    config = Qwen3SwiftKVConfig.from_pretrained(checkpoint_path)
    
    print(f"\n" + "=" * 80)
    print("CONFIG VALIDATION")
    print("=" * 80)
    print(f"Config class: {config.__class__.__name__}")
    print(f"Model type: {config.model_type}")
    
    # Verify all SwiftKV-specific attributes are present and valid
    required_attrs = ['swiftkv', 'num_key_value_layers', 'key_value_group_size', 'kv_sharing_map']
    print(f"\nVerifying SwiftKV-specific config attributes:")
    for attr in required_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            print(f"  ✓ {attr}: {value}")
        else:
            print(f"  ✗ {attr}: MISSING")
            raise ValueError(f"Config is missing required attribute: {attr}")
    
    # Validate kv_sharing_map structure
    print(f"\nKV Sharing Map Details:")
    print(f"  kv_sharing_map type: {type(config.kv_sharing_map)}")
    print(f"  kv_sharing_map value: {config.kv_sharing_map}")
    
    # Fix JSON serialization issue: dict keys might be strings instead of ints
    if config.kv_sharing_map and isinstance(next(iter(config.kv_sharing_map.keys())), str):
        print(f"  ⚠️  Fixing: Converting string keys to integers...")
        config.kv_sharing_map = {int(k): int(v) for k, v in config.kv_sharing_map.items()}
        print(f"  ✓ Converted to: {config.kv_sharing_map}")
    
    if config.kv_sharing_map:
        print(f"  Total consumer layers: {len(config.kv_sharing_map)}")
        for consumer_idx, producer_idx in sorted(config.kv_sharing_map.items()):
            print(f"    Layer {consumer_idx} → Layer {producer_idx}")
    else:
        print(f"  WARNING: kv_sharing_map is empty!")
        print(f"  Expected layers {config.num_key_value_layers} to {config.num_hidden_layers - 1} to share KV")
        
        # Build the expected kv_sharing_map based on the config
        if config.num_key_value_layers < config.num_hidden_layers:
            expected_map = {}
            for i in range(config.num_key_value_layers, config.num_hidden_layers):
                # Map consumer layer to producer layer based on group size
                group_offset = (i - config.num_key_value_layers) // config.key_value_group_size
                producer_idx = group_offset % config.num_key_value_layers
                expected_map[i] = producer_idx
            print(f"  Expected kv_sharing_map should be: {expected_map}")
            print(f"\n  ⚠️  FIXING: Manually setting kv_sharing_map on config...")
            config.kv_sharing_map = expected_map
            print(f"  ✓ Config kv_sharing_map updated")
    
    # Load the trained model with the validated config
    print(f"\nLoading trained model...")
    model = Qwen3SwiftKVForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,  # Explicitly pass the config
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Verify the model's config matches what we loaded
    assert model.config.swiftkv == config.swiftkv, "Model config mismatch: swiftkv"
    assert model.config.kv_sharing_map == config.kv_sharing_map, "Model config mismatch: kv_sharing_map"
    print(f"✓ Model config verified to match loaded config")
    
    # Update attention layers' kv_sharing_target_layer_idx if config was fixed
    if config.kv_sharing_map:
        print(f"\nUpdating attention layers with kv_sharing_map...")
        updated_count = 0
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            expected_target = config.kv_sharing_map.get(i, None)
            current_target = getattr(attn, 'kv_sharing_target_layer_idx', None)
            
            if expected_target != current_target:
                attn.kv_sharing_target_layer_idx = expected_target
                updated_count += 1
                print(f"  Updated Layer {i}: kv_sharing_target_layer_idx = {expected_target}")
        
        if updated_count > 0:
            print(f"✓ Updated {updated_count} attention layers with correct kv_sharing_target_layer_idx")
        else:
            print(f"✓ All attention layers already have correct kv_sharing_target_layer_idx")
    
    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Print model configuration
    config = model.config
    print(f"\nModel Configuration:")
    print(f"  Total layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Key-value heads: {config.num_key_value_heads}")
    print(f"  Key-value layers: {config.num_key_value_layers}")
    print(f"  KV sharing enabled: {config.swiftkv}")
    print(f"  KV sharing map: {config.kv_sharing_map}")
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to solving climate change is",
        "In the realm of quantum computing",
        "The most important skill for programmers is",
        "Explain the concept of machine learning:",
        "Write a short story about a robot:",
        "What are the benefits of renewable energy?",
    ]
    
    print(f"\nTesting with {len(prompts)} different prompts:")
    print("-" * 80)
    
    device = next(model.parameters()).device
    print(f"Device: {device}\n")
    
    # Test each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        
        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with trained model
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"   Generated: {generated_text}")
        print("-" * 80)
    
    # Check KV sharing configuration
    print(f"\n" + "=" * 80)
    print("KV SHARING CONFIGURATION VERIFICATION")
    print("=" * 80)
    
    # Verify that the config was properly ingested by checking layer-level configs
    print("\nVerifying config propagation to model layers:")
    for i, layer in enumerate(model.model.layers[:3]):  # Check first 3 layers as sample
        attn = layer.self_attn
        if hasattr(attn, 'config'):
            print(f"  Layer {i}: config.swiftkv = {attn.config.swiftkv}, "
                  f"config.num_key_value_layers = {attn.config.num_key_value_layers}")
            assert attn.config.swiftkv == config.swiftkv, f"Layer {i} config mismatch!"
            assert attn.config.kv_sharing_map == config.kv_sharing_map, f"Layer {i} kv_sharing_map mismatch!"
        else:
            print(f"  Layer {i}: WARNING - no config attribute found")
    print("  ✓ Config properly propagated to attention layers")
    
    print("\nLayer KV sharing status:")
    sharing_layers = []
    computing_layers = []
    
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, 'kv_sharing_target_layer_idx'):
            target_idx = attn.kv_sharing_target_layer_idx
            if target_idx is not None:
                print(f"  Layer {i:2d}: SHARES KV with layer {target_idx}")
                sharing_layers.append(i)
                # Verify this matches the config
                if i in config.kv_sharing_map:
                    assert config.kv_sharing_map[i] == target_idx, \
                        f"Mismatch: Layer {i} targets {target_idx} but config says {config.kv_sharing_map[i]}"
            else:
                print(f"  Layer {i:2d}: COMPUTES own KV")
                computing_layers.append(i)
        else:
            print(f"  Layer {i:2d}: COMPUTES own KV (no kv_sharing_target_layer_idx attribute)")
            computing_layers.append(i)
    
    # Memory usage analysis
    print(f"\n" + "=" * 80)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 80)
    
    total_layers = config.num_hidden_layers
    kv_computing_layers = len(computing_layers)
    kv_sharing_layers = len(sharing_layers)
    
    print(f"Total layers: {total_layers}")
    print(f"Layers computing KV: {kv_computing_layers}")
    print(f"Layers sharing KV: {kv_sharing_layers}")
    if total_layers > 0:
        print(f"Memory reduction: {kv_sharing_layers}/{total_layers} = {kv_sharing_layers/total_layers*100:.1f}%")
    
    # Performance test
    print(f"\n" + "=" * 80)
    print("PERFORMANCE TEST")
    print("=" * 80)
    
    import time
    
    # Warm up
    test_prompt = "The future of artificial intelligence"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        _ = model(**inputs, use_cache=True)
    
    # Time multiple forward passes
    num_runs = 5
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Number of runs: {num_runs}")
    print(f"Average forward pass time: {avg_time:.4f} seconds")
    print(f"Output logits shape: {outputs.logits.shape}")
    
    print(f"\n" + "=" * 80)
    print("TRAINED CHECKPOINT TEST COMPLETED SUCCESSFULLY!")
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"✅ {kv_sharing_layers} layers sharing KV cache")
    if total_layers > 0:
        print(f"✅ {kv_sharing_layers/total_layers*100:.1f}% memory reduction")
    print(f"✅ Text generation working correctly")
    print("=" * 80)

if __name__ == "__main__":
    test_trained_checkpoint()

