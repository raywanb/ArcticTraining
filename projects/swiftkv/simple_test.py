#!/usr/bin/env python3
"""
Simple test to verify KV cache sharing works with Qwen3 1.7B model.
"""

import torch
import sys
import os
from transformers import AutoTokenizer, AutoConfig, Qwen3ForCausalLM

# Add the projects directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from projects.swiftkv.models import Qwen3SwiftKVConfig, Qwen3SwiftKVForCausalLM

def main():
    print("🚀 Testing SwiftKV KV Cache Sharing")
    print("=" * 40)
    
    # Load the pretrained Qwen3 model first with Flash Attention
    print("Loading pretrained Qwen3 1.7B model with Flash Attention...")
    base_model = Qwen3ForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_config = base_model.config
    
    print(f"✅ Loaded model with {base_config.num_hidden_layers} layers using Flash Attention 2")
    
    # Create SwiftKV config with KV sharing and Flash Attention
    swiftkv_config = Qwen3SwiftKVConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        max_position_embeddings=base_config.max_position_embeddings,
        rms_norm_eps=base_config.rms_norm_eps,
        attention_bias=base_config.attention_bias,
        pad_token_id=base_config.pad_token_id,
        eos_token_id=base_config.eos_token_id,
        num_key_value_layers=20,  # Layers 20-27 will use SwiftKV
        attn_implementation="flash_attention_2",
    )
    
    # Initialize SwiftKV model directly from pretrained base model
    print("Initializing SwiftKV model directly from pretrained model...")
    model = Qwen3SwiftKVForCausalLM.from_pretrained_base(
        "Qwen/Qwen3-1.7B",
        swiftkv_config=swiftkv_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("✅ SwiftKV model initialized successfully!")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    # Test with a simple prompt
    prompt = "Hello, how are you?"
    print(f"\nTesting with prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(base_model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # # Set both models to eval mode
    # model.eval()
    # base_model.eval()
    
    print("\n" + "="*50)
    print("COMPARISON TEST: Base Model vs SwiftKV Model (Both using Flash Attention 2)")
    print("="*50)
    

    # Generate with SwiftKV model (temperature = 0, deterministic)
    print("\n🟢 SwiftKV Model Generation:")
    with torch.no_grad():
        swiftkv_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
        
        swiftkv_generated_text = tokenizer.decode(swiftkv_outputs[0], skip_special_tokens=True)
        print(f"Generated: '{swiftkv_generated_text}'")
    
        # Generate with base model (temperature = 0, deterministic)
    print("\n🔵 Base Model Generation:")
    with torch.no_grad():
        base_outputs = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
        
        base_generated_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        print(f"Generated: '{base_generated_text}'")
    
    # Compare results
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    if base_generated_text == swiftkv_generated_text:
        print("✅ PERFECT MATCH! Both models generated identical text.")
        print("✅ SwiftKV implementation is working correctly!")
    else:
        print("❌ DIFFERENCE DETECTED!")
        print(f"Base model length: {len(base_generated_text)}")
        print(f"SwiftKV model length: {len(swiftkv_generated_text)}")
        
        # Show character-by-character differences
        print("\nDetailed comparison:")
        for i, (base_char, swiftkv_char) in enumerate(zip(base_generated_text, swiftkv_generated_text)):
            if base_char != swiftkv_char:
                print(f"First difference at position {i}: '{base_char}' vs '{swiftkv_char}'")
                break
        else:
            if len(base_generated_text) != len(swiftkv_generated_text):
                print(f"Length difference: base={len(base_generated_text)}, swiftkv={len(swiftkv_generated_text)}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
