#!/usr/bin/env python3
import torch
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import Qwen3ForCausalLM
from projects.swiftkv.models import Qwen3SwiftKVConfig, Qwen3SwiftKVForCausalLM

print('Loading base model...')
base_model = Qwen3ForCausalLM.from_pretrained(
    'Qwen/Qwen3-8B',
    attn_implementation='flash_attention_2',
    torch_dtype=torch.float16,
    device_map='auto'
)

print('Loading SwiftKV model...')
swiftkv_config = Qwen3SwiftKVConfig.from_pretrained('Qwen/Qwen3-8B')
swiftkv_config.kv_sharing_map = {}
swiftkv_config.attn_implementation = 'flash_attention_2'

model = Qwen3SwiftKVForCausalLM.from_pretrained(
    'Qwen/Qwen3-8B',
    config=swiftkv_config,
    attn_implementation='flash_attention_2',
    torch_dtype=torch.float16,
    device_map='auto'
)

# Compare weights
base_weight = base_model.model.layers[0].self_attn.q_proj.weight
swiftkv_weight = model.model.layers[0].self_attn.q_proj.weight
print(f'Base q_proj weight shape: {base_weight.shape}')
print(f'SwiftKV q_proj weight shape: {swiftkv_weight.shape}')
print(f'Weights match: {torch.allclose(base_weight.cpu(), swiftkv_weight.cpu(), atol=1e-5)}')
print(f'Max difference: {(base_weight.cpu() - swiftkv_weight.cpu()).abs().max().item()}')











