# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Union

import torch
import transformers
from deepspeed.runtime.zero import GatheredParameters
from packaging import version
from torch.distributed import ReduceOp

from arctic_training import DSCheckpointEngine
from arctic_training import HFCheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import TrainerConfig
from arctic_training.trainer.sft_trainer import to_device
from projects.swiftkv.models import DeepseekV2SwiftKVConfig
from projects.swiftkv.models import LlamaSwiftKVConfig
from projects.swiftkv.models import Qwen2SwiftKVConfig
from projects.swiftkv.models import Qwen3SwiftKVConfig
from projects.swiftkv.models import register_all_swiftkv
from projects.swiftkv.models.deepseek_v2 import register_deepseek_v2
from projects.swiftkv.train import SwiftKVModelConfig, SwiftKVTrainerConfig

register_all_swiftkv()

if version.parse(transformers.__version__) < version.parse("4.54.0"):
    register_deepseek_v2()  # Explicitly register because it's not in transformers


class SwiftKVSFTModelFactory(HFModelFactory):
    name = "swiftkv_sft"
    config: SwiftKVModelConfig

    def post_create_config_callback(self, hf_config):
        config_dict = hf_config.to_dict()

        model_type = config_dict.get("model_type")
        if model_type in ["deepseek_v2", "deepseek_v2_swiftkv"]:
            hf_config = DeepseekV2SwiftKVConfig.from_dict(config_dict)
        elif model_type in ["llama", "llama_swiftkv"]:
            hf_config = LlamaSwiftKVConfig.from_dict(config_dict)
        elif model_type in ["qwen2", "qwen2_swiftkv"]:
            hf_config = Qwen2SwiftKVConfig.from_dict(config_dict)
        elif model_type in ["qwen3", "qwen3_swiftkv"]:
            hf_config = Qwen3SwiftKVConfig.from_dict(config_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        hf_config.num_key_value_layers = self.config.num_key_value_layers
        
        if hasattr(self.config, 'kv_sharing_map'):
            hf_config.kv_sharing_map = self.config.kv_sharing_map
        
        if hasattr(self.config, 'mlp_tuning_enabled'):
            hf_config.mlp_tuning_enabled = self.config.mlp_tuning_enabled
        
        if hasattr(self.config, 'layernorm_tuning_enabled'):
            hf_config.layernorm_tuning_enabled = self.config.layernorm_tuning_enabled

        return hf_config

    def post_create_model_callback(self, model):
        # Check if this is a SwiftKV model
        is_swiftkv_model = "swiftkv" in model.config.model_type.lower()
        
        if model.config.model_type in ["deepseek_v2", "deepseek_v2_swiftkv"]:
            if model.config.q_lora_rank is None:
                q_modules = ["q_proj"]
            else:
                q_modules = ["q_a_proj", "q_b_proj", "q_a_layernorm"]
            kv_modules = ["kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm"]
            mlp_modules = []
            layernorm_modules = []
        elif model.config.model_type in ["qwen3", "qwen3_swiftkv"]:
            q_modules = ["q_proj", "q_norm"]
            # kv_modules = ["k_proj", "k_norm", "v_proj"]
            kv_modules = []
            mlp_modules = ["gate_proj", "up_proj", "down_proj"] if getattr(model.config, 'mlp_tuning_enabled', True) else []
            layernorm_modules = ["input_layernorm", "post_attention_layernorm"] if getattr(model.config, 'layernorm_tuning_enabled', False) else []
        elif model.config.model_type in ["llama", "llama_swiftkv"]:
            q_modules = ["q_proj"]
            kv_modules = ["k_proj", "v_proj"]
            mlp_modules = ["gate_proj", "up_proj", "down_proj"] if getattr(model.config, 'mlp_tuning_enabled', False) else []
            layernorm_modules = ["input_layernorm", "post_attention_layernorm"] if getattr(model.config, 'layernorm_tuning_enabled', False) else []
        else:
            q_modules = ["q_proj"]
            kv_modules = ["k_proj", "v_proj"]
            mlp_modules = []
            layernorm_modules = []

        # Freeze all teacher parameters
        for param in model.parameters():
            param.requires_grad = False
    
        # If this is a SwiftKV model, load SwiftKV parameters and set them to trainable
        if is_swiftkv_model:
            print(f"Detected SwiftKV model type: {model.config.model_type}")
            print("Setting SwiftKV parameters to trainable...")
            
            swiftkv_params_found = 0
            swiftkv_params_set_trainable = 0
            
            # Set all parameters with "swiftkv" in the name to trainable
            for name, param in model.named_parameters():
                if "swiftkv" in name.lower():
                    swiftkv_params_found += 1
                    param.requires_grad = True
                    swiftkv_params_set_trainable += 1
            
            print(f"Found {swiftkv_params_found} SwiftKV parameters")
            print(f"Set {swiftkv_params_set_trainable} SwiftKV parameters to trainable")
            
        else:
            # Original initialization logic for non-checkpoint cases
            if hasattr(model.model, "norm_swiftkv"):
                # Initialize the swiftkv norm from the original model's norm.
                with GatheredParameters(
                    list(model.model.norm_swiftkv.parameters()) + list(model.model.norm.parameters()), modifier_rank=0
                ):
                    model.model.norm_swiftkv.weight.data.copy_(model.model.norm.weight.data)
                model.model.norm_swiftkv.weight.requires_grad = False

            
            # Initialize all query parameters directly from the corresponding teacher layer
            # for layers that are consumers in the kv_sharing_map
            consumer_layers = set(model.config.kv_sharing_map.keys())
            for layer_idx in consumer_layers:
                layer = model.model.layers[layer_idx]
                attn = layer.self_attn
                mlp = layer.mlp
                with GatheredParameters(attn.parameters(), modifier_rank=0):
                    for q_module in q_modules:
                        teacher_params = getattr(attn, q_module).parameters()
                        student_params = getattr(attn, f"{q_module}_swiftkv").parameters()
                        for teacher_param, student_param in zip(teacher_params, student_params):
                            student_param.data.copy_(teacher_param.data)
                            student_param.requires_grad = True
                
                if mlp_modules:
                    with GatheredParameters(mlp.parameters(), modifier_rank=0):
                        for mlp_module in mlp_modules:
                            teacher_params = getattr(mlp, mlp_module).parameters()
                            student_params = getattr(mlp, f"{mlp_module}_swiftkv").parameters()
                            for teacher_param, student_param in zip(teacher_params, student_params):
                                student_param.data.copy_(teacher_param.data)
                                student_param.requires_grad = True
                
                if layernorm_modules:
                    with GatheredParameters(layer.parameters(), modifier_rank=0):
                        for layernorm_module in layernorm_modules:
                            teacher_params = getattr(layer, layernorm_module).parameters()
                            student_params = getattr(layer, f"{layernorm_module}_swiftkv").parameters()
                            for teacher_param, student_param in zip(teacher_params, student_params):
                                student_param.data.copy_(teacher_param.data)
                                student_param.requires_grad = True
        
        # Check if we have any trainable parameters, if not set all parameters trainable
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        if trainable_count == 0:
            print("WARNING: No trainable parameters found. Setting all parameters to trainable.")
            for param in model.parameters():
                param.requires_grad = True
            trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"Set {trainable_count} parameters to trainable")
        else:
            print(f"Found {trainable_count} trainable parameters")

        # Initialize all kv parameters to the mean of the teacher layers in each kv group.
        # for idx, layer in enumerate(model.model.layers[model.config.num_key_value_layers :]):
        #     attn = layer.self_attn
        #     if idx % model.config.key_value_group_size == 0:
        #         # This layer has swiftkv parameters, zero them out.
        #         kv_attn = attn
        #         with GatheredParameters(kv_attn.parameters(), modifier_rank=0):
        #             # Zero out the swiftkv parameters
        #             for kv_module in kv_modules:
        #                 for param in getattr(kv_attn, f"{kv_module}_swiftkv").parameters():
        #                     param.data.zero_()
        #                     param.requires_grad = True
        #     with GatheredParameters(attn.parameters(), modifier_rank=0):
        #         # Accumulate the teacher parameters into the swiftkv parameters.
        #         for kv_module in kv_modules:
        #             teacher_params = getattr(attn, kv_module).parameters()
        #             student_params = getattr(kv_attn, f"{kv_module}_swiftkv").parameters()
        #             for teacher_param, student_param in zip(teacher_params, student_params):
        #                 student_param.data.add_(teacher_param.data / model.config.key_value_group_size)

        # print("\n=== TRAINABLE PARAMETERS ===")
        # trainable_params = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         trainable_params.append((name, param.shape, param.numel()))
        
        # for name, shape, numel in trainable_params[:30]:  # Print first 30
        #     print(f"  {name}: {shape} ({numel} params)")
        
        # print(f"\nTotal trainable parameters: {len(trainable_params)}")
        # print(f"Total trainable elements: {sum(n for _, _, n in trainable_params)}")
        
        # if len(trainable_params) == 0:
        #     raise RuntimeError("ERROR: No trainable parameters found!")
        
        # model.gradient_checkpointing_enable()
        return model


class SwiftKVSFTTrainer(SFTTrainer):
    name = "swiftkv_sft"
    config: SwiftKVTrainerConfig
    model_factory: SwiftKVSFTModelFactory
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]

    def forward(self, batch):
        batch = to_device(batch, self.device)
        
        # Run model in SwiftKV mode
        self.model.swiftkv(True)
        self.model.train()
        outputs = self.model(**batch)
        
        return outputs
    
    def loss(self, batch) -> torch.Tensor:
        outputs = self.forward(batch)
        
        # Standard cross-entropy loss
        loss = outputs.loss
        
        # Apply sequence parallel reduction if needed
        use_sequence_parallel = self.config.sequence_parallel_size > 1
        if use_sequence_parallel:
            loss = torch.distributed.nn.functional.all_reduce(loss, op=ReduceOp.AVG, group=self.sp_group)
        
        return loss
