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

register_all_swiftkv()

if version.parse(transformers.__version__) < version.parse("4.54.0"):
    register_deepseek_v2()  # Explicitly register because it's not in transformers


class SwiftKVModelConfig(ModelConfig):
    num_key_value_layers: int
    """
    Initial number of layers that compute KV cache. The output from layer
    `num_key_value_layers` is used to compute the KV for all subsequent layers.
    """

    key_value_group_size: int = 1
    """
    Number of consecutive layers that share the same KV cache, only applies to
    layers after `num_key_value_layers`.
    """
    
    kv_sharing_map: dict = {}
    """
    Mapping from consumer layer index to producer layer index for KV sharing.
    If layer i maps to layer j, then layer i will use the KV cache from layer j.
    If empty, no SwiftKV parameters will be created and all parameters are trainable.
    """




class SwiftKVTrainerConfig(TrainerConfig):
    logits_loss_temp: float = 2.0
    """Temperature for the distillation (KL-div) loss on logits."""

    hidden_loss_mult: float = 1.0
    """
    Weight for the distillation (MSE) loss on hidden states. The final loss
    is computed as `logits_loss + hidden_loss_mult * hidden_loss`.
    """

    hidden_loss_layer: int = -2
    """The index of the layer whose output is used for the hidden loss."""


class SwiftKVModelFactory(HFModelFactory):
    name = "swiftkv"
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
        hf_config.key_value_group_size = self.config.key_value_group_size
        
        if hasattr(self.config, 'kv_sharing_map'):
            hf_config.kv_sharing_map = self.config.kv_sharing_map

        return hf_config

    def post_create_model_callback(self, model):
        if model.config.model_type in ["deepseek_v2", "deepseek_v2_swiftkv"]:
            if model.config.q_lora_rank is None:
                q_modules = ["q_proj"]
            else:
                q_modules = ["q_a_proj", "q_b_proj", "q_a_layernorm"]
            kv_modules = ["kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm"]
        elif model.config.model_type in ["qwen3", "qwen3_swiftkv"]:
            q_modules = ["q_proj", "q_norm"]
            # kv_modules = ["k_proj", "k_norm", "v_proj"]
            kv_modules = []
        else:
            q_modules = ["q_proj"]
            kv_modules = ["k_proj", "v_proj"]

        if self.config.kv_sharing_map:
            # SwiftKV mode: Freeze all parameters, then unfreeze only SwiftKV params
            for param in model.parameters():
                param.requires_grad = False
        
            if hasattr(model.model, "norm_swiftkv"):
                # Initialize the swiftkv norm from the original model's norm.
                with GatheredParameters(
                    list(model.model.norm_swiftkv.parameters()) + list(model.model.norm.parameters()), modifier_rank=0
                ):
                    model.model.norm_swiftkv.weight.data.copy_(model.model.norm.weight.data)
                model.model.norm_swiftkv.weight.requires_grad = False

            # Initialize all query parameters directly from the corresponding teacher layer.
            consumer_layers = set(model.config.kv_sharing_map.keys())
            for layer_idx in consumer_layers:
                layer = model.model.layers[layer_idx]
                attn = layer.self_attn
                with GatheredParameters(attn.parameters(), modifier_rank=0):
                    for q_module in q_modules:
                        teacher_params = getattr(attn, q_module).parameters()
                        student_params = getattr(attn, f"{q_module}_swiftkv").parameters()
                        for teacher_param, student_param in zip(teacher_params, student_params):
                            student_param.data.copy_(teacher_param.data)
                            student_param.requires_grad = True

            # Initialize all kv parameters to the mean of the teacher layers in each kv group.
            for idx, layer in enumerate(model.model.layers[model.config.num_key_value_layers :]):
                attn = layer.self_attn
                if idx % model.config.key_value_group_size == 0:
                    # This layer has swiftkv parameters, zero them out.
                    kv_attn = attn
                    with GatheredParameters(kv_attn.parameters(), modifier_rank=0):
                        # Zero out the swiftkv parameters
                        for kv_module in kv_modules:
                            for param in getattr(kv_attn, f"{kv_module}_swiftkv").parameters():
                                param.data.zero_()
                                param.requires_grad = True
                with GatheredParameters(attn.parameters(), modifier_rank=0):
                    # Accumulate the teacher parameters into the swiftkv parameters.
                    for kv_module in kv_modules:
                        teacher_params = getattr(attn, kv_module).parameters()
                        student_params = getattr(kv_attn, f"{kv_module}_swiftkv").parameters()
                        for teacher_param, student_param in zip(teacher_params, student_params):
                            student_param.data.add_(teacher_param.data / model.config.key_value_group_size)
        else:
            # Full fine-tuning mode: All parameters are trainable (default behavior)
            # No SwiftKV parameters are created, so just ensure all params are trainable
            for param in model.parameters():
                param.requires_grad = True

        # model.gradient_checkpointing_enable()
        return model


class SwiftKVTrainer(SFTTrainer):
    name = "swiftkv"
    config: SwiftKVTrainerConfig
    model_factory: SwiftKVModelFactory
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]

    def loss(self, batch) -> torch.Tensor:
        # Set SwiftKV mode based on whether we have a kv_sharing_map
        # If kv_sharing_map exists: use SwiftKV parameters (swiftkv=True)
        # If kv_sharing_map is empty: use all original parameters (swiftkv=False)
        use_swiftkv = len(self.config.model.kv_sharing_map) > 0
        self.model.swiftkv(use_swiftkv)
        
        # Call parent's loss method which handles all the complexity:
        # - Proper use_cache=False
        # - Sequence parallel with proper token weighting
        # - Liger kernel support
        # - Tiled logits computation
        return super().loss(batch)
