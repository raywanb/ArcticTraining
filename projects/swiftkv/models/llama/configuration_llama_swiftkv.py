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

from typing import Dict, Optional

from transformers import LlamaConfig


class LlamaSwiftKVConfig(LlamaConfig):
    """
    Args:
        swiftkv (bool, optional):
            Whether to enable SwiftKV mode. Default: False.
        num_key_value_layers (int, optional):
            Informational parameter indicating the number of layers that produce KV cache.
            If not set and kv_sharing_map is provided, it will be computed as the number
            of unique producer layers. If None and no kv_sharing_map, defaults to num_hidden_layers.
        kv_sharing_map (Dict[int, int], optional):
            Primary mechanism for defining KV cache sharing. Maps consumer layer index to
            producer layer index. If layer i maps to layer j, then layer i will use the
            KV cache from layer j. This allows flexible sharing patterns (e.g., layer 3
            shares from layer 2, but layer 4 produces its own KV cache).
        mlp_tuning_enabled (bool, optional):
            Whether to enable separate trainable MLP projections for layers that share KV cache.
            When enabled, consumer layers will have separate gate_proj_swiftkv, up_proj_swiftkv,
            and down_proj_swiftkv parameters. Default: False.
        layernorm_tuning_enabled (bool, optional):
            Whether to enable separate trainable layer norms for layers that share KV cache.
            When enabled, consumer layers will have separate input_layernorm_swiftkv and
            post_attention_layernorm_swiftkv parameters. Default: False.
        key_value_group_size (int, optional):
            DEPRECATED. No longer used. Kept for backward compatibility only.
    """

    model_type = "llama_swiftkv"

    def __init__(
        self,
        swiftkv: bool = False,
        num_key_value_layers: Optional[int] = None,
        key_value_group_size: Optional[int] = None,
        kv_sharing_map: Optional[Dict[int, int]] = None,
        mlp_tuning_enabled: bool = False,
        layernorm_tuning_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiftkv = swiftkv
        self.kv_sharing_map = kv_sharing_map or {}
        self.mlp_tuning_enabled = mlp_tuning_enabled
        self.layernorm_tuning_enabled = layernorm_tuning_enabled
        
        if num_key_value_layers is None:
            if self.kv_sharing_map:
                consumer_layers = set(self.kv_sharing_map.keys())
                all_layers = set(range(self.num_hidden_layers))
                producer_layers = all_layers - consumer_layers
                self.num_key_value_layers = len(producer_layers)
            else:
                self.num_key_value_layers = self.num_hidden_layers
        else:
            self.num_key_value_layers = num_key_value_layers
        
        self.key_value_group_size = key_value_group_size or 1
