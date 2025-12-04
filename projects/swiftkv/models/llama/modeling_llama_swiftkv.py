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

from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .configuration_llama_swiftkv import LlamaSwiftKVConfig

logger = logging.get_logger(__name__)


class LlamaSwiftKVAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaSwiftKVConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)

        self.layer_idx = layer_idx
        self.kv_sharing_target_layer_idx = config.kv_sharing_map.get(layer_idx, None)
        print("layer_idx", layer_idx)
        print("kv_sharing_target_layer_idx", self.kv_sharing_target_layer_idx)
        # Create SwiftKV projections for layers that consume KV from other layers
        if layer_idx in config.kv_sharing_map:
            self.q_proj_swiftkv = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=config.attention_bias,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Compute queries (always layer-specific)
        if self.config.swiftkv and self.layer_idx in self.config.kv_sharing_map:
            query_states = self.q_proj_swiftkv(hidden_states).view(hidden_shape)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)

        # print("SHARE KV", self.config.swiftkv, self.kv_sharing_target_layer_idx, past_key_value)
        shares_kv = (self.config.swiftkv and 
                     self.kv_sharing_target_layer_idx is not None and 
                     past_key_value is not None)

        if shares_kv:
            target_idx = self.kv_sharing_target_layer_idx
            
            key_states = past_key_value.layers[target_idx].keys
            value_states = past_key_value.layers[target_idx].values
    
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

        # Reshape for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2) if not shares_kv else key_states
        value_states = value_states.transpose(1, 2) if not shares_kv else value_states

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update cache only for layers that compute their own KV
        if past_key_value is not None and not shares_kv:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Attention computation
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaSwiftKVMLP(LlamaMLP):
    """MLP layer with SwiftKV support for KV distillation"""
    
    def __init__(self, config: LlamaSwiftKVConfig, layer_idx: int):
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx
        
        # Create SwiftKV-specific projections for layers that share KV
        if config.mlp_tuning_enabled and layer_idx in config.kv_sharing_map:
            self.gate_proj_swiftkv = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj_swiftkv = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj_swiftkv = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use SwiftKV projections for layers that share KV cache when MLP tuning is enabled
        if self.config.mlp_tuning_enabled and self.config.swiftkv and self.layer_idx in self.config.kv_sharing_map:
            down_proj = self.down_proj_swiftkv(
                self.act_fn(self.gate_proj_swiftkv(x)) * self.up_proj_swiftkv(x)
            )
        else:
            # Use original MLP projections
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaSwiftKVDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config: LlamaSwiftKVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaSwiftKVAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaSwiftKVMLP(config, layer_idx=layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Create SwiftKV-specific layer norms for consumer layers
        if config.layernorm_tuning_enabled and layer_idx in config.kv_sharing_map:
            self.input_layernorm_swiftkv = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm_swiftkv = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Use SwiftKV layer norm for consumer layers when enabled
        if self.config.layernorm_tuning_enabled and self.config.swiftkv and self.layer_idx in self.config.kv_sharing_map:
            hidden_states = self.input_layernorm_swiftkv(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # Use SwiftKV layer norm for consumer layers when enabled
        if self.config.layernorm_tuning_enabled and self.config.swiftkv and self.layer_idx in self.config.kv_sharing_map:
            hidden_states = self.post_attention_layernorm_swiftkv(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaSwiftKVModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`LlamaSwiftKVDecoderLayer`].

    Args:
        config: LlamaSwiftKVConfig
    """

    config_class = LlamaSwiftKVConfig

    def __init__(self, config: LlamaSwiftKVConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaSwiftKVDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.training and getattr(decoder_layer, "gradient_checkpointing", False):
                hidden_states.requires_grad_(True)  # make the checkpointed input valid
           
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaSwiftKVForCausalLM(LlamaForCausalLM):

    config_class = LlamaSwiftKVConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaSwiftKVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def swiftkv(self, swiftkv: bool = True):
        self.config.swiftkv = swiftkv
        return self
