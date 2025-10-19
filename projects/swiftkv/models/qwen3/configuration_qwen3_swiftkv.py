

from typing import Dict, Optional

from transformers import Qwen3Config


class Qwen3SwiftKVConfig(Qwen3Config):
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
        key_value_group_size (int, optional):
            DEPRECATED. No longer used. Kept for backward compatibility only.
    """

    model_type = "qwen3_swiftkv"

    def __init__(
        self,
        swiftkv: bool = False,
        num_key_value_layers: Optional[int] = None,
        key_value_group_size: Optional[int] = None,
        kv_sharing_map: Optional[Dict[int, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiftkv = swiftkv
        self.kv_sharing_map = kv_sharing_map or {}
        
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

        