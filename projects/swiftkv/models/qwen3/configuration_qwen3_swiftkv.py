

from typing import Dict, Optional

from transformers import Qwen3Config


class Qwen3SwiftKVConfig(Qwen3Config):
    """
    Args:
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
        key_value_group_size (int, optional):
            The number of layers in each group after num_key_value_layers
            that will share a single KV cache.
        kv_sharing_map (Dict[int, int], optional):
            Mapping from layer index to the layer index whose KV cache should be used.
            If provided, this overrides key_value_group_size behavior.
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
        self.num_key_value_layers = num_key_value_layers or self.num_hidden_layers
        self.key_value_group_size = key_value_group_size or 1
        self.kv_sharing_map = kv_sharing_map or {}
        
        # If kv_sharing_map is provided, it overrides the group size behavior
        if not self.kv_sharing_map:
            assert (self.num_hidden_layers - self.num_key_value_layers) % self.key_value_group_size == 0

        