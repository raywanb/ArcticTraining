from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM

from .configuration_qwen3_swiftkv import Qwen3SwiftKVConfig
from .modeling_qwen3_swiftkv import Qwen3SwiftKVForCausalLM
from .modeling_qwen3_swiftkv import Qwen3SwiftKVModel

def register_qwen3_swiftkv():
    AutoConfig.register("qwen3_swiftkv", Qwen3SwiftKVConfig)
    AutoModel.register(Qwen3SwiftKVConfig, Qwen3SwiftKVModel)
    AutoModelForCausalLM.register(Qwen3SwiftKVConfig, Qwen3SwiftKVForCausalLM)

__all__ = [
    "Qwen3SwiftKVConfig",
    "Qwen3SwiftKVForCausalLM",
    "Qwen3SwiftKVModel",
    "register_qwen3_swiftkv",
]