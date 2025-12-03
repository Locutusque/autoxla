from .qwen3 import Qwen3SplashAttention
from .gemma2 import Gemma2SplashAttention
from .llama import LlamaSplashAttention
from .mistral import MistralSplashAttention
from .ministral3 import Ministral3SplashAttention
from ..sa_xla import _BaseSplashAttentionWrapper
from transformers import PreTrainedModel

ATTN_CLASSES = {
    "llama": LlamaSplashAttention,
    "gemma2": Gemma2SplashAttention,
    "qwen3": Qwen3SplashAttention,
    "mistral": MistralSplashAttention,
    "ministral3": Ministral3SplashAttention
}

def _get_attention_class_from_model(model: PreTrainedModel):
    if hasattr(model.config, "vision_config"):
        model_type = model.config.text_config.model_type
        if model_type == "mistral":
            model_type = "ministral3"
    else:
        model_type = model.config.model_type
    attn_class = ATTN_CLASSES.get(model_type, _BaseSplashAttentionWrapper)
    return attn_class
