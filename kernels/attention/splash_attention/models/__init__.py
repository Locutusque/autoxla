from .qwen3 import Qwen3SplashAttention
from .gemma2 import Gemma2SplashAttention
from .llama import LlamaSplashAttention
from ..sa_xla import _BaseSplashAttentionWrapper
from transformers import PreTrainedModel

ATTN_CLASSES = {
    "llama": LlamaSplashAttention,
    "gemma2": Gemma2SplashAttention,
    "qwen3": Qwen3SplashAttention
}

def _get_attention_class_from_model(model: PreTrainedModel):
    attn_class = ATTN_CLASSES.get(model.config.model_type, _BaseSplashAttentionWrapper)
    return attn_class