## Copyright IsNoobGrammer 2025. All Rights Reserved.
## Credit goes to IsNoobGrammer for this code. Copied from IsNoobgrammer/Optimized-Attention-Torch-XLA
## Apache License 2.0

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .sa_kernel import SplashAttentionConfig, splash_attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

SPLASH_ATTENTION_AVAILABLE = True
_use_splash_attention = False

class _BaseSplashAttentionWrapper(nn.Module):
    def __init__(
        self,
        original_attention: nn.Module,
        config: SplashAttentionConfig,
        logits_soft_cap: Optional[float] = None,
        rotatry_func=apply_rotary_pos_emb,
    ):
        """
        A wrapper to replace the original attention mechanism with Splash Attention.

        Args:
            original_attention: The original attention module (e.g., LlamaAttention).
            config: An instance of SplashAttentionConfig containing all necessary parameters.
        """
        super().__init__()
        self.original_attention = original_attention
        self.config = config
        if hasattr(original_attention, "num_key_value_groups"):
            self.num_key_value_groups = original_attention.num_key_value_groups
        # Extract attributes from original attention
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.num_
        self.head_dim = original_attention.head_dim
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx
        self.logits_soft_cap = logits_soft_cap
        self.rotatry_func = rotatry_func
        global _use_splash_attention
        _use_splash_attention = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError(
            "The model you are using is currently not supported.\n"
            "Please open an issue at https://github.com/Locutusque/autoxla"
            " and request support for the model you are trying to use."
            )

print(f"SPLASH_ATTENTION_AVAILABLE : {SPLASH_ATTENTION_AVAILABLE}")
