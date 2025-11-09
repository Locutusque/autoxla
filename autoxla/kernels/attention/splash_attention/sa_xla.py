## Copyright IsNoobGrammer 2025. All Rights Reserved.
## Credit goes to IsNoobGrammer for this code. Copied from IsNoobgrammer/Optimized-Attention-Torch-XLA
## Apache License 2.0

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .sa_kernel import SplashAttentionConfig, splash_attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

SPLASH_ATTENTION_AVAILABLE = True
USE_SPLASH_ATTENTION = False

class SplashAttentionWrapper(nn.Module):
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

        # Extract attributes from original attention
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.head_dim = original_attention.head_dim
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx
        self.logits_soft_cap = logits_soft_cap
        self.rotatry_func = rotatry_func
        USE_SPLASH_ATTENTION = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Compute Q, K, V
        query_states = self.original_attention.q_proj(hidden_states)
        key_states = self.original_attention.k_proj(hidden_states)
        value_states = self.original_attention.v_proj(hidden_states)

        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        
        cos, sin = position_embeddings
        query_states, key_states = self.rotatry_func(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Scale query states
        query_states = query_states * self.scaling  ## query_states /= math.sqrt(self.head_dim)

        attn_output = splash_attention(
            query_states,
            key_states,
            value_states,
            self.config.to_json(),
            decoder_segment_ids=None,
            attn_logits_soft_cap=self.logits_soft_cap,
        )

        # Reshape output and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None

print(f"SPLASH_ATTENTION_AVAILABLE : {SPLASH_ATTENTION_AVAILABLE}")