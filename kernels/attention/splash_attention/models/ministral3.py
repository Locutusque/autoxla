from ..sa_xla import _BaseSplashAttentionWrapper
import torch.nn as nn
import torch
from ..sa_kernel import SplashAttentionConfig, splash_attention
from typing import Optional, Unpack
from transformers.models.ministral3.modeling_ministral3 import apply_rotary_pos_emb

def _get_llama_4_attn_scale(positions_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    scaling = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
    return scaling.unsqueeze(-1)

class Ministral3SplashAttention(_BaseSplashAttentionWrapper):
    def __init__(
            self,
            original_attention: nn.Module,
            config: SplashAttentionConfig,
            logits_soft_cap: Optional[float] = None,
    ):
        super().__init__(
            original_attention=original_attention,
            config=config,
            logits_soft_cap=logits_soft_cap,
            rotatry_func=apply_rotary_pos_emb,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.original_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.original_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.original_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states * _get_llama_4_attn_scale(
            cache_position,
            self.original_attention.config.rope_parameters.get("llama_4_scaling_beta"),
            self.original_attention.config.rope_parameters.get("original_max_position_embeddings"),
        ).to(query_states.dtype)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states * self.scaling

        attn_output = splash_attention(
            query_states,
            key_states,
            value_states,
            self.config.to_json(),
            decoder_segment_ids=None,
            attn_logits_soft_cap=self.logits_soft_cap,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
