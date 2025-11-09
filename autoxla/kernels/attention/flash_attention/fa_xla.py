# Copyright 2025 IsNoobGrammer. All Rights Reserved.
# Credit goes to IsNoobGrammer for this code. Copied from IsNoobgrammer/Optimized-Attention-Torch-XLA
# Apache License 2.0


import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd.xla_sharding as xs
from typing import Tuple,Optional
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

# Assuming XLA Flash Attention is available as defined
from torch_xla.experimental.custom_kernel import flash_attention
FLASH_ATTENTION_AVAILABLE = True
USE_FLASH_ATTENTION = False

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class XLAFlashAttentionWrapper(nn.Module):
    def __init__(self, original_attention, mesh, partition_spec,rotary_func=apply_rotary_pos_emb):
        super().__init__()
        self.original_attention = original_attention
        self.mesh = mesh
        self.partition_spec = partition_spec
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # Compute groups
        self.head_dim = original_attention.head_dim
        self.hidden_size = original_attention.config.hidden_size
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx
        self.rotatry_func=rotary_func
        USE_FLASH_ATTENTION = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional['Cache'] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )


        if self.num_kv_groups > 1:
            key_states = repeat_kv(key_states, self.num_kv_groups)
            value_states = repeat_kv(value_states, self.num_kv_groups)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

       
        attn_output = flash_attention(
            q=query_states,  # [bsz, num_heads, q_len, head_dim]
            k=key_states,    # [bsz, num_heads, kv_len, head_dim] after repeat
            v=value_states,  # [bsz, num_heads, kv_len, head_dim] after repeat
            causal=True,     
            sm_scale=self.scaling, #here scaling is dim**.5 , but we can apply scaling to q and set it to 1
        )

        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None