## Apache License 2.0

import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any
from abc import ABC, abstractmethod
from transformers import PreTrainedModel
from .splash_attention.sa_xla import SplashAttentionConfig
from .splash_attention.models import _get_attention_class_from_model


class BaseAttentionPatcher(ABC):
    """
    Base class for patching Hugging Face language models with custom attention mechanisms.
    
    Supports models like LLaMA, Mistral, Qwen, Gemma, and other decoder-only transformers.
    """
    
    # Mapping of model types to their attention layer paths
    ATTENTION_LAYER_MAPPING = {
        "llama": "model.layers.{}.self_attn",
        "mistral": "model.layers.{}.self_attn",
        "qwen2": "model.layers.{}.self_attn",
        "gemma": "model.layers.{}.self_attn",
        "gemma2": "model.layers.{}.self_attn",
        "phi": "model.layers.{}.self_attn",
        "phi3": "model.layers.{}.self_attn",
        "gpt_neox": "gpt_neox.layers.{}.attention",
        "gptj": "transformer.h.{}.attn",
        "opt": "model.decoder.layers.{}.self_attn",
    }
    
    def __init__(
        self,
        model_type: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        logits_soft_cap: Optional[float] = None,
    ):
        """
        Initialize the attention patcher.
        
        Args:
            model_type: Model architecture type (e.g., 'llama', 'mistral'). 
                       If None, will be auto-detected.
            layer_indices: Specific layer indices to patch. If None, patches all layers.
            logits_soft_cap: Optional soft cap for attention logits (used in Gemma2, etc.).
        """
        self.model_type = model_type
        self.layer_indices = layer_indices
        self.logits_soft_cap = logits_soft_cap
        self.patched_layers = []
        
    def _detect_model_type(self, model: PreTrainedModel) -> str:
        """Auto-detect the model type from the model config."""
        model_type = model.config.model_type
        
        return model_type
    
    def _get_attention_layers(self, model: PreTrainedModel, model_type: str):
        """Get all attention layer modules from the model."""
        attention_pattern = self.ATTENTION_LAYER_MAPPING.get(model_type, "model.layers.{}.self_attn")
        num_layers = model.config.num_hidden_layers
        
        attention_layers = []
        for layer_idx in range(num_layers):
            if self.layer_indices is not None and layer_idx not in self.layer_indices:
                continue
                
            layer_path = attention_pattern.format(layer_idx)
            layer_parts = layer_path.split('.')
            
            # Navigate through the model hierarchy
            current_module = model
            for part in layer_parts:
                if part.isdigit():
                    current_module = current_module[int(part)]
                else:
                    current_module = getattr(current_module, part)
            
            attention_layers.append((layer_idx, layer_path, current_module))
        
        return attention_layers
    
    def _get_rotary_func(self, model_type: str):
        """Get the appropriate rotary embedding function for the model type."""
        try:
            if model_type == "llama":
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            elif model_type == "mistral":
                from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
            elif model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            elif model_type == "gemma" or model_type == "gemma2":
                from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
            elif model_type == "phi" or model_type == "phi3":
                from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
            else:
                # Fallback to a generic implementation
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            
            return apply_rotary_pos_emb
        except ImportError:
            # If specific import fails, use default
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            return apply_rotary_pos_emb
    
    @abstractmethod
    def _create_attention_wrapper(
        self, 
        model: PreTrainedModel,
        original_attention: nn.Module, 
        rotary_func,
        **kwargs
    ) -> nn.Module:
        """
        Create the attention wrapper for a specific attention layer.
        Must be implemented by subclasses.
        
        Args:
            original_attention: The original attention module.
            rotary_func: The rotary embedding function.
            **kwargs: Additional arguments specific to the attention type.
        
        Returns:
            The wrapped attention module.
        """
        pass
    
    @abstractmethod
    def _get_wrapper_name(self) -> str:
        """Get the name of the attention wrapper for logging."""
        pass
    
    def patch_model(self, model: PreTrainedModel, inplace: bool = True) -> PreTrainedModel:
        """
        Patch the model to use custom attention.
        
        Args:
            model: The Hugging Face model to patch.
            inplace: If True, modifies the model in place. If False, creates a copy.
        
        Returns:
            The patched model.
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        
        # Auto-detect model type if not provided
        model_type = self.model_type or self._detect_model_type(model)
        
        # Get rotary embedding function
        rotary_func = self._get_rotary_func(model_type)
        
        # Get attention layers
        attention_layers = self._get_attention_layers(model, model_type)
        
        wrapper_name = self._get_wrapper_name()
        print(f"Patching {len(attention_layers)} attention layers with {wrapper_name}...")
        
        # Replace each attention layer
        for layer_idx, layer_path, original_attention in attention_layers:
            # Create wrapper
            attention_wrapper = self._create_attention_wrapper(
                model,
                original_attention=original_attention,
                rotary_func=rotary_func,
            )
            
            # Replace in model
            layer_parts = layer_path.split('.')
            parent_module = model
            
            for part in layer_parts[:-1]:
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                else:
                    parent_module = getattr(parent_module, part)
            
            setattr(parent_module, layer_parts[-1], attention_wrapper)
            self.patched_layers.append(layer_idx)
        
        print(f"Successfully patched layers: {self.patched_layers}")
        return model


class SplashAttentionPatcher(BaseAttentionPatcher):
    """
    Patches Hugging Face models to use Splash Attention.
    """
    
    def __init__(
        self,
        splash_config: 'SplashAttentionConfig',
        model_type: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        logits_soft_cap: Optional[float] = None,
    ):
        """
        Initialize the Splash Attention patcher.
        
        Args:
            splash_config: Configuration for Splash Attention.
            model_type: Model architecture type. If None, auto-detected.
            layer_indices: Specific layer indices to patch. If None, patches all layers.
            logits_soft_cap: Optional soft cap for attention logits.
        """
        super().__init__(model_type, layer_indices, logits_soft_cap)
        self.splash_config = splash_config
    
    def _create_attention_wrapper(
        self,
        model: PreTrainedModel,
        original_attention: nn.Module, 
        rotary_func,
        **kwargs
    ) -> nn.Module:
        """Create Splash Attention wrapper."""
        attn_class = _get_attention_class_from_model(model)

        
        return attn_class.__init__(
            original_attention=original_attention,
            config=self.splash_config,
            logits_soft_cap=self.logits_soft_cap,
        )
    
    def _get_wrapper_name(self) -> str:
        return "Splash Attention"
    
    @staticmethod
    def from_model_config(
        model: PreTrainedModel,
        layer_indices: Optional[List[int]] = None,
        **splash_kwargs
    ) -> 'SplashAttentionPatcher':
        """
        Create a patcher with auto-configured Splash Attention settings.
        
        Args:
            model: The model to configure for.
            block_size: Block size for Splash Attention.
            mask_k: Mask parameter for Splash Attention.
            layer_indices: Specific layers to patch.
            **splash_kwargs: Additional SplashAttentionConfig parameters.
        
        Returns:
            A configured SplashAttentionPatcher instance.
        """
        from .splash_attention.sa_kernel import SplashAttentionConfig
        
        splash_config = SplashAttentionConfig(
            **splash_kwargs
        )
        
        # Auto-detect soft cap for certain models
        logits_soft_cap = None
        if hasattr(model.config, 'attn_logit_softcapping'):
            logits_soft_cap = model.config.attn_logit_softcapping
        
        return SplashAttentionPatcher(
            splash_config=splash_config,
            layer_indices=layer_indices,
            logits_soft_cap=logits_soft_cap,
        )


class XLAFlashAttentionPatcher(BaseAttentionPatcher):
    """
    Patches Hugging Face models to use XLA Flash Attention.
    """
    
    def __init__(
        self,
        mesh: Any,
        partition_spec: Any,
        model_type: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        logits_soft_cap: Optional[float] = None,
    ):
        """
        Initialize the XLA Flash Attention patcher.
        
        Args:
            mesh: XLA mesh for distributed computation.
            partition_spec: XLA partition specification.
            model_type: Model architecture type. If None, auto-detected.
            layer_indices: Specific layer indices to patch. If None, patches all layers.
            logits_soft_cap: Optional soft cap for attention logits.
        """
        super().__init__(model_type, layer_indices, logits_soft_cap)
        self.mesh = mesh
        self.partition_spec = partition_spec
    
    def _create_attention_wrapper(
        self, 
        model: PreTrainedModel,
        original_attention: nn.Module, 
        rotary_func,
        **kwargs
    ) -> nn.Module:
        """Create XLA Flash Attention wrapper."""
        from .flash_attention.fa_xla import XLAFlashAttentionWrapper
        
        return XLAFlashAttentionWrapper(
            original_attention=original_attention,
            mesh=self.mesh,
            partition_spec=self.partition_spec,
            rotary_func=rotary_func,
        )
    
    def _get_wrapper_name(self) -> str:
        return "XLA Flash Attention"
    
    @staticmethod
    def from_model_config(
        model: PreTrainedModel,
        mesh: Any,
        partition_spec: Any,
        layer_indices: Optional[List[int]] = None,
    ) -> 'XLAFlashAttentionPatcher':
        """
        Create a patcher with auto-configured XLA Flash Attention settings.
        
        Args:
            model: The model to configure for.
            mesh: XLA mesh for distributed computation.
            partition_spec: XLA partition specification.
            layer_indices: Specific layers to patch.
        
        Returns:
            A configured XLAFlashAttentionPatcher instance.
        """
        # Auto-detect soft cap for certain models
        logits_soft_cap = None
        if hasattr(model.config, 'attn_logit_softcapping'):
            logits_soft_cap = model.config.attn_logit_softcapping
        
        return XLAFlashAttentionPatcher(
            mesh=mesh,
            partition_spec=partition_spec,
            layer_indices=layer_indices,
            logits_soft_cap=logits_soft_cap,
        )


# Convenience functions
def apply_splash_attention(
    model: PreTrainedModel,
    splash_config: Optional['SplashAttentionConfig'] = None,
    block_size: int = 512,
    mask_k: int = 512,
    layer_indices: Optional[List[int]] = None,
    inplace: bool = True,
    **splash_kwargs
) -> PreTrainedModel:
    """
    Convenience function to quickly apply Splash Attention to a model.
    
    Args:
        model: The Hugging Face model to patch.
        splash_config: Pre-configured SplashAttentionConfig. If None, creates one.
        block_size: Block size for Splash Attention (if splash_config is None).
        mask_k: Mask parameter (if splash_config is None).
        layer_indices: Specific layers to patch.
        inplace: Modify model in place.
        **splash_kwargs: Additional config parameters.
    
    Returns:
        The patched model.
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = apply_splash_attention(model, block_size=512, mask_k=512)
    """
    if splash_config is None:
        from .sa_kernel import SplashAttentionConfig
        splash_config = SplashAttentionConfig(
            block_size=block_size,
            mask_k=mask_k,
            **splash_kwargs
        )
    
    patcher = SplashAttentionPatcher(
        splash_config=splash_config,
        layer_indices=layer_indices,
    )
    
    return patcher.patch_model(model, inplace=inplace)


def apply_xla_flash_attention(
    model: PreTrainedModel,
    mesh: Any,
    partition_spec: Any,
    layer_indices: Optional[List[int]] = None,
    inplace: bool = True,
) -> PreTrainedModel:
    """
    Convenience function to quickly apply XLA Flash Attention to a model.
    
    Args:
        model: The Hugging Face model to patch.
        mesh: XLA mesh for distributed computation.
        partition_spec: XLA partition specification.
        layer_indices: Specific layers to patch.
        inplace: Modify model in place.
    
    Returns:
        The patched model.
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> import torch_xla.distributed.spmd as xs
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> mesh = xs.get_global_mesh()
        >>> partition_spec = xs.PartitionSpec(...)
        >>> model = apply_xla_flash_attention(model, mesh, partition_spec)
    """
    patcher = XLAFlashAttentionPatcher(
        mesh=mesh,
        partition_spec=partition_spec,
        layer_indices=layer_indices,
    )
    
    return patcher.patch_model(model, inplace=inplace)
