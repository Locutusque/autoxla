from .kernels.attention.flash_attention.fa_xla import _use_flash_attention
from .kernels.attention.splash_attention.sa_xla import _use_splash_attention
from transformers import PreTrainedModel

import torch


def _restore_original_attention(model: PreTrainedModel):
    """
    Restores any attention modules that were patched with Flash or Splash
    attention kernels by replacing them with their backed-up
    `original_attention` attribute.

    This function recursively scans all submodules and restores any
    module that has an `original_attention` attribute.
    """
    modules = dict(model.named_modules())

    for name, module in modules.items():
        if hasattr(module, "original_attention"):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            if parent_name:
                parent = modules[parent_name]
            else:
                parent = model

            setattr(parent, attr_name, module.original_attention)


def prepare_model_for_checkpoint(model: PreTrainedModel, dequantize: bool = True):
    """
    Prepares a model for being checkpointed or pushed to the Hub. The
    model is moved to CPU, patched attention kernels are restored, and
    the model is optionally dequantized.

    :param model: The model to prepare for saving.
    :type model: PreTrainedModel
    :param dequantize: Whether to cast parameters to the embedding
        dtype. This does NOT fully dequantize quantized models but
        ensures consistent CPU-safe types.
    :type dequantize: bool
    """
    model = model.cpu()

    if _use_flash_attention or _use_splash_attention:
        _restore_original_attention(model)

    if dequantize:
        target_dtype = model.get_input_embeddings().weight.dtype
        model.to(dtype = target_dtype)

    return model
