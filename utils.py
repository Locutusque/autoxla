from .kernels.attention.flash_attention.fa_xla import _use_flash_attention
from .kernels.attention.splash_attention.sa_xla import _use_splash_attention
from transformers import PreTrainedModel

import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _restore_original_attention(model: PreTrainedModel):
    """
    Scan all modules of `model` and replace any submodule that was wrapped
    with a kernel-wrapper by its `original_attention` attribute.

    The function is architecture-agnostic: it uses `model.named_modules()` to
    discover wrapped modules, locates their parent module, and replaces the
    attribute with the original (eager) attention implementation.

    This will work for wrappers that kept a direct reference at
    `wrapper.original_attention`. If a wrapper used another attribute name
    the wrapper must be adjusted to expose `original_attention`.
    """
    modules: Dict[str, torch.nn.Module] = dict(model.named_modules())

    # collect replacements first to avoid mutating modules() while iterating it
    replacements = []

    for full_name, module in modules.items():
        if hasattr(module, "original_attention"):
            try:
                original = getattr(module, "original_attention")
                # find parent name and attribute name
                if "." in full_name:
                    parent_name, attr_name = full_name.rsplit(".", 1)
                else:
                    parent_name, attr_name = "", full_name

                if parent_name:
                    parent = modules.get(parent_name)
                    if parent is None:
                        # fallback to top-level model
                        parent = model
                else:
                    parent = model

                replacements.append((parent, attr_name, original, full_name))
            except Exception as exc:
                logger.warning("Failed to inspect wrapped module %s: %s", full_name, exc)

    # apply replacements
    for parent, attr_name, original, full_name in replacements:
        try:
            # If attribute exists on parent, replace it
            # This works for standard Modules, ModuleList, Sequential, etc.
            if hasattr(parent, attr_name):
                setattr(parent, attr_name, original)
                logger.info("Restored original attention for %s", full_name)
            else:
                # As a fallback, try setting as attribute anyway (covers dynamic setups)
                setattr(parent, attr_name, original)
                logger.info("Set attribute %s on parent %s as fallback", attr_name, parent)
        except Exception as exc:
            logger.warning(
                "Failed to restore original attention for %s (parent=%s, attr=%s): %s",
                full_name, parent, attr_name, exc
            )


def prepare_model_for_checkpoint(model: PreTrainedModel, dequantize: bool = True):
    """
    Prepare `model` for saving or pushing to the Hub.

    Steps performed:
    1. Move model to CPU (eager kernels require CPU-safe modules).
    2. Restore any attention modules wrapped by Flash/Splash kernel wrappers
       to their `original_attention` (the eager implementation).
    3. Optionally cast model parameters to the dtype of the input embeddings.
       NOTE: this is a best-effort cast and is not a full dequantization for
       library-specific quantized wrappers (bitsandbytes, GPTQ, AWQ, etc.).
       Those require their own library-specific unwrapping.

    :param model: The Hugging Face model instance to prepare.
    :param dequantize: Whether to cast parameters to the embedding dtype.
    :return: The modified model ready for checkpointing.
    """
    # Move to CPU first so any device-specific kernels are inactive
    model = model.cpu()

    # Restore wrapped attention modules regardless of the module path,
    # but only if those kernel wrappers are in use (fast path guard).
    # We call restore if either flag is set OR if some wrappers exist.
    if _use_flash_attention or _use_splash_attention:
        _restore_original_attention(model)
    else:
        # Even if the global flags are false, still attempt a safe sweep:
        # this allows generic unwrapping if wrappers are present.
        _restore_original_attention(model)

    # Best-effort cast to the input-embedding dtype to reduce surprises on CPU.
    if dequantize:
        try:
            input_emb = model.get_input_embeddings()
            if input_emb is not None and hasattr(input_emb, "weight"):
                target_dtype = input_emb.weight.dtype
                # Only cast if target dtype is a floating dtype
                if torch.is_floating_point(torch.tensor(0, dtype=target_dtype)):
                    model.to(dtype = target_dtype)
                else:
                    # If embedding dtype is not floating (unusual), default to float32
                    model.to(dtype = torch.float32)
            else:
                model.to(dtype = torch.float32)
        except Exception as exc:
            logger.warning("Failed to cast model dtype during dequantize step: %s", exc)

    return model
