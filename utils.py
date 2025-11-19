from .kernels.attention.flash_attention.fa_xla import _use_flash_attention
from .kernels.attention.splash_attention.sa_xla import _use_splash_attention
from transformers import PreTrainedModel

import torch

def prepare_model_for_checkpoint(model: PreTrainedModel, dequantize: bool = True):
    """
    Prepares a given model to be saved or pushed to the hugging face
    hub. Moves the model to the CPU, unloads attention kernels, and
    dequantizes the model.
    
    :param model: The model that needs to be prepared for saving.
    :type model: PreTrainedModel
    """
    model = model.cpu()
    if _use_flash_attention:
        for layer in model.model.layers:
            kernel_to_remove = layer.self_attn
            layer.self_attn = kernel_to_remove.original_attention
    elif _use_splash_attention:
        for layer in model.model.layers:
            kernel_to_remove = layer.self_attn
            layer.self_attn = kernel_to_remove.original_attention
    if dequantize:
        model.to(model.get_input_embeddings().weight.dtype)
    return model
