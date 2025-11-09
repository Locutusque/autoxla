from .kernels.attention.flash_attention.fa_xla import USE_FLASH_ATTENTION
from .kernels.attention.splash_attention.sa_xla import USE_SPLASH_ATTENTION
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
    if USE_FLASH_ATTENTION:
        for layer in model.model.layers:
            kernel_to_remove = layer.self_attn
            layer.self_attn = kernel_to_remove.original_attention
    elif USE_SPLASH_ATTENTION:
        for layer in model.model.layers:
            kernel_to_remove = layer.self_attn
            layer.self_attn = kernel_to_remove.original_attention
    if dequantize:
        model.to(model.get_input_embeddings().weight.dtype)
    return model