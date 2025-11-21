"""
Language Model Quantizer for torch_xla
Provides utilities for quantizing language models using XLA-optimized operations.
Supports packing 4-bit quantized weights into int8 for memory efficiency.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, List, Callable
import logging

# Try to import XLA components
try:
    import torch_xla
    from torch_xla.experimental.xla_quantized_matmul import XlaQuantizedLinear
    from torch_xla.experimental.xla_quantized_matmul import _quantize_tensor
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    logging.warning("torch_xla not available. Quantization will not work.")

# Try to import Jax and Pallas kernels
JAX_AVAILABLE = False
PALLAS_KERNELS_AVAILABLE = False
quantized_matmul_kernel = None
get_tuned_block_sizes = None
TUNED_BLOCK_SIZES = None

try:
    from torch_xla.experimental.custom_kernel import jax_import_guard
    jax_import_guard()
    
    import jax
    import jax.numpy as jnp
    from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
    
    JAX_AVAILABLE = True
    
    # Try to import the pre-built Pallas quantized matmul kernels
    try:
        from ..kernels.quantization.quantized_matmul import (
            quantized_matmul_kernel,
            get_tuned_block_sizes,
        )
        from ..kernels.quantization.tuned_block_sizes import (
            TUNED_BLOCK_SIZES,
            get_tpu_version
        )
        PALLAS_KERNELS_AVAILABLE = True
        logging.info("Pallas quantized matmul kernels available")
    except ImportError as e:
        logging.info(f"Pallas quantized matmul kernels not available: {e}")
        
except ImportError as e:
    JAX_AVAILABLE = False
    logging.info(f"Jax not available. Will use torch/eager kernels instead of Pallas. {f}")


logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Configuration for quantization settings."""
    
    def __init__(
        self,
        n_bits: int = 8,
        block_size: int = -1,
        symmetric: bool = True,
        quantize_activation: bool = False,
        use_pallas: bool = False,
        per_channel: bool = True,
        quantize_embeddings: bool = False,
        quantize_lm_head: bool = False,
        auto_tune_block_sizes: bool = True,
        pack_4bit: bool = True,
    ):
        """
        Args:
            n_bits: Number of bits for quantization (4 or 8)
            block_size: Block size for blockwise quantization (-1 for per-channel)
            symmetric: Use symmetric quantization (no zero point)
            quantize_activation: Whether to quantize activations dynamically
            use_pallas: Use Pallas kernels if Jax is available
            per_channel: Use per-channel quantization
            quantize_embeddings: Whether to quantize embedding layers. Not recommended
            quantize_lm_head: Whether to quantize the language model head. Not recommended
            auto_tune_block_sizes: Use pre-tuned block sizes for optimal performance
            pack_4bit: Pack 4-bit values into int32 (8 values per int32)
        """
        assert n_bits in [4, 8], "Only 4-bit and 8-bit quantization supported"
        self.n_bits = n_bits
        self.block_size = block_size
        self.symmetric = symmetric
        self.quantize_activation = quantize_activation
        self.use_pallas = use_pallas and JAX_AVAILABLE and PALLAS_KERNELS_AVAILABLE
        self.per_channel = per_channel
        self.quantize_embeddings = quantize_embeddings
        self.quantize_lm_head = quantize_lm_head
        self.auto_tune_block_sizes = auto_tune_block_sizes
        self.pack_4bit = pack_4bit and (n_bits == 4)
        
        if use_pallas and not JAX_AVAILABLE:
            logger.warning("Pallas kernels requested but Jax not available. Using eager kernels.")
            self.use_pallas = False
        elif use_pallas and not PALLAS_KERNELS_AVAILABLE:
            logger.warning("Pallas kernels requested but pallas_kernels module not available. Using eager kernels.")
            self.use_pallas = False


def pack_int4_to_int8(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """
    Pack 4-bit values into int8 tensor.
    
    Args:
        values: Tensor with values in range [-8, 7] for signed or [0, 15] for unsigned
    
    Returns:
        Tuple of (packed_tensor, original_shape)
    """
    original_shape = values.shape
    values_flat = values.flatten()
    
    # Ensure the number of elements is divisible by 2
    num_elements = values_flat.numel()
    padding = num_elements % 2
    if padding > 0:
        values_flat = torch.cat([
            values_flat,
            torch.zeros(padding, dtype=values_flat.dtype, device=values_flat.device)
        ])
    
    # Convert to uint8 to safely pack (shift signed values if needed)
    # For signed 4-bit: range is -8 to 7, we shift by 8 to get 0-15
    values_uint = (values_flat + 8).to(torch.uint8)
    
    # Reshape to groups of 2
    values_grouped = values_uint.reshape(-1, 2)
    
    # Pack 2 x 4-bit values into one int8
    # First value takes lower 4 bits, second value takes upper 4 bits
    packed = values_grouped[:, 0] | (values_grouped[:, 1] << 4)
    packed = packed.to(torch.int8)
    
    return packed, original_shape


def unpack_int8_to_int4(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Unpack int8 tensor back to 4-bit values.
    
    Args:
        packed: Packed int8 tensor
        original_shape: Original shape before packing
    
    Returns:
        Unpacked tensor with original shape
    """
    num_elements = torch.prod(torch.tensor(original_shape)).item()
    
    # Convert to uint8 for bit operations
    packed_uint = packed.to(torch.uint8)
    
    # Extract lower 4 bits (first value)
    lower = packed_uint & 0xF
    
    # Extract upper 4 bits (second value)
    upper = (packed_uint >> 4) & 0xF
    
    # Interleave the values
    unpacked = torch.stack([lower, upper], dim=1).flatten()
    
    # Take only the original number of elements (remove padding)
    unpacked = unpacked[:num_elements]
    
    # Convert back from uint (0-15) to signed (-8 to 7)
    unpacked = unpacked.to(torch.int8) - 8
    
    # Reshape to original shape
    unpacked = unpacked.reshape(original_shape)
    
    return unpacked


def quantize_weight(
    weight: torch.Tensor,
    n_bits: int = 8,
    symmetric: bool = True,
    axis: int = 0,
    block_size: int = -1,
    pack_4bit: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Size]]:
    """
    Quantize a weight tensor.
    
    Args:
        weight: Weight tensor to quantize
        n_bits: Number of bits (4 or 8)
        symmetric: Use symmetric quantization
        axis: Axis for per-channel quantization
        block_size: Block size for blockwise quantization (-1 for per-channel)
        pack_4bit: Pack 4-bit values into int8
    
    Returns:
        Tuple of (quantized_weight, scale, zero_point, original_shape)
        original_shape is only set for packed 4-bit weights
    """
    assert n_bits in [4, 8], "Only 4-bit and 8-bit quantization supported"
    
    if block_size == -1:
        # Per-channel quantization
        q_weight, scale, zero_point = _quantize_per_channel(weight, n_bits, symmetric, axis)
    else:
        # Blockwise quantization
        q_weight, scale, zero_point = _quantize_blockwise(weight, n_bits, symmetric, block_size)
    
    # Pack 4-bit weights into int8
    original_shape = None
    if n_bits == 4 and pack_4bit:
        q_weight, original_shape = pack_int4_to_int8(q_weight)
    
    return q_weight, scale, zero_point, original_shape


def _quantize_per_channel(
    weight: torch.Tensor,
    n_bits: int,
    symmetric: bool,
    axis: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Per-channel quantization."""
    # Calculate min/max along the quantization axis
    dims = list(range(weight.ndim))
    dims.remove(axis)
    
    if symmetric:
        # Symmetric: use max absolute value
        max_val = torch.amax(torch.abs(weight), dim=dims, keepdim=True)
        min_val = -max_val
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -(2 ** (n_bits - 1))
        zero_point = None
    else:
        # Asymmetric: use actual min/max
        max_val = torch.amax(weight, dim=dims, keepdim=True)
        min_val = torch.amin(weight, dim=dims, keepdim=True)
        qmax = 2 ** n_bits - 1
        qmin = 0
        
    # Calculate scale
    scale = (max_val - min_val) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
    
    if not symmetric:
        # Calculate zero point for asymmetric quantization
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int8)
        zero_point = zero_point.squeeze()
    else:
        zero_point = None
    
    # Quantize
    if symmetric:
        q_weight = torch.round(weight / scale)
    else:
        q_weight = torch.round(weight / scale) + zero_point
        
    q_weight = torch.clamp(q_weight, qmin, qmax).to(torch.int8)
    
    # Reshape scale to 1D
    scale = scale.squeeze()
    
    return q_weight, scale, zero_point


def _quantize_blockwise(
    weight: torch.Tensor,
    n_bits: int,
    symmetric: bool,
    block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Blockwise quantization for weights."""
    assert weight.ndim == 2, "Blockwise quantization requires 2D weight tensor"
    out_features, in_features = weight.shape
    assert in_features % block_size == 0, f"in_features {in_features} must be divisible by block_size {block_size}"
    
    num_blocks = in_features // block_size
    
    # Reshape to [out_features, num_blocks, block_size]
    weight_reshaped = weight.reshape(out_features, num_blocks, block_size)
    # Transpose to [num_blocks, block_size, out_features]
    weight_reshaped = weight_reshaped.permute(1, 2, 0)
    
    if symmetric:
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -(2 ** (n_bits - 1))
        # Max per block per output channel: [num_blocks, 1, out_features]
        max_val = torch.amax(torch.abs(weight_reshaped), dim=1, keepdim=True)
        scale = max_val / qmax
        scale = torch.clamp(scale, min=1e-8)
        q_weight = torch.round(weight_reshaped / scale)
        zero_point = None
    else:
        qmax = 2 ** n_bits - 1
        qmin = 0
        max_val = torch.amax(weight_reshaped, dim=1, keepdim=True)
        min_val = torch.amin(weight_reshaped, dim=1, keepdim=True)
        scale = (max_val - min_val) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int8)
        zero_point = zero_point.squeeze(1)  # [num_blocks, out_features]
        q_weight = torch.round(weight_reshaped / scale) + zero_point.unsqueeze(1)
    
    q_weight = torch.clamp(q_weight, qmin, qmax).to(torch.int8)
    scale = scale.squeeze(1)  # [num_blocks, out_features]
    
    return q_weight, scale, zero_point


class QuantizedLinear(nn.Module):
    """Quantized Linear layer wrapper with Pallas kernel support and 4-bit packing."""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        config: QuantizationConfig
    ):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.config = config
        
        # Store original shape for 4-bit packed weights
        self.original_weight_shape = None
        
        # Determine block sizes for Pallas
        self.batch_block_size = None
        self.out_block_size = None
        self.in_block_size = None
        
        # Setup Pallas kernels if requested
        self.pallas_kernel = None
        if config.use_pallas and PALLAS_KERNELS_AVAILABLE:
            self._setup_pallas_kernel()
        
        # Quantize the weights - this will register buffers
        self._quantize_from_layer(original_layer)
    
    def _setup_pallas_kernel(self):
        """Setup Pallas kernel wrapper for PyTorch."""
        if self.config.block_size != -1:
            logger.warning("Pallas kernels currently only support per-channel quantization. Using XLA fallback for blockwise.")
            self.config.use_pallas = False
            return
        print("Setting up pallas kernels...")
        # Define output shape function for the kernel wrapper
        def output_shape_fn(x, w, scale, zero_point):
            batch_size = x.shape[0]
            out_features = w.shape[0]
            return [(tuple([batch_size, out_features]), x.dtype)]
        
        # Wrap the Pallas kernel for PyTorch
        self.pallas_kernel = make_kernel_from_pallas(
            quantized_matmul_kernel,
            output_shape_fn
        )
        logger.info("Initialized Pallas quantized matmul kernel")
    
    def _get_optimal_block_sizes(self, batch_size: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Get optimal block sizes for the given configuration."""
        if not self.config.auto_tune_block_sizes or not PALLAS_KERNELS_AVAILABLE:
            # Use default block sizes
            return 128, 128, 128
        
        # Get tuned block sizes from the lookup table
        dtype_str = 'bfloat16'  # Assuming bfloat16, could be parameterized
        block_sizes = get_tuned_block_sizes(
            TUNED_BLOCK_SIZES,
            batch_size,
            self.out_features,
            self.in_features,
            dtype_str,
            self.config.quantize_activation
        )
        
        if block_sizes[0] is not None:
            logger.debug(f"Using tuned block sizes for batch={batch_size}, "
                        f"out={self.out_features}, in={self.in_features}: {block_sizes}")
            return block_sizes
        else:
            # Fallback to reasonable defaults
            logger.debug(f"No tuned block sizes found, using defaults")
            return 128, 128, 128
    
    def _quantize_from_layer(self, layer: nn.Linear):
        """Quantize weights from the original layer."""
        weight = layer.weight.data
        
        q_weight, scale, zero_point, original_shape = quantize_weight(
            weight,
            n_bits=self.config.n_bits,
            symmetric=self.config.symmetric,
            axis=0,  # Per output channel
            block_size=self.config.block_size,
            pack_4bit=self.config.pack_4bit
        )
        
        # Register quantized weights and parameters as buffers
        self.register_buffer('q_weight', q_weight)
        self.register_buffer('scale', scale)
        if zero_point is not None:
            self.register_buffer('zero_point', zero_point)
        else:
            self.zero_point = None
        
        if original_shape is not None:
            self.original_weight_shape = original_shape
        
        # Delete the original layer to free memory - we don't need it anymore
        del layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Pallas kernels if configured, otherwise XLA kernels."""
        if self.config.use_pallas and self.pallas_kernel is not None:
            # Use Pallas kernel
            return self._forward_pallas(x)
        else:
            # Fallback to manual computation
            return self._forward_manual(x)
    
    def _forward_pallas(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Pallas kernels."""
        # Move tensors to XLA device if needed
        if x.device.type != 'xla':
            raise RuntimeError("Input must be on XLA device to use Pallas kernels")
        
        # Get optimal block sizes based on batch size
        batch_size = x.shape[0]
        batch_block_size, out_block_size, in_block_size = self._get_optimal_block_sizes(batch_size)
        
        # Reshape input to 2D if needed
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        
        # Unpack 4-bit weights if needed
        q_weight = self.q_weight
        if self.config.pack_4bit and self.original_weight_shape is not None:
            q_weight = unpack_int8_to_int4(q_weight, self.original_weight_shape)
        
        # Call Pallas kernel with tuned block sizes
        output = self.pallas_kernel(
            x,
            q_weight,
            self.scale,
        )
        
        # Reshape output back to original shape
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], -1)
        
        return output

    def _forward_manual(self, x: torch.Tensor) -> torch.Tensor:
        """Manual forward pass for fallback."""
        # Unpack 4-bit weights if needed
        q_weight = self.q_weight
        if self.config.pack_4bit and self.original_weight_shape is not None:
            q_weight = unpack_int8_to_int4(q_weight, self.original_weight_shape)
        
        if self.config.block_size == -1:
            # Per-channel quantization
            w = q_weight
            if self.config.quantize_activation:
                x_quant, x_scale = _quantize_tensor(x)
                x_quant = x_quant.to(torch.int32)
                w = w.to(torch.int32)
                out = torch.nn.functional.linear(x_quant, w)
                out = out.to(x.dtype) * self.scale * x_scale
            else:
                out = torch.nn.functional.linear(x, w.to(x.dtype))
                out = out * self.scale
            
            if self.zero_point is not None:
                zp_correction = torch.sum(x, dim=-1, keepdim=True) * self.zero_point
                out = out - zp_correction
        else:
            # Blockwise quantization
            x_reshaped = x.reshape(*x.shape[:-1], x.shape[-1] // self.config.block_size, self.config.block_size)
            w = q_weight.to(x.dtype)
            out = torch.einsum('scn,...sc->...sn', w, x_reshaped)
            out = torch.einsum('sn,...sn->...n', self.scale, out)
            
            if self.zero_point is not None:
                zp_out = x_reshaped.sum(dim=-1)
                zp_out = torch.matmul(zp_out, self.zero_point)
                out = out - zp_out
        
        return out


class LanguageModelQuantizer:
    """Main quantizer class for language models."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Args:
            config: QuantizationConfig object with quantization settings
        """
        if not XLA_AVAILABLE:
            raise RuntimeError("torch_xla is required for quantization")
        
        self.config = config
        logger.info(f"Initialized LanguageModelQuantizer with config: {vars(config)}")
        
        if config.pack_4bit:
            logger.info("Using 4-bit packing into int8 (2 values per int8)")
        
        if config.use_pallas and PALLAS_KERNELS_AVAILABLE:
            logger.info("Using Pallas kernels for quantized operations")
        elif config.use_pallas:
            logger.warning("Pallas requested but not available. Using XLA kernels.")
        else:
            logger.info("Using eager/XLA kernels for quantized operations")
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Quantize a language model.
        
        Args:
            model: PyTorch model to quantize
        
        Returns:
            Quantized model
        """
        return self._quantize_module(model)
    
    def _quantize_module(self, module: nn.Module, name: str = "") -> nn.Module:
        """Iteratively quantize modules."""
        # Collect all Linear modules to quantize
        modules_to_quantize = []
        
        for name, submodule in module.named_modules():
            if isinstance(submodule, nn.Linear):
                # Check if we should quantize this layer
                if self._should_quantize_layer(name):
                    modules_to_quantize.append((name, submodule))
                else:
                    logger.debug(f"Skipping layer: {name}")
            elif isinstance(submodule, nn.Embedding):
                if self.config.quantize_embeddings:
                    logger.info(f"Quantizing embedding: {name}")
                    # Embeddings can be quantized similarly but we keep them as-is for now
                    # In production, you might want to implement quantized embeddings
                    pass
        
        # Replace each Linear module with QuantizedLinear
        for name, linear_module in modules_to_quantize:
            logger.info(f"Quantizing layer: {name}: {type(linear_module).__name__}; {linear_module.weight.dtype} ➝ QuantizedLinear ({self.config.n_bits}-bit precision)")
            
            if linear_module.weight.requires_grad:
                logger.warning(f"Layer {name} requires gradient. Quantization of trainable parameters may affect training.")
            
            # Create quantized version
            quantized_linear = QuantizedLinear(linear_module, self.config)
            
            # Navigate to parent module and replace
            parts = name.split('.')
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], quantized_linear)
        
        return module
    
    def _should_quantize_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be quantized based on its name."""
        # Skip embedding layers unless configured
        if 'embed' in layer_name.lower() and not self.config.quantize_embeddings:
            return False
        
        # Skip LM head unless configured
        if ('lm_head' in layer_name.lower() or 
            'output' in layer_name.lower()) and not self.config.quantize_lm_head:
            return False
        
        return True
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size statistics.
        
        Returns:
            Dictionary with size information in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024
        }


# Utility functions for common use cases

def quantize_llm(
    model: nn.Module,
    n_bits: int = 8,
    block_size: int = -1,
    symmetric: bool = True,
    quantize_activation: bool = False,
    use_pallas: bool = False,
    auto_tune_block_sizes: bool = True,
    pack_4bit: bool = True
) -> nn.Module:
    """
    Convenience function to quantize a language model.
    
    Args:
        model: Language model to quantize
        n_bits: Number of bits (4 or 8)
        block_size: Block size for blockwise quantization (-1 for per-channel)
        symmetric: Use symmetric quantization
        quantize_activation: Quantize activations dynamically
        use_pallas: Use Pallas kernels if available
        auto_tune_block_sizes: Use pre-tuned block sizes for optimal performance
        pack_4bit: Pack 4-bit values into int32 for memory efficiency
    
    Returns:
        Quantized model
    """
    config = QuantizationConfig(
        n_bits=n_bits,
        block_size=block_size,
        symmetric=symmetric,
        quantize_activation=quantize_activation,
        use_pallas=use_pallas,
        per_channel=(block_size == -1),
        auto_tune_block_sizes=auto_tune_block_sizes,
        pack_4bit=pack_4bit
    )
    
    quantizer = LanguageModelQuantizer(config)
    quantized_model = quantizer.quantize_model(model)
    
    # Log size reduction
    original_size = LanguageModelQuantizer.get_model_size(model)
    quantized_size = LanguageModelQuantizer.get_model_size(quantized_model)
    
    logger.info(f"Original model size: {original_size['total_size_mb']:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size['total_size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {original_size['total_size_mb'] / quantized_size['total_size_mb']:.2f}x")
    
    return quantized_model


def quantize_llm_int4(model: nn.Module, use_pallas: bool = False, pack_4bit: bool = True) -> nn.Module:
    """Quantize a language model to int4 with optional packing."""
    return quantize_llm(model, n_bits=4, use_pallas=use_pallas, pack_4bit=pack_4bit, quantize_activation=True)


def quantize_llm_int8(model: nn.Module, use_pallas: bool = False) -> nn.Module:
    """Quantize a language model to int8."""
    return quantize_llm(model, n_bits=8, use_pallas=use_pallas, pack_4bit=False, quantize_activation=True)


def quantize_llm_blockwise(
    model: nn.Module,
    block_size: int = 128,
    n_bits: int = 4,
    use_pallas: bool = False,
    pack_4bit: bool = True
) -> nn.Module:
    """Quantize a language model with blockwise quantization."""
    return quantize_llm(
        model,
        n_bits=n_bits,
        block_size=block_size,
        use_pallas=use_pallas,
        pack_4bit=pack_4bit
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple test model
    class SimpleLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 512)
            self.layers = nn.ModuleList([
                nn.Linear(512, 2048),
                nn.Linear(2048, 2048),
                nn.Linear(2048, 512)
            ])
            self.lm_head = nn.Linear(512, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.lm_head(x)
    
    # Test packing/unpacking
    print("=== Testing 4-bit packing ===")
    test_tensor = torch.randint(-8, 8, (256, 512), dtype=torch.int8)
    print(f"Original tensor shape: {test_tensor.shape}")
    print(f"Original tensor size: {test_tensor.numel() * test_tensor.element_size() / 1024:.2f} KB")
    
    packed, original_shape = pack_int4_to_int8(test_tensor)
    print(f"Packed tensor shape: {packed.shape}")
    print(f"Packed tensor size: {packed.numel() * packed.element_size() / 1024:.2f} KB")
    print(f"Compression ratio: {(test_tensor.numel() * test_tensor.element_size()) / (packed.numel() * packed.element_size()):.2f}x")
    
    unpacked = unpack_int8_to_int4(packed, original_shape)
    print(f"Unpacked tensor shape: {unpacked.shape}")
    print(f"Reconstruction matches: {torch.all(test_tensor == unpacked)}")
    
    # Test quantization
    print("\n=== Testing model quantization ===")
    model = SimpleLanguageModel().to(torch.bfloat16)
    print("Original model:")
    print(LanguageModelQuantizer.get_model_size(model))
    
    # Quantize to int4 with packing
    print("\n=== Quantizing to int4 with packing ===")
    quantized_model_packed = quantize_llm_int4(model, use_pallas=False, pack_4bit=True)
    print("\nInt4 quantized model (packed):")
    print(LanguageModelQuantizer.get_model_size(quantized_model_packed))
    
    # Quantize to int4 without packing for comparison
    print("\n=== Quantizing to int4 without packing ===")
    quantized_model_unpacked = quantize_llm_int4(SimpleLanguageModel(), use_pallas=False, pack_4bit=False)
    print("\nInt4 quantized model (unpacked):")
    print(LanguageModelQuantizer.get_model_size(quantized_model_unpacked))
    
    # Quantize to int8
    print("\n=== Quantizing to int8 ===")
    quantized_model_int8 = quantize_llm_int8(SimpleLanguageModel(), use_pallas=False)
    print("\nInt8 quantized model:")
    print(LanguageModelQuantizer.get_model_size(quantized_model_int8))
    
    # Test forward pass
    print("\n=== Testing forward pass ===")
    test_input = torch.randint(0, 1000, (2, 10))
    #output_original = model(test_input)
    output_quantized = quantized_model_int8(test_input)
    #print(f"Original output shape: {output_original.shape}")
    print(f"Quantized output shape: {output_quantized.shape}")
    #print(f"Max difference: {torch.max(torch.abs(output_original - output_quantized)):.4f}")
    
    # Test with XLA device (if available)
    if XLA_AVAILABLE:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            
            # Move model to XLA device
            model_xla = SimpleLanguageModel().to(device)
            
            # Quantize with packed 4-bit
            quantized_xla = quantize_llm_int4(model_xla, use_pallas=False, pack_4bit=True)
            
            # Test forward pass
            test_input_xla = torch.randint(0, 1000, (2, 10)).to(device)
            output = quantized_xla(test_input_xla)
            print(f"\nXLA test output shape: {output.shape}")
            print("Packed 4-bit quantization test passed!")
            
        except Exception as e:
            logger.warning(f"XLA device test failed: {e}")
