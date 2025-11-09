import os
from typing import Optional, Union, Callable
from functools import partial
from transformers import AutoModelForCausalLM
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from ..quantization import QuantizationConfig, LanguageModelQuantizer
from ..kernels.attention.impl_utils import (
    SplashAttentionPatcher, 
    XLAFlashAttentionPatcher,
)
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
import torch
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch_xla.core.xla_model as xm
import numpy as np
from torch_xla.utils.checkpoint import checkpoint

USE_FSDP_V2 = False


class AutoXLAModelForCausalLM(object):
    _model_mapping = _LazyAutoMapping(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, CONFIG_MAPPING_NAMES)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _create_mesh(n_devices, strategy="fsdp"):
        """
        Create device mesh based on the specified sharding strategy.
        
        Args:
            n_devices: Number of available devices
            strategy: Sharding strategy - "fsdp", "dp" (data parallel), 
                     "mp" (model parallel), or "2d" (2D sharding)
        
        Returns:
            xs.Mesh object configured for the specified strategy
        """
        device_ids = np.arange(n_devices)
        
        if strategy.lower() == "fsdp":
            # Fully Sharded Data Parallel - all devices on fsdp axis
            mesh_shape = (n_devices, 1)
            mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
            
        elif strategy.lower() == "dp" or strategy.lower() == "data_parallel":
            # Data Parallel - all devices on data axis
            mesh_shape = (n_devices,)
            mesh = xs.Mesh(device_ids, mesh_shape, ('data',))
            
        elif strategy.lower() == "mp" or strategy.lower() == "model_parallel":
            # Model Parallel - all devices on model axis
            mesh_shape = (1, n_devices)
            mesh = xs.Mesh(device_ids, mesh_shape, ('data', 'model'))
            
        elif strategy.lower() == "2d":
            # 2D sharding - balanced across data and model axes
            # Try to create a roughly square mesh
            if n_devices == 1:
                mesh_shape = (1, 1)
            elif n_devices == 2:
                mesh_shape = (2, 1)
            elif n_devices == 4:
                mesh_shape = (2, 2)
            elif n_devices == 8:
                mesh_shape = (4, 2)
            elif n_devices == 16:
                mesh_shape = (4, 4)
            else:
                # For other device counts, try to find good factors
                data_axis = int(np.sqrt(n_devices))
                while n_devices % data_axis != 0:
                    data_axis -= 1
                model_axis = n_devices // data_axis
                mesh_shape = (data_axis, model_axis)
                
            mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
            
        elif strategy.lower() == "3d":
            # 3D sharding with data, fsdp, and model parallelism
            # Requires careful factorization
            if n_devices < 8:
                raise ValueError(f"3D sharding requires at least 8 devices, got {n_devices}")
            
            # Try to create balanced 3D mesh
            if n_devices == 8:
                mesh_shape = (2, 2, 2)
            elif n_devices == 16:
                mesh_shape = (2, 4, 2)
            elif n_devices == 32:
                mesh_shape = (2, 4, 4)
            else:
                # Heuristic for other sizes
                cube_root = int(np.cbrt(n_devices))
                data_axis = cube_root
                remaining = n_devices // data_axis
                fsdp_axis = int(np.sqrt(remaining))
                while remaining % fsdp_axis != 0:
                    fsdp_axis -= 1
                model_axis = remaining // fsdp_axis
                mesh_shape = (data_axis, fsdp_axis, model_axis)
            
            mesh = xs.Mesh(device_ids, mesh_shape, ('data', 'fsdp', 'model'))
        else:
            raise NotImplementedError(
                f"The sharding strategy '{strategy}' is not implemented. "
                f"Choose from: 'fsdp', 'dp', 'mp', '2d', '3d'"
            )
        
        if xm.is_master_ordinal():
            xm.master_print(f"> Created {strategy.upper()} mesh with shape {mesh_shape}")
        xs.set_global_mesh(mesh)
        return mesh

    @staticmethod
    def _partition_model(model, device, mesh, strategy="fsdp", verbose=True):
        """
        Manually partition model parameters and buffers based on sharding strategy.
        This is applied before wrapping with FSDPv2.
        """
        model.to(device)
        
        # Partition parameters
        for name, param in model.named_parameters():
            # Skip 1D tensors (bias, layernorm, etc.)
            if len(param.shape) == 1:
                continue
            
            partition_spec = AutoXLAModelForCausalLM._get_partition_spec(name, param.shape, strategy)
            if partition_spec is None:
                continue
            
            # Apply sharding
            xs.mark_sharding(param, mesh, partition_spec)
            if verbose and xm.is_master_ordinal():
                xm.master_print(f'> [{strategy.upper()}] Sharding parameter {name} {param.shape} with spec {partition_spec}')
        
        # Partition buffers (for quantized weights, etc.)
        for name, buffer in model.named_buffers():
            # Skip 1D tensors and very small buffers
            if len(buffer.shape) == 1:
                continue
            
            partition_spec = AutoXLAModelForCausalLM._get_partition_spec(name, buffer.shape, strategy)
            if partition_spec is None:
                continue
            
            # Apply sharding
            xs.mark_sharding(buffer, mesh, partition_spec)
            if verbose and xm.is_master_ordinal():
                xm.master_print(f'> [{strategy.upper()}] Sharding buffer {name} {buffer.shape} with spec {partition_spec}')

    @staticmethod
    def _get_partition_spec(name, shape, strategy):
        """
        Determine partition spec based on tensor name, shape, and strategy.
        """
        if strategy.lower() == "fsdp":
            # FSDP: shard first dimension on fsdp axis
            if 'embed_tokens' in name or 'lm_head' in name:
                return ('fsdp', "mp")
            elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                return ('fsdp', "mp")
            elif 'o_proj' in name:
                return ("mp", 'fsdp')
            elif 'gate_proj' in name or 'up_proj' in name:
                return ("mp", 'fsdp')
            elif 'down_proj' in name:
                return ('fsdp', "mp")
            else:
                return ('fsdp', "mp")
                
        elif strategy.lower() in ["dp", "data_parallel"]:
            # Data parallel: replicate all parameters
            return tuple([None] * len(shape))
            
        elif strategy.lower() in ["mp", "model_parallel"]:
            # Model parallel: shard on model axis
            if 'embed_tokens' in name or 'lm_head' in name:
                return ('model', None)
            elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                return (None, 'model')
            elif 'o_proj' in name:
                return ('model', None)
            elif 'gate_proj' in name or 'up_proj' in name:
                return ('model', None)
            elif 'down_proj' in name:
                return (None, 'model')
            else:
                return (None, 'model')
                
        elif strategy.lower() == "2d":
            # 2D sharding: use both fsdp and model axes
            if 'embed_tokens' in name:
                return ('model', 'fsdp')
            elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                return ('fsdp', 'model')
            elif 'o_proj' in name:
                return ('model', 'fsdp')
            elif 'gate_proj' in name or 'up_proj' in name:
                return ('model', 'fsdp')
            elif 'down_proj' in name:
                return ('fsdp', 'model')
            elif 'lm_head' in name:
                return ('model', 'fsdp')
            else:
                return ('fsdp', 'model')
                
        elif strategy.lower() == "3d":
            # 3D sharding: use data, fsdp, and model axes
            # Typically replicate on data axis, shard on fsdp and model
            if 'embed_tokens' in name:
                return (None, 'model', 'fsdp')
            elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                return (None, 'fsdp', 'model')
            elif 'o_proj' in name:
                return (None, 'model', 'fsdp')
            elif 'gate_proj' in name or 'up_proj' in name:
                return (None, 'model', 'fsdp')
            elif 'down_proj' in name:
                return (None, 'fsdp', 'model')
            elif 'lm_head' in name:
                return (None, 'model', 'fsdp')
            else:
                return (None, 'fsdp', 'model')
        else:
            return None

    @staticmethod
    def _shard_output(output, mesh, strategy="fsdp"):
        """
        Custom output sharding function for FSDPv2.
        Required when model output is not a single tensor or tuple with activation at index 0.
        """
        if hasattr(output, 'logits'):
            if strategy.lower() in ["fsdp", "dp", "data_parallel"]:
                xs.mark_sharding(output.logits, mesh, ('fsdp', None, None))
            elif strategy.lower() in ["mp", "model_parallel"]:
                xs.mark_sharding(output.logits, mesh, (None, None, 'model'))
            elif strategy.lower() == "2d":
                xs.mark_sharding(output.logits, mesh, ('fsdp', None, 'model'))
            elif strategy.lower() == "3d":
                xs.mark_sharding(output.logits, mesh, ('data', None, 'model'))
        return output

    @staticmethod
    def _apply_attention_kernel(
        model,
        attn_implementation: str,
        mesh: Optional[xs.Mesh] = None,
        partition_spec: Optional[tuple] = None,
        attention_config: Optional[dict] = None,
        layer_indices: Optional[list] = None,
        verbose: bool = True
    ):
        """
        Apply custom attention kernel to the model based on attn_implementation.
        
        Args:
            model: The loaded model
            attn_implementation: Type of attention - "xla_flash_attention", "splash_attention", or "eager"
            mesh: XLA mesh (required for xla_flash_attention)
            partition_spec: Partition specification (required for xla_flash_attention)
            attention_config: Configuration dict for attention kernel (for splash_attention)
            layer_indices: Specific layers to patch (None = all layers)
            verbose: Print patching information
        
        Returns:
            Model with patched attention
        """
        if attn_implementation is None or attn_implementation.lower() == "eager":
            if verbose and xm.is_master_ordinal():
                xm.master_print("> Using eager (standard) attention implementation")
            return model
        
        attn_impl = attn_implementation.lower()
        
        if attn_impl == "xla_flash_attention":
            if verbose and xm.is_master_ordinal():
                xm.master_print("> Applying XLA Flash Attention")
            
            if mesh is None:
                raise ValueError(
                    "mesh is required for xla_flash_attention. "
                    "Please provide a mesh or let auto_shard create one."
                )
            
            if partition_spec is None:
                if verbose and xm.is_master_ordinal():
                    xm.master_print("> No partition_spec provided, using default for XLA Flash Attention")
                partition_spec = ('fsdp', None)  # Default partition spec
            
            patcher = XLAFlashAttentionPatcher(
                mesh=mesh,
                partition_spec=partition_spec,
                layer_indices=layer_indices
            )
            model = patcher.patch_model(model, inplace=True)
            
        elif attn_impl == "splash_attention":
            if verbose and xm.is_master_ordinal():
                xm.master_print("> Applying Splash Attention")
            
            # Import here to avoid circular imports
            from ..kernels.attention.splash_attention import SplashAttentionConfig
            
            # Create splash config from dict or use defaults
            if attention_config is None:
                if verbose and xm.is_master_ordinal():
                    xm.master_print("> No attention_config provided, using default Splash Attention settings")
                splash_config = SplashAttentionConfig(mesh=str(mesh), qkv_partition_spec=partition_spec, segment_ids_partition_spec=partition_spec)
            else:
                splash_config = SplashAttentionConfig(**attention_config)
            
            patcher = SplashAttentionPatcher(
                splash_config=splash_config,
                layer_indices=layer_indices
            )
            model = patcher.patch_model(model, inplace=True)
            
        else:
            raise ValueError(
                f"Unknown attention implementation: {attn_implementation}. "
                f"Choose from: 'eager', 'xla_flash_attention', 'splash_attention'"
            )
        
        if verbose and xm.is_master_ordinal():
            xm.master_print(f"> Successfully applied {attn_implementation}")
        
        return model

    @classmethod
    def from_pretrained(
        cls: type["AutoXLAModelForCausalLM"],
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        auto_shard: Optional[bool] = True,
        sharding_strategy: Optional[str] = "fsdp",
        gradient_checkpointing: Optional[bool] = True,
        use_fsdp_wrap: Optional[bool] = True,
        verbose: Optional[bool] = True,
        mesh: Optional[xs.Mesh] = None,
        auto_wrap_policy: Optional[Callable] = None,
        shard_output_callable: Optional[Callable] = None,
        do_quant: Optional[bool] = False,
        quantization_config: Optional[QuantizationConfig] = None,
        attn_implementation: Optional[str] = "eager",
        attention_config: Optional[dict] = None,
        attention_layer_indices: Optional[list] = None,
        attention_partition_spec: Optional[tuple] = None,
        *model_args,
        **kwargs
    ):
        """
        Load a pretrained model with XLA sharding support and optimizations.
        
        Args:
            pretrained_model_name_or_path: Model name or path
            auto_shard: Whether to automatically partition model parameters. 
                    If False, you must provide a mesh and handle sharding manually.
            sharding_strategy: Strategy for sharding - "fsdp", "dp", "mp", "2d", "3d"
                            Only used when auto_shard=True
            use_fsdp_wrap: Whether to wrap model with FSDPv2 after manual sharding
            verbose: Whether to print sharding information
            mesh: Custom mesh. Required if auto_shard=False.
                If provided with auto_shard=True, will use this mesh instead of creating one.
            auto_wrap_policy: Policy for auto-wrapping layers with FSDPv2
            shard_output_callable: Custom function for sharding model output
            do_quant: Whether to apply quantization
            quantization_config: Configuration for quantization
            attn_implementation: Attention implementation - "eager", "xla_flash_attention", "splash_attention"
            attention_config: Configuration dict for attention kernel (used with splash_attention)
                            Example: {"sa_block_q": 1024, ...}
            attention_layer_indices: Specific layers to patch with custom attention (None = all layers)
            attention_partition_spec: Partition spec for attention (used with xla_flash_attention)
            *model_args: Additional arguments for model loading
            **kwargs: Additional keyword arguments for model loading
        
        Returns:
            Sharded model ready for training with custom attention if specified
        
        Example:
            >>> # Load with XLA Flash Attention
            >>> model = AutoXLAModelForCausalLM.from_pretrained(
            ...     "meta-llama/Llama-2-7b-hf",
            ...     attn_implementation="xla_flash_attention",
            ...     sharding_strategy="fsdp"
            ... )
            
            >>> # Load with Splash Attention
            >>> model = AutoXLAModelForCausalLM.from_pretrained(
            ...     "meta-llama/Llama-2-7b-hf",
            ...     attn_implementation="splash_attention",
            ...     attention_config={"sa_block_q": 1024, ...}
            ... )
            
            >>> # Load with custom attention on specific layers
            >>> model = AutoXLAModelForCausalLM.from_pretrained(
            ...     "meta-llama/Llama-2-7b-hf",
            ...     attn_implementation="splash_attention",
            ...     attention_layer_indices=[0, 5, 10, 15, 20]
            ... )
        """
        xr.use_spmd()

        USE_FSDP_V2 = use_fsdp_wrap
        
        # Validation
        if not auto_shard and mesh is None:
            raise ValueError(
                "You set auto_shard to False, but have not provided a mesh. "
                "Please provide a mesh or set auto_shard=True."
            )
        
        if do_quant and quantization_config is None:
            raise ValueError(
                "You set do_quant to true, but did not provide a quantization configuration. "
                "Please pass a QuantizationConfig to quantization_config."
            )
        
        # Load the base model
        if verbose and xm.is_master_ordinal():
            xm.master_print(f"> Loading model from {pretrained_model_name_or_path}")
        if attention_partition_spec is None:
            if verbose:
                xm.master_print("> No partition spec for the attention was supplied. Setting it automatically.")
            attention_partition_spec = ("fsdp", "mp")
        # Always load with eager attention first, we'll patch it later
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            attn_implementation="eager", 
            *model_args, 
            **kwargs
        )  
        model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)
        model._set_gradient_checkpointing(enable=gradient_checkpointing, gradient_checkpointing_func=checkpoint)
        # Apply quantization if requested
        if do_quant:
            if verbose and xm.is_master_ordinal():
                xm.master_print("> Applying quantization")
            model = LanguageModelQuantizer(config=quantization_config).quantize_model(model)
        
        # Get XLA device
        device = xm.xla_device()
        
        # Move model to device (always needed)
        model.to(device)
        
        if auto_shard:
            # Automatic sharding path
            
            # Create or use provided mesh
            if mesh is None:
                devices = xr.global_runtime_device_count()
                mesh = cls._create_mesh(devices, strategy=sharding_strategy)
                if verbose and xm.is_master_ordinal():
                    xm.master_print(f"> Created mesh automatically with {devices} devices")
            else:
                if verbose and xm.is_master_ordinal():
                    xm.master_print(f"> Using provided mesh")
            
            # Apply custom attention BEFORE sharding
            # This is important because attention wrappers need to be in place before FSDP wrapping
            if attn_implementation and attn_implementation.lower() != "eager":
                if verbose and xm.is_master_ordinal():
                    xm.master_print(f"> Applying custom attention before sharding")
                
                model = cls._apply_attention_kernel(
                    model=model,
                    attn_implementation=attn_implementation,
                    mesh=mesh,
                    partition_spec=attention_partition_spec,
                    attention_config=attention_config,
                    layer_indices=attention_layer_indices,
                    verbose=verbose
                )
            
            # Apply manual parameter sharding
            if verbose and xm.is_master_ordinal():
                xm.master_print(f"> Applying manual parameter sharding with strategy: {sharding_strategy}")
            
            cls._partition_model(model, device, mesh, strategy=sharding_strategy, verbose=verbose)
            
            # Wrap with FSDPv2 to handle any remaining parameters and enable FSDP features
            if use_fsdp_wrap:
                if verbose and xm.is_master_ordinal():
                    xm.master_print("> Wrapping model with FSDPv2")
                
                # Create shard_output function if not provided
                if shard_output_callable is None:
                    shard_output_callable = partial(cls._shard_output, strategy=sharding_strategy)
                
                # Wrap the model with FSDPv2
                model = FSDPv2(
                    model,
                    mesh=mesh,
                    auto_wrap_policy=auto_wrap_policy,
                    shard_output=shard_output_callable
                )
                
                if verbose and xm.is_master_ordinal():
                    xm.master_print("> Model successfully wrapped with FSDPv2")
        
        else:
            # Manual sharding path - user provides mesh and handles sharding themselves
            if verbose and xm.is_master_ordinal():
                xm.master_print("> auto_shard=False: Using mesh provided")
            
            # Apply custom attention BEFORE sharding
            if attn_implementation and attn_implementation.lower() != "eager":
                if verbose and xm.is_master_ordinal():
                    xm.master_print(f"> Applying custom attention before sharding")
                
                model = cls._apply_attention_kernel(
                    model=model,
                    attn_implementation=attn_implementation,
                    mesh=mesh,
                    partition_spec=attention_partition_spec,
                    attention_config=attention_config,
                    layer_indices=attention_layer_indices,
                    verbose=verbose
                )
            
            cls._partition_model(model, device, mesh, strategy=sharding_strategy, verbose=verbose)
            
            # Still wrap with FSDPv2 if requested (will use auto-wrap policy if provided)
            if use_fsdp_wrap:
                if mesh is None:
                    raise ValueError(
                        "mesh is required when use_fsdp_wrap=True, even with auto_shard=False"
                    )
                
                if verbose and xm.is_master_ordinal():
                    xm.master_print("> Wrapping model with FSDPv2 (no manual sharding)")
                
                # Create shard_output function if not provided
                if shard_output_callable is None:
                    if verbose and xm.is_master_ordinal():
                        xm.master_print("> Warning: No shard_output_callable provided. Using default FSDP sharding.")
                    # Default to FSDP-style output sharding
                    shard_output_callable = partial(cls._shard_output, mesh=mesh, strategy="fsdp")
                
                model = FSDPv2(
                    model,
                    mesh=mesh,
                    auto_wrap_policy=auto_wrap_policy,
                    shard_output=shard_output_callable
                )
                
                if verbose and xm.is_master_ordinal():
                    xm.master_print("> Model successfully wrapped with FSDPv2")
        
        if verbose and xm.is_master_ordinal():
            xm.master_print("> Model loading and sharding complete")
        
        return model