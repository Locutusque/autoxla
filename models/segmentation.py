"""
AutoXLA support for image segmentation models on TPU.

Two entry points are provided by :class:`AutoXLAModelForImageSegmentation`:

* ``from_pretrained`` — loads a Hugging Face segmentation checkpoint through
  the matching auto class (``AutoModelForMaskGeneration`` for SAM/SAM2/SAM3,
  ``AutoModelForUniversalSegmentation`` for Mask2Former/OneFormer,
  ``AutoModelForSemanticSegmentation`` for SegFormer/UperNet, ...) and then
  quantizes/shards it for TPU execution.

* ``from_model`` — applies the same quantization/sharding pipeline to an
  already-instantiated ``nn.Module``. This is the integration point for
  projects like MedSAM3 that build SAM3 natively (e.g. via
  ``sam3.model_builder.build_sam3_image_model``) instead of going through
  ``transformers``::

      from sam3.model_builder import build_sam3_image_model
      from AutoXLA import AutoXLAModelForImageSegmentation

      model = build_sam3_image_model(load_from_HF=True, eval_mode=False)
      model = AutoXLAModelForImageSegmentation.from_model(
          model,
          sharding_strategy="fsdp",
          do_quant=True,
          quantization_config=QuantizationConfig(n_bits=8, use_pallas=True),
      )
"""

import os
from typing import Optional, Union, Callable, List
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoConfig

from ..quantization import QuantizationConfig, ModelQuantizer

import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
from torch_xla.utils.checkpoint import checkpoint

from .auto_model import AutoXLAModelForCausalLM


# Hugging Face auto classes tried in order when resolving a segmentation
# checkpoint. Names are resolved lazily so older transformers versions that
# lack some of them still work.
_SEGMENTATION_AUTO_CLASS_NAMES = (
    "AutoModelForMaskGeneration",         # SAM, SAM-HQ, SAM2, SAM3
    "AutoModelForUniversalSegmentation",  # Mask2Former, OneFormer, MaskFormer
    "AutoModelForInstanceSegmentation",
    "AutoModelForSemanticSegmentation",   # SegFormer, UperNet, DPT, ...
    "AutoModelForImageSegmentation",
)

# Substrings identifying column-parallel linears (shard the output dim).
# Covers HF SAM (`q_proj`/`k_proj`/`v_proj`, `lin1`), timm/native ViT and
# SAM2/SAM3 vision encoders (fused `qkv`, `fc1`), CLIP-style text encoders,
# and DETR-style decoders (`linear1`).
_COLUMN_PARALLEL_PATTERNS = (
    "q_proj", "k_proj", "v_proj", "qkv", "in_proj",
    "fc1", "lin1", "linear1", "gate_proj", "up_proj", "w1", "w3",
)

# Substrings identifying row-parallel linears (shard the input dim).
_ROW_PARALLEL_PATTERNS = (
    "o_proj", "out_proj", "proj_out",
    "fc2", "lin2", "linear2", "down_proj", "w2", "wo",
)

# Output attributes that carry the main batch-major activation for the
# common segmentation output types (Sam*ImageSegmentationOutput,
# Mask2Former/MaskFormer outputs, SemanticSegmenterOutput, ...).
_OUTPUT_TENSOR_ATTRS = (
    "pred_masks",
    "masks_queries_logits",
    "class_queries_logits",
    "logits",
    "iou_scores",
    "low_res_masks",
    "masks",
    "last_hidden_state",
)


class AutoXLAModelForImageSegmentation(object):
    """TPU-optimized loader/wrapper for image segmentation models.

    Mirrors :class:`AutoXLAModelForCausalLM` (automatic mesh creation,
    parameter sharding, optional quantization and FSDPv2 wrapping) with
    vision-aware partition specs and segmentation-aware output sharding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Reuse the mesh factory from the causal-LM class so both entry points
    # produce identical meshes for a given strategy.
    _create_mesh = AutoXLAModelForCausalLM._create_mesh

    @staticmethod
    def _resolve_auto_class(config):
        """Pick the transformers auto class that maps this config, in priority
        order (mask generation first, so SAM/SAM2/SAM3 resolve correctly)."""
        import transformers

        for auto_name in _SEGMENTATION_AUTO_CLASS_NAMES:
            auto_cls = getattr(transformers, auto_name, None)
            if auto_cls is None:
                continue
            mapping = getattr(auto_cls, "_model_mapping", None)
            if mapping is None:
                continue
            try:
                mapping[type(config)]
            except KeyError:
                continue
            return auto_cls

        # Last resort: the bare AutoModel (covers backbones and remote code).
        return transformers.AutoModel

    @staticmethod
    def _mesh_axis_sizes(mesh) -> dict:
        """Map mesh axis name -> number of devices along that axis."""
        try:
            return dict(zip(mesh.axis_names, mesh.mesh_shape))
        except (AttributeError, TypeError):
            return {}

    @staticmethod
    def _spec_fits(spec, shape, axis_sizes) -> bool:
        """True when every sharded dimension is divisible by its axis size.

        Vision models are full of oddly-sized tensors (relative position
        embeddings, small decoder convs); replicating those is always correct,
        while sharding a non-divisible dim raises in torch_xla.
        """
        for dim, axis in zip(shape, spec):
            if axis is None:
                continue
            size = axis_sizes.get(axis, 1)
            if size and dim % size != 0:
                return False
        return True

    @staticmethod
    def _get_partition_spec(name, shape, strategy):
        """Partition spec for vision/segmentation model tensors.

        Weight layout conventions follow nn.Linear ([out, in]) and nn.Conv2d
        ([out, in, kh, kw]). Column-parallel layers (QKV projections, first
        MLP linears) shard the output dim; row-parallel layers (attention
        output projections, second MLP linears) shard the input dim.
        """
        strategy = strategy.lower()

        if strategy in ("dp", "data_parallel"):
            # Data parallel: replicate all parameters
            return tuple([None] * len(shape))

        # (column spec, row spec, default spec) for 2D weights, plus the axis
        # used for the output-channel dim of conv kernels.
        if strategy == "fsdp":
            col, row, default, conv_axis = ('fsdp', 'model'), ('model', 'fsdp'), ('fsdp', 'model'), 'fsdp'
        elif strategy in ("mp", "model_parallel"):
            col, row, default, conv_axis = ('model', None), (None, 'model'), ('model', None), 'model'
        elif strategy == "2d":
            col, row, default, conv_axis = ('model', 'fsdp'), ('fsdp', 'model'), ('fsdp', 'model'), 'fsdp'
        elif strategy == "3d":
            # Parameters are replicated on the data axis; specs must match the
            # tensor rank, so 2D weights get 2-element specs.
            col, row, default, conv_axis = ('model', 'fsdp'), ('fsdp', 'model'), ('fsdp', 'model'), 'fsdp'
        else:
            return None

        if len(shape) == 4:
            # Conv kernels (patch embeddings, mask decoder upscaling):
            # shard output channels only.
            return (conv_axis, None, None, None)

        if len(shape) != 2:
            # 3D/5D+ tensors (relative position embeddings, learned queries
            # with leading batch dims, ...): replicate.
            return None

        lname = name.lower()

        for pattern in _COLUMN_PARALLEL_PATTERNS:
            if pattern in lname:
                return col
        for pattern in _ROW_PARALLEL_PATTERNS:
            if pattern in lname:
                return row
        # ViT/SAM attention output projections are usually called just
        # `<attn>.proj`; patch embedding convs were already handled above.
        if lname.endswith(".proj") or lname.endswith(".proj.weight"):
            return row

        return default

    @classmethod
    def _partition_model(cls, model, device, mesh, strategy="fsdp", verbose=True):
        """Shard parameters and buffers with vision-aware partition specs."""
        model.to(device)
        axis_sizes = cls._mesh_axis_sizes(mesh)

        def _shard(name, tensor, kind):
            if tensor.ndim <= 1:
                return
            spec = cls._get_partition_spec(name, tensor.shape, strategy)
            if spec is None:
                return
            if not cls._spec_fits(spec, tensor.shape, axis_sizes):
                if verbose and xm.is_master_ordinal():
                    xm.master_print(
                        f'> [{strategy.upper()}] Replicating {kind} {name} '
                        f'{tuple(tensor.shape)} (not divisible by mesh axes)')
                return
            xs.mark_sharding(tensor, mesh, spec)
            if verbose and xm.is_master_ordinal():
                xm.master_print(
                    f'> [{strategy.upper()}] Sharding {kind} {name} '
                    f'{tuple(tensor.shape)} with spec {spec}')

        for name, param in model.named_parameters():
            _shard(name, param, "parameter")
        for name, buffer in model.named_buffers():
            _shard(name, buffer, "buffer")

    @classmethod
    def _shard_output(cls, output, mesh, strategy="fsdp"):
        """Output sharding for FSDPv2 that understands segmentation outputs.

        Handles plain tensors, tuples, and HF ModelOutput objects carrying
        `pred_masks` / `masks_queries_logits` / `logits` etc. — the default
        FSDPv2 shard_output only understands tensors and `.logits`.
        """
        batch_axis = {
            "fsdp": "fsdp",
            "dp": "data",
            "data_parallel": "data",
            "2d": "fsdp",
            "3d": "data",
        }.get(strategy.lower())
        if batch_axis is None:
            # Pure model parallelism: activations are replicated on batch.
            return output

        axis_sizes = cls._mesh_axis_sizes(mesh)
        axis_size = axis_sizes.get(batch_axis, 1)

        def _mark(t):
            if not torch.is_tensor(t) or t.dim() == 0:
                return False
            if axis_size and t.shape[0] % axis_size != 0:
                return False
            xs.mark_sharding(t, mesh, (batch_axis,) + (None,) * (t.dim() - 1))
            return True

        marked = False
        for attr in _OUTPUT_TENSOR_ATTRS:
            t = getattr(output, attr, None)
            if torch.is_tensor(t):
                marked = _mark(t) or marked

        if not marked:
            if torch.is_tensor(output):
                _mark(output)
            elif isinstance(output, (tuple, list)):
                for t in output:
                    if torch.is_tensor(t) and _mark(t):
                        break

        return output

    @classmethod
    def _prepare(
        cls,
        model: nn.Module,
        auto_shard: bool = True,
        sharding_strategy: str = "fsdp",
        use_fsdp_wrap: bool = True,
        verbose: bool = True,
        mesh=None,
        auto_wrap_policy: Optional[Callable] = None,
        shard_output_callable: Optional[Callable] = None,
        do_quant: bool = False,
        quantization_config: Optional[QuantizationConfig] = None,
        xla_patch_linear: bool = True,
    ) -> nn.Module:
        """Shared quantize -> shard -> wrap pipeline."""
        xr.use_spmd()

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

        if xla_patch_linear:
            model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

        if do_quant:
            if verbose and xm.is_master_ordinal():
                xm.master_print("> Applying quantization")
            model = ModelQuantizer(config=quantization_config).quantize_model(model)

        device = xm.xla_device()
        model.to(device)

        if mesh is None:
            devices = xr.global_runtime_device_count()
            mesh = cls._create_mesh(devices, strategy=sharding_strategy)
            if verbose and xm.is_master_ordinal():
                xm.master_print(f"> Created mesh automatically with {devices} devices")
        elif verbose and xm.is_master_ordinal():
            xm.master_print("> Using provided mesh")

        if auto_shard:
            if verbose and xm.is_master_ordinal():
                xm.master_print(f"> Applying manual parameter sharding with strategy: {sharding_strategy}")
            cls._partition_model(model, device, mesh, strategy=sharding_strategy, verbose=verbose)

        if use_fsdp_wrap:
            if verbose and xm.is_master_ordinal():
                xm.master_print("> Wrapping model with FSDPv2")

            if shard_output_callable is None:
                shard_output_callable = partial(cls._shard_output, strategy=sharding_strategy)

            model = FSDPv2(
                model,
                mesh=mesh,
                auto_wrap_policy=auto_wrap_policy,
                shard_output=shard_output_callable,
            )

            if verbose and xm.is_master_ordinal():
                xm.master_print("> Model successfully wrapped with FSDPv2")

        if verbose and xm.is_master_ordinal():
            xm.master_print("> Model preparation and sharding complete")

        return model

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        auto_shard: bool = True,
        sharding_strategy: str = "fsdp",
        use_fsdp_wrap: bool = True,
        verbose: bool = True,
        mesh=None,
        auto_wrap_policy: Optional[Callable] = None,
        shard_output_callable: Optional[Callable] = None,
        do_quant: bool = False,
        quantization_config: Optional[QuantizationConfig] = None,
        xla_patch_linear: bool = True,
    ) -> nn.Module:
        """
        Prepare an already-instantiated segmentation model for TPU execution.

        Use this when the model is not loaded through transformers — e.g.
        MedSAM3's native SAM3 built with `sam3.model_builder.build_sam3_image_model`
        — and you still want AutoXLA's quantization, sharding, and FSDPv2
        wrapping.

        Args:
            model: Any torch nn.Module (SAM3, SAM2, MedSAM variants, ...)
            auto_shard: Whether to automatically partition model parameters
            sharding_strategy: "fsdp", "dp", "mp", "2d", or "3d"
            use_fsdp_wrap: Whether to wrap with FSDPv2 after sharding
            verbose: Print sharding information
            mesh: Custom mesh; created automatically when None
            auto_wrap_policy: Policy for auto-wrapping layers with FSDPv2
            shard_output_callable: Custom function for sharding model output
            do_quant: Whether to apply quantization
            quantization_config: Configuration for quantization
            xla_patch_linear: Apply torch_xla's einsum-based nn.Linear patch
                (improves SPMD sharding propagation). Quantized layers are
                unaffected since they replace nn.Linear entirely.

        Returns:
            The prepared (optionally quantized, sharded, FSDPv2-wrapped) model.
        """
        return cls._prepare(
            model,
            auto_shard=auto_shard,
            sharding_strategy=sharding_strategy,
            use_fsdp_wrap=use_fsdp_wrap,
            verbose=verbose,
            mesh=mesh,
            auto_wrap_policy=auto_wrap_policy,
            shard_output_callable=shard_output_callable,
            do_quant=do_quant,
            quantization_config=quantization_config,
            xla_patch_linear=xla_patch_linear,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        auto_shard: Optional[bool] = True,
        sharding_strategy: Optional[str] = "fsdp",
        gradient_checkpointing: Optional[bool] = False,
        use_fsdp_wrap: Optional[bool] = True,
        verbose: Optional[bool] = True,
        mesh=None,
        auto_wrap_policy: Optional[Callable] = None,
        shard_output_callable: Optional[Callable] = None,
        do_quant: Optional[bool] = False,
        quantization_config: Optional[QuantizationConfig] = None,
        attn_implementation: Optional[str] = "eager",
        xla_patch_linear: Optional[bool] = True,
        *model_args,
        **kwargs,
    ) -> nn.Module:
        """
        Load a pretrained segmentation model with XLA sharding support.

        The checkpoint is resolved against the segmentation auto classes in
        priority order: mask generation (SAM/SAM2/SAM3), universal
        segmentation (Mask2Former/OneFormer), instance segmentation, semantic
        segmentation, then plain AutoModel as a fallback.

        Args:
            pretrained_model_name_or_path: Model name or path
                (e.g. "facebook/sam3", "facebook/sam2-hiera-large",
                "facebook/sam-vit-huge", "nvidia/segformer-b0-finetuned-ade-512-512")
            auto_shard: Whether to automatically partition model parameters
            sharding_strategy: "fsdp", "dp", "mp", "2d", or "3d"
            gradient_checkpointing: Enable gradient checkpointing when the
                model supports it
            use_fsdp_wrap: Whether to wrap with FSDPv2 after sharding
            verbose: Print sharding information
            mesh: Custom mesh; created automatically when None
            auto_wrap_policy: Policy for auto-wrapping layers with FSDPv2
            shard_output_callable: Custom function for sharding model output
            do_quant: Whether to apply quantization
            quantization_config: Configuration for quantization
            attn_implementation: Only "eager" is supported for segmentation
                models — the splash/flash attention patchers implement causal
                decoder attention, which is incorrect for the bidirectional
                attention used by vision encoders.
            xla_patch_linear: Apply torch_xla's einsum-based nn.Linear patch
            *model_args, **kwargs: Forwarded to the underlying from_pretrained

        Returns:
            The prepared (optionally quantized, sharded, FSDPv2-wrapped) model.

        Example:
            >>> from AutoXLA import AutoXLAModelForImageSegmentation
            >>> from AutoXLA.quantization import QuantizationConfig
            >>>
            >>> model = AutoXLAModelForImageSegmentation.from_pretrained(
            ...     "facebook/sam3",
            ...     sharding_strategy="fsdp",
            ...     do_quant=True,
            ...     quantization_config=QuantizationConfig(n_bits=8, use_pallas=True),
            ... )
        """
        if attn_implementation and attn_implementation.lower() != "eager":
            if xm.is_master_ordinal():
                xm.master_print(
                    f"> attn_implementation='{attn_implementation}' is not supported for "
                    "image segmentation models (the custom kernels implement causal "
                    "decoder attention). Falling back to eager attention.")

        if verbose and xm.is_master_ordinal():
            xm.master_print(f"> Loading segmentation model from {pretrained_model_name_or_path}")

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        auto_cls = cls._resolve_auto_class(config)

        if verbose and xm.is_master_ordinal():
            xm.master_print(f"> Resolved auto class: {auto_cls.__name__}")

        model = auto_cls.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        if gradient_checkpointing:
            try:
                model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=checkpoint)
            except (AttributeError, TypeError):
                try:
                    model.gradient_checkpointing_enable()
                except (AttributeError, ValueError):
                    if verbose and xm.is_master_ordinal():
                        xm.master_print("> Model does not support gradient checkpointing; skipping")

        return cls._prepare(
            model,
            auto_shard=auto_shard,
            sharding_strategy=sharding_strategy,
            use_fsdp_wrap=use_fsdp_wrap,
            verbose=verbose,
            mesh=mesh,
            auto_wrap_policy=auto_wrap_policy,
            shard_output_callable=shard_output_callable,
            do_quant=do_quant,
            quantization_config=quantization_config,
            xla_patch_linear=xla_patch_linear,
        )


# SAM-family checkpoints resolve through AutoModelForMaskGeneration; expose a
# matching alias for discoverability.
AutoXLAModelForMaskGeneration = AutoXLAModelForImageSegmentation
