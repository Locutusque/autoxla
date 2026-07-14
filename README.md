
---

# AutoXLA

**AutoXLA** is a library designed to automate the **distribution, optimization, and quantization** of large language models on **TPUs** using PyTorch/XLA.
It extends the Hugging Face Transformers interface with TPU-aware features like **automatic sharding**, **custom attention kernels**, and **quantization support** — enabling efficient large-scale model deployment and training.

> **Note: This is an experimental repository and may have a lot of issues. Please open issues if you find errors in the code.**
---

## Features

* **Automatic Sharding**
  Supports multiple parallelization strategies:

  * `fsdp` – Fully Sharded Data Parallel
  * `dp` / `data_parallel` – Data Parallel
  * `mp` / `model_parallel` – Model Parallel
  * `2d` – Hybrid Data-Model Sharding
  * `3d` – 3-Axis Parallelism (Data + FSDP + Model)

* **Attention Kernel Patching**
  Swap standard attention with TPU-optimized implementations:

  * `xla_flash_attention` – Flash Attention V2
  * `splash_attention` – Block-structured sparse attention
  * `eager` – Standard PyTorch attention
  * More to come in future versions

* **Quantization Support**
  Integrates with `ModelQuantizer` (alias `LanguageModelQuantizer`) via `QuantizationConfig` for parameter quantization before distribution. Three quantized-matmul backends are wired, in order of preference:

  * A Pallas TPU kernel (per-channel symmetric int8/int4, optional dynamic activation quantization)
  * torch_xla's fused `torch.ops.xla.quantized_matmul` kernel
  * A pure-torch dequantize+matmul fallback (also covers blockwise and asymmetric quantization)

* **Image Segmentation Models**
  `AutoXLAModelForImageSegmentation` loads Hugging Face segmentation checkpoints (SAM/SAM2/SAM3, Mask2Former, SegFormer, ...) with vision-aware sharding, and `from_model` applies the same quantize/shard/wrap pipeline to natively built models (e.g. [MedSAM3](https://github.com/Locutusque/MedSAM3)'s SAM3).

* **Flexible FSDP Wrapping**
  Integrates with `torch_xla.experimental.SpmdFullyShardedDataParallel` (`FSDPv2`) for efficient parameter and activation partitioning.

---

## Installation

AutoXLA depends on **PyTorch/XLA** and **Transformers**.
Ensure a TPU runtime is available before installation.

### Install from Source (Recommended)

Installing from source ensures all TPU-specific dependencies, including PyTorch/XLA with the correct build links, are properly configured:
```bash
git clone https://github.com/Locutusque/autoxla.git
cd AutoXLA
pip install -r requirements.txt
pip install -e .
```

### Install from PyPI

If you prefer to install from PyPI, you'll need to manually install PyTorch/XLA first:
```bash
pip install torch~=2.8.0
pip install torch_xla[tpu]~=2.8.0 --find-links=https://storage.googleapis.com/libtpu-releases/index.html --find-links=https://storage.googleapis.com/libtpu-wheels/index.html
pip install torch_xla[pallas] --find-links=https://storage.googleapis.com/jax-releases/jax_nightly_releases.html --find-links=https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip install autoxla
```

---

## Quick Start

Below is a minimal example using `AutoXLAModelForCausalLM` to load and shard a Hugging Face model across TPU devices.

```python
from AutoXLA.modeling import AutoXLAModelForCausalLM

# Create block sizes
sa_blocks = {
    "sa_block_q": 512,
    "sa_block_kv": 512,
    "sa_block_kv_compute": 512,
    "sa_block_q_dkv": 512,
    "sa_block_kv_dkv": 512,
    "sa_block_kv_dkv_compute": 512,
    "sa_block_q_dq": 512,
    "sa_block_kv_dq": 512,
    "mesh": str(mesh)
}

# Load a pretrained model with FSDP sharding and XLA Flash Attention
model = AutoXLAModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="splash_attention",
    sharding_strategy="fsdp"
)

# The model is now sharded and ready for TPU-based training or inference
```

### Example Variants

**Splash Attention:**

```python
model = AutoXLAModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="splash_attention",
    attention_config=sa_blocks,
)
```

**3D Sharding (Data + FSDP + Model Parallelism):**

```python
model = AutoXLAModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    sharding_strategy="3d"
)
```

**Quantized Loading:**

```python
from AutoXLA.quantization import QuantizationConfig

quant_cfg = QuantizationConfig(bits=8, use_pallas=True, quantize_activation=True)

model = AutoXLAModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    do_quant=True,
    quantization_config=quant_cfg
)
```

---

## Image Segmentation Models

`AutoXLAModelForImageSegmentation` brings the same automatic sharding and quantization to segmentation models. Checkpoints are resolved against the segmentation auto classes in priority order: mask generation (SAM/SAM2/SAM3), universal segmentation (Mask2Former/OneFormer), instance segmentation, and semantic segmentation (SegFormer/UperNet).

```python
from AutoXLA import AutoXLAModelForImageSegmentation
from AutoXLA.quantization import QuantizationConfig

model = AutoXLAModelForImageSegmentation.from_pretrained(
    "facebook/sam3",
    sharding_strategy="fsdp",
    do_quant=True,
    quantization_config=QuantizationConfig(n_bits=8, use_pallas=True),
)
```

**Wrapping a natively built model (e.g. MedSAM3):**

Projects such as [MedSAM3](https://github.com/Locutusque/MedSAM3) build SAM3 through the native `sam3` package instead of `transformers`. Use `from_model` to apply AutoXLA's quantize → shard → FSDPv2 pipeline to any already-instantiated `nn.Module`:

```python
from sam3.model_builder import build_sam3_image_model
from AutoXLA import AutoXLAModelForImageSegmentation
from AutoXLA.quantization import QuantizationConfig

model = build_sam3_image_model(load_from_HF=True, eval_mode=False)

model = AutoXLAModelForImageSegmentation.from_model(
    model,
    sharding_strategy="fsdp",
    do_quant=True,
    quantization_config=QuantizationConfig(
        n_bits=8,
        use_pallas=True,
        # keep accuracy-sensitive heads in full precision
        exclude_layers=["iou_prediction_head", "output_upscaling"],
    ),
)
```

Notes for segmentation models:

* Layer biases are preserved in full precision by quantization (SAM/ViT linears all carry biases).
* Partition specs understand vision naming conventions (fused `qkv`, `proj`/`out_proj`, `fc1`/`fc2`, `lin1`/`lin2`, DETR's `linear1`/`linear2`) and conv kernels; tensors whose dimensions don't divide the mesh (e.g. relative position embeddings) are replicated automatically.
* `attn_implementation` other than `"eager"` is ignored for segmentation models — the splash/flash kernels implement causal decoder attention, which is incorrect for bidirectional vision attention.

---

## API Reference

### `AutoXLAModelForCausalLM`

A TPU-optimized version of `AutoModelForCausalLM` with automatic model partitioning, quantization, and attention kernel patching.

#### **Class Methods**

##### `from_pretrained(pretrained_model_name_or_path, **kwargs)`

Load a pretrained model with XLA-specific optimizations.

**Key Arguments:**

* `pretrained_model_name_or_path` — Model identifier or path
* `auto_shard` (`bool`, default `True`) — Automatically create and apply sharding
* `sharding_strategy` (`str`) — One of `"fsdp"`, `"dp"`, `"mp"`, `"2d"`, `"3d"`
* `do_quant` (`bool`) — Enable quantization
* `quantization_config` — Instance of `QuantizationConfig`
* `attn_implementation` (`str`) — `"eager"`, `"xla_flash_attention"`, or `"splash_attention"`
* `attention_config` (`dict`) — Additional configuration for Splash Attention
* `use_fsdp_wrap` (`bool`) — Whether to wrap with `FSDPv2` after sharding

**Returns:**
A fully-loaded, sharded, and optionally quantized model ready for TPU execution.

### `AutoXLAModelForImageSegmentation`

A TPU-optimized loader/wrapper for image segmentation models (also exported as `AutoXLAModelForMaskGeneration`).

#### **Class Methods**

##### `from_pretrained(pretrained_model_name_or_path, **kwargs)`

Load a Hugging Face segmentation checkpoint (e.g. `facebook/sam3`, `facebook/sam2-hiera-large`, `facebook/sam-vit-huge`, `nvidia/segformer-b0-finetuned-ade-512-512`) with automatic sharding and optional quantization. Accepts the same sharding/quantization arguments as `AutoXLAModelForCausalLM.from_pretrained`.

##### `from_model(model, **kwargs)`

Apply the quantize → shard → FSDPv2 pipeline to an already-instantiated `nn.Module` — the integration point for natively built models such as MedSAM3's SAM3.

**Key Arguments (both methods):**

* `sharding_strategy` (`str`) — One of `"fsdp"`, `"dp"`, `"mp"`, `"2d"`, `"3d"`
* `do_quant` (`bool`) — Enable quantization
* `quantization_config` — Instance of `QuantizationConfig`
* `use_fsdp_wrap` (`bool`) — Whether to wrap with `FSDPv2` after sharding
* `xla_patch_linear` (`bool`) — Apply torch_xla's einsum-based `nn.Linear` patch

---

## Sharding Strategies

| Strategy | Description                          | Typical Use                    |
| -------- | ------------------------------------ | ------------------------------ |
| `fsdp`   | Shards parameters across all devices | Training large models          |
| `dp`     | Replicates model across devices      | Small-scale fine-tuning        |
| `mp`     | Splits layers across devices         | Model-parallel inference       |
| `2d`     | Combines FSDP + model parallel       | Balanced training/inference    |
| `3d`     | Adds data parallelism axis           | Large-scale distributed setups |

---

## Attention Implementations

| Implementation        | Description                                        |
| --------------------- | -------------------------------------------------- |
| `eager`               | Default PyTorch attention                          |
| `xla_flash_attention` | Optimized fused attention kernel using XLA         |
| `splash_attention`    | Sparse attention kernel configurable by block size |

---

## License

This repository is distributed under the apache-2.0 license.
Credits go to IsNoobGrammer, vLLM, and torchprime for some of the kernels.

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
---




