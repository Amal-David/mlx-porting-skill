# Quantization

## Quantization is a model change

Do not treat file-size reduction as proof of usable acceleration. MLX performance depends on supported kernels, matrix shapes, group size, bit width, packing, dequantization, and workload.

## Baseline sequence

1. Establish FP16/BF16/FP32 parity.
2. Identify memory- versus compute-bound stages.
3. Quantize only eligible linear/projection modules.
4. Keep embeddings, output heads, norms, small matrices, sensitive recurrent/state modules, and codec/vocoder layers in higher precision until tested.
5. Evaluate quality and speed per recipe.
6. Add KV/state quantization separately.

## Weight-only options

Use framework/library-supported affine or floating low-bit modes first. Test at least one conservative baseline (for example 8- or 6-bit where available) and common 4-bit settings before lower or mixed precision.

Current MLX quantization modes include `affine`, `mxfp4`, `mxfp8`, and `nvfp4`, with mode-specific group-size, bit-width, scale, and bias behavior. For `mlx.nn.quantize`, activation/input quantization through `quantize_input=True` is limited to supported Linear-layer paths such as `nvfp4` and `mxfp8`; do not generalize it to embeddings, norms, convolutions, or arbitrary modules.

MLX-Audio documents `affine`, `mxfp4`, `mxfp8`, and `nvfp4` conversion modes plus 3/4/6/8-bit recipes. Its size-reduction guidance is source-reported and should remain tied to the documented model family and recipe. TTS needs listening and speaker/prosody checks; STT needs WER/CER and timestamp checks.

For every recipe record:

- bits and format;
- group size;
- scale/bias layout;
- module predicate/exclusions;
- original dtype;
- calibration method if any;
- packing/runtime version;
- quality and performance metrics.

## Mixed-bit policies

Allocate precision by measured sensitivity, not parameter name folklore. Candidate signals include layerwise output error, Hessian/activation statistics, outlier magnitude, task-quality ablation, and module size. A mixed recipe must be deterministic and versioned.

## Rotation and activation quantization

Hadamard/learned rotations and activation quantization can reduce outliers but require additional transforms and compatible kernels. MLX exposing a Hadamard primitive does not by itself prove end-to-end benefit. Treat QuaRot/SpinQuant-style methods as experiments until implemented and benchmarked on the target.

## Audio-specific exclusions

Often-sensitive components include:

- codec codebooks and distance/search paths;
- final waveform projections;
- normalization and gain controls;
- transposed-convolution or Fourier decoder stages;
- speaker/style encoders;
- duration/pitch predictors;
- small conditioning projections;
- streaming state with accumulated error.

Quantize one subsystem at a time. Listen and measure boundary artifacts, clipping, intelligibility, speaker similarity, and RTF.

## KV/state quantization

Measure quality versus context/stream duration and include quantize/dequantize overhead. Keys and values may need different precision. Preserve recent/sink/global state where the method requires it.

Treat adaptive bit allocation, RoPE-aware key allocation, hardware-aware grouping, and FP4/FP8-style KV cache formats as research candidates until the target MLX runtime has compatible packing, attention, and dequantization paths. A paper result on CUDA, AMD, or another serving stack is not an MLX performance claim.

For MoE, quantize experts separately from routers, shared expert gates, norms, and small projections. `gather_qmm` is the native indexed quantized path to investigate after loop parity, but small routed groups can lose the theoretical memory-bandwidth benefit.

## Quantized publication gate

Publish only with the unquantized source revision, exact recipe, exclusions, converter/library version, representative quality results, benchmark metadata, and a clear statement that performance varies by Mac and workload.

For a local, non-promotable comparison of an already-supported `mlx_lm`
conversion against its bf16 reference, use
[`scripts/quant_quality_gate.py`](../scripts/quant_quality_gate.py) and follow
the measured boundary in
[`quantization-quality-gate.md`](quantization-quality-gate.md). Its output is a
user diagnostic only and must not be ingested into the sealed claims or receipt
pipeline.
