# Runbook: non-generative CV backbones

## Applies to

Image classification, feature extraction, and backbone-only ports such as ResNet,
ConvNeXt, EfficientNet, Swin-style classifiers, and similar non-generative
vision encoders. Route detection, segmentation, promptable masks, OCR, depth,
pose, and image generation to separate research tracks until they have their own
fixtures and task-specific gates.

## Architecture fingerprint

Confirm:

- processor resize, crop, interpolation, rescale, mean/std, channel order, and
  batch layout;
- source tensor layout: PyTorch-style NCHW, channels-last NHWC, or mixed
  boundaries;
- stem, residual/stage/block structure, downsampling, grouped/depthwise
  convolution, dilation, and padding conventions;
- normalization type, epsilon, affine behavior, activation, stochastic-depth or
  dropout policy, and train/eval mode;
- pooling, classifier or embedding head, label order, and feature tap used by
  downstream callers.

## Source oracle checkpoints

Capture decoded/resized pixels, normalized tensor, post-layout tensor, stem
output, one representative stage/block, pooled feature, classifier logits, and
top-k labels. For feature extractors, also capture the exact hidden-state or
embedding tensor the product consumes.

## Weight conversion

- preserve OIHW convolution order unless converting weights once at load time;
- document every transpose between NCHW, NHWC, and MLX operation boundaries;
- map grouped and depthwise convolutions explicitly, including group count;
- preserve BatchNorm/LayerNorm/RMSNorm epsilon and affine parameters;
- preserve residual/downsample branch ordering;
- keep classifier label mapping and top-k order identical to the source.

## Minimal MLX path

1. Port preprocessing and layout conversion.
2. Port the stem convolution/norm/activation.
3. Port one residual or stage block with fixed tensor parity.
4. Port all stages and pooling.
5. Add classifier or feature head.
6. Save/reload and compare logits, embeddings, and top-k labels.

## Parity traps

- RGB/BGR or PIL/OpenCV resize differences;
- rescale before versus after dtype cast;
- NCHW/NHWC transposes hidden inside helper functions;
- SAME versus explicit padding and ceil/floor pooling behavior;
- BatchNorm using training statistics instead of stored eval statistics;
- grouped/depthwise convolution channel grouping mistakes;
- stochastic depth or dropout left active during parity;
- label mapping reordered by sorted class names.

## Optimization ladder

1. Remove accidental host synchronization in preprocessing and postprocessing.
2. Compile stable stem/stage/head regions after primitive and block parity pass.
3. Move repeated transposes to input or weight-load boundaries.
4. Batch fixed image sizes when the product needs throughput.
5. Quantize linear-heavy or pointwise-heavy heads only after top-k and embedding
   quality checks.
6. Consider custom kernels only for a measured hotspot with a readable MLX
   fallback and task-level benchmark.

## Completion gates

- preprocessing tensor matches the source on fixed images;
- NCHW/NHWC conversions are explicit and tested;
- stem, stage/block, pooled feature, and logits pass staged parity;
- top-k labels match on a fixed image set;
- feature extractors document the exact returned layer and shape;
- compile or layout changes show measured end-to-end benefit and preserve parity.
