# Runbook: diffusion and flow models

## Applies to

Image/video/audio diffusion, DiT, latent diffusion, rectified flow, flow matching, consistency/distilled samplers, and related iterative denoising/generation systems. For TTS-specific conditioning and waveform decoding, also use the flow-TTS runbook.

## Architecture fingerprint

Record:

- latent or waveform/pixel domain;
- encoder/decoder/VAE and scaling constant;
- denoiser architecture: U-Net, DiT/Transformer, hybrid;
- noise/flow parameterization (`epsilon`, `v`, `x0`, velocity);
- scheduler/solver and timestep/sigma convention;
- conditioning encoders, CFG, masks, control inputs;
- positional/temporal embeddings;
- attention variants and patch/video layouts;
- fixed versus dynamic shape and number of steps.

## Cost model before optimization

Budget the full request before choosing a technique:

`total latency = conditioning/preprocess + (denoising steps x denoiser branches per step x cost per branch) + decode/postprocess`

Profile each term independently. A faster denoiser kernel does not help if
conditioning or decode dominates, and reducing branch or step count is a model
change rather than a lossless runtime optimization. When techniques target
different terms, measure them separately and then measure the exact stack
together; do not multiply unmeasured speedups.

The following experimental modes require explicit opt-in and their own model
artifacts:

- Quantization-aware distillation may be investigated when post-training
  low-bit inference fails quality. The training fake-quantization path must
  match the exported MLX mode closely enough to test, gradients must flow
  through the simulated quantizer, and final judgment must use the deployed
  real quantized kernel plus task-level quality. A falling teacher/student loss
  is not an image, video, or audio quality gate.
- CFG branch distillation may train one conditional branch to predict the
  guided two-branch teacher. Validate prompt and negative-prompt behavior,
  controls/edits, guidance sensitivity, diversity, text/color fidelity, and
  artifacts; this is not equivalent to setting guidance to one on an
  unmodified model.
- Few-step or distribution-matching distillation requires a pinned trained
  student and scheduler. Sweep the supported step counts and validate
  trajectory stability, adherence, diversity, fine detail, text/color or audio
  quality, and end-to-end latency against the original teacher.

These are research candidates from `paper-2601-20088`, `paper-2405-14867`, and
`fal-ideogram-v4-serving-2026-07-09`, not supported MLX speed claims. The fal
implementation uses NVIDIA-specific NVFP4, Blackwell, and CUTLASS paths; only
the decision framework and validation failures are portable.

## Source oracle checkpoints

Capture fixed-seed:

1. preprocessed conditioning;
2. initial noise/latent;
3. timestep/sigma embedding;
4. one denoiser block and full denoiser output;
5. scheduler update for selected steps;
6. latents after early/mid/final steps;
7. decoded output before postprocessing.

## Weight conversion

- map convolution/linear layout explicitly;
- preserve normalization and modulation order;
- map fused QKV and adaptive norm projections;
- preserve patchify/unpatchify and video axis order;
- map VAE scaling and latent channels;
- keep scheduler config as a first-class artifact;
- preserve conditioning projection and tokenizer/encoder revisions.

## Minimal MLX path

1. Port conditioning encoder or use a pinned compatible MLX implementation.
2. Port VAE/codec encode-decode and establish parity.
3. Port denoiser for one timestep.
4. Port scheduler update.
5. Run a tiny fixed-seed, few-step end-to-end path.
6. Scale to intended resolution/duration and steps.

## Parity traps

- timestep/sigma scaling and ordering;
- `epsilon` versus `v` prediction conversion;
- latent scaling constant;
- CFG concatenation/order and negative conditioning;
- patch/video axis order;
- padding/interpolation behavior;
- attention mask or rotary positions over space-time;
- random generator state and scheduler off-by-one;
- decoder output range/color/audio normalization.

## Optimization ladder

1. Keep random/latent state on MLX and remove host sync per step.
2. Compile the denoiser for stable shapes.
3. Reuse conditioning and static embeddings across steps.
4. Use fast attention and fused norm/linear paths.
5. Cache unconditional/conditional data where CFG formulation allows.
6. Quantize large text encoders/denoiser linears separately and validate quality.
7. Use tiled/chunked VAE or temporal processing for memory.
8. Re-profile norms, activations, casts, and materialized intermediates after
   making a large matmul cheaper; the surrounding operations can become the
   bottleneck.
9. Adopt quantization-aware, guidance-branch, or fewer-step distillation only
   as separately trained experimental model/quality modes, never as lossless
   runtime toggles.
10. Add custom kernels only for profiler-proven repeated hotspots after a
    readable MLX reference, native-fast-op and `mx.compile` attempts, fallback,
    and end-to-end benchmark exist. Do not translate a CUTLASS epilogue or tile
    schedule directly to Metal.

## Completion gates

- one-step denoiser and scheduler parity pass;
- fixed-seed latent trajectory is within declared tolerance;
- output decode/postprocess is verified;
- multiple shapes/resolutions and boundary frames pass;
- quality comparison accompanies any quantization or step reduction;
- guidance-distilled models pass negative-prompt, control/edit, diversity, and
  prompt-adherence gates against the original guided teacher;
- quantization-aware students are evaluated through the real exported MLX
  low-bit inference path, not only fake-quantized training loss;
- benchmark reports first compile separately from per-step and end-to-end time.
