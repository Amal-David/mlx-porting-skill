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
8. Adopt fewer-step/distilled solvers only as a model/quality mode, not a lossless runtime tweak.
9. Add custom kernels only for profiler-proven repeated hotspots after a readable MLX reference and end-to-end benchmark exist.

## Completion gates

- one-step denoiser and scheduler parity pass;
- fixed-seed latent trajectory is within declared tolerance;
- output decode/postprocess is verified;
- multiple shapes/resolutions and boundary frames pass;
- quality comparison accompanies any quantization or step reduction;
- benchmark reports first compile separately from per-step and end-to-end time.
