# Runbook: flow, diffusion, and non-autoregressive TTS

## Applies to

Matcha-TTS, E2-TTS, F5-TTS, flow-matching acoustic models, diffusion TTS, duration-based non-autoregressive systems, and hybrid TTS where text/reference conditioning produces mel or continuous codec features iteratively.

## Architecture fingerprint

Record:

- text normalization, grapheme/phoneme tokenizer, alignment assumptions;
- reference audio conditioning and speaker/style encoder;
- target representation: mel, latent, continuous codec feature, waveform;
- duration/alignment model or in-context masking scheme;
- denoiser/flow Transformer or U-Net;
- ODE/scheduler, steps, guidance, sway/temperature parameters;
- vocoder/codec revision;
- variable-length mask and padding behavior;
- streaming or chunked synthesis strategy.

## Source oracle checkpoints

Capture fixed-seed:

1. normalized text/tokens;
2. reference mel/features/embedding;
3. duration or target-length calculation;
4. input mask/conditioning sequence;
5. initial noise/latent;
6. model prediction at selected time values;
7. solver states early/mid/final;
8. generated mel/latent;
9. vocoder/codec output.

## Weight conversion

- preserve text embedding and frontend normalization;
- map adaptive normalization/modulation projections carefully;
- preserve time embedding and conditioning concatenation order;
- map convolution and attention layouts;
- store mel/STFT and solver configuration;
- port vocoder/codec independently;
- preserve reference masking and positional behavior.

## Minimal MLX path

1. Freeze preprocessing and target length.
2. Port vocoder/codec and verify source representation decode.
3. Port one denoiser/flow evaluation at a fixed time.
4. Port the exact solver with a tiny number of steps.
5. Compare latent trajectory and generated representation.
6. Run end-to-end fixed-seed synthesis.
7. Add variable lengths, languages, voices, and streaming.

## Parity traps

- different mel frontend or normalization;
- time direction/range and solver convention;
- classifier-free guidance batch ordering;
- random seed/state and noise shape;
- target duration rounding and masks;
- reference segment placement;
- text padding included in attention;
- output representation scaling before vocoder;
- chunked inference changes receptive context.

## Optimization ladder

1. Cache text/reference encodings.
2. Compile stable denoiser shape buckets.
3. Reuse time embeddings and static conditioning.
4. Use fast attention and fused norms.
5. Quantize large linears conservatively; preserve final projections and vocoder initially.
6. Chunk long sequences with overlap and context if architecture permits.
7. Reduce solver steps only as a measured quality mode.
8. Distilled/consistency variants require matching trained weights.
9. Optimize vocoder separately.

## Completion gates

- fixed-time prediction and solver-step parity;
- fixed-seed latent/mel trajectory comparison;
- duration and mask edge cases;
- intelligibility, speaker similarity, prosody, and listening review;
- time to first audio and end-to-end RTF include vocoder;
- step reduction/quantization has explicit quality trade-off.
