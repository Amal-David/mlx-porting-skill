# Runbook: neural audio codec and speech tokenizer

## Applies to

SoundStream/EnCodec/DAC/Mimi/SNAC/S3/Fish DAC/WavTokenizer/FSQ- or RVQ-based codecs, semantic-acoustic tokenizers, and codec components embedded in TTS or speech-to-speech systems.

## Architecture fingerprint

Record:

- sample rate, channels, normalization and loudness policy;
- encoder/decoder stride product and frame/token rate;
- causal versus noncausal padding and receptive field;
- convolution, residual, Transformer, or Fourier blocks;
- quantizer type: RVQ, grouped RVQ, VQ, FSQ, scalar/product;
- number of codebooks, codebook size/dim, dropout/scalable bitrate;
- semantic and acoustic streams;
- delay/interleaving pattern;
- streaming state, overlap, and final flush;
- decoder/vocoder output range and clipping.

## Source oracle checkpoints

Use synthetic impulses/sines/noise and real speech/music. Capture:

1. normalized waveform;
2. encoder feature after each downsample stage;
3. pre-quantization latent;
4. distances/scores and selected code indices;
5. quantized latent per codebook and aggregate;
6. decoder features after each upsample stage;
7. raw and postprocessed waveform;
8. streaming state and chunk outputs.

## Weight conversion

- convert Conv1d/ConvTranspose1d layouts explicitly;
- verify weight normalization parametrization and removal/folding;
- preserve causal padding/cropping rules;
- map codebook embeddings and ordering exactly;
- distinguish EMA buffers from trainable codebooks;
- preserve quantizer scales, projections, and residual order;
- map Fourier/Vocos heads and STFT parameters;
- record missing training-only discriminators separately.

## Minimal MLX path

1. Implement waveform preprocessing.
2. Port encoder without quantization and compare latents.
3. Port quantizer lookup/search and compare exact indices on fixtures.
4. Port decoder and compare reconstruction from source codes.
5. Run full encode-decode.
6. Add variable bitrate/codebook count.
7. Add streaming/chunk state and flush.

## Parity traps

- one-sample padding/cropping errors that accumulate across strides;
- ConvTranspose output padding;
- channel-first versus channel-last layout;
- weight norm reconstructed incorrectly;
- distance uses normalized versus raw vectors;
- codebook order or residual update reversed;
- delay pattern conflated with codec itself;
- sample-rate mismatch or resampler differences;
- tanh/clipping/gain applied twice;
- streaming overlap or receptive context reset.

## Optimization ladder

1. Eliminate repeated layout conversion between convolution and Transformer sections.
2. Compile stable encoder/decoder regions.
3. Reuse streaming state and overlap buffers.
4. Vectorize codebook distance/search; benchmark native matmul versus custom kernel.
5. Decode only active codebooks for scalable bitrate.
6. Stream decoder output and overlap-add when architecture supports it.
7. Quantize large Transformer/linear components first; keep codebooks, norms, final waveform layers high precision initially.
8. Evaluate low-frame-rate or fewer-codebook modes only if the model was trained for them.
9. Custom kernels for transposed convolution, quantizer search, or Fourier synthesis require end-to-end RTF proof.

## Quality gates

- exact code-index agreement for deterministic unquantized path where practical;
- waveform alignment and length equality;
- no boundary clicks or DC drift;
- spectral and intelligibility metrics plus listening review;
- speech, music, and noise cases if advertised as general audio;
- streaming output matches offline within declared boundary tolerance;
- bitrate/token rate and active codebook count reported.
