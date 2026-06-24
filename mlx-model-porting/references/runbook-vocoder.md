# Runbook: neural vocoder and waveform decoder

## Applies to

HiFi-GAN, BigVGAN, Vocos, iSTFTNet, WaveGlow, UnivNet, GAN/Fourier/flow vocoders, and waveform decoder submodules used by codecs or TTS.

## Architecture fingerprint

Record:

- input representation and normalization;
- sample rate, hop length, FFT/window settings;
- generator type: transposed-convolution GAN, Fourier/iSTFT, flow, autoregressive;
- upsample rates/kernels and residual blocks;
- periodic activations, anti-aliasing, weight norm;
- output activation/gain/clipping;
- causal/streaming behavior and receptive context.

## Source oracle checkpoints

Use fixed mel/latent fixtures and capture:

1. normalized input;
2. after each upsample stage;
3. residual block outputs;
4. Fourier magnitude/phase or flow intermediates;
5. raw waveform before output activation;
6. final waveform and exact length.

## Weight conversion

- map Conv1d/ConvTranspose layouts and groups;
- reconstruct/fold weight normalization exactly;
- preserve periodic activation parameters;
- map complex/Fourier heads and iSTFT convention;
- preserve padding, output padding, and cropping;
- exclude training discriminators and record them as intentionally unused.

## Minimal MLX path

1. Freeze input representation from source.
2. Port one upsample/residual block.
3. Port full generator.
4. Verify waveform length/alignment.
5. Add batch and variable length.
6. Add streaming only when architecture supports causal/chunked decode.

## Parity traps

- ConvTranspose output-length mismatch;
- channel layout conversion at every layer;
- weight norm omitted or folded incorrectly;
- periodic activation approximation;
- STFT window centering/padding convention;
- complex real/imag axis swapped;
- output gain/tanh/clipping mismatch;
- chunk reset produces clicks.

## Optimization ladder

1. Remove redundant transposes and host conversions.
2. Compile stable convolution/Fourier regions.
3. Reuse FFT windows and overlap buffers.
4. Test native convolution paths by realistic shape.
5. Quantize only large safe layers; retain final waveform and sensitive normalization at higher precision.
6. Stream with overlap-add and enough context.
7. Consider custom convolution/Fourier kernels only after end-to-end profiling.

## Completion gates

- exact waveform length and alignment;
- spectral/perceptual metrics and listening review;
- silence, impulse, sine, voiced speech, and high-frequency fixtures;
- clipping/DC/boundary checks;
- offline versus streaming comparison;
- RTF includes input conversion and final output handling.
