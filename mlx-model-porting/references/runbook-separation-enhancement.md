# Runbook: audio separation and enhancement

## Applies to

Demucs/Roformer/MDX-style source separation, denoising, dereverberation, speech enhancement, and masking/complex-spectral models.

## Architecture fingerprint

Record:

- waveform versus STFT domain;
- sample rate, FFT/window/hop, center/padding;
- complex representation layout;
- chunk length, overlap, context, and window;
- encoder/decoder, U-Net, Transformer/Roformer structure;
- mask type and mixture consistency;
- number/order of output stems;
- normalization and rescaling;
- streaming or offline assumptions.

## Source oracle checkpoints

Capture:

1. normalized waveform and chunk windows;
2. STFT/complex features;
3. encoder/bottleneck features;
4. masks or estimated spectra;
5. inverse STFT output before overlap-add;
6. overlap/window accumulation and normalization;
7. final stems and sum consistency.

## Weight conversion

- map convolution/linear layouts;
- preserve complex real/imag ordering;
- preserve STFT and window config;
- map frequency/time positional encodings;
- preserve stem/output channel order;
- retain normalization statistics and mixture-consistency settings.

## Minimal MLX path

1. Match STFT/iSTFT on deterministic waveforms.
2. Port one block and mask head.
3. Run a short single chunk.
4. Add chunking and overlap-add.
5. Add all stems and postprocessing.
6. Add streaming only if architecture/context allows.

## Parity traps

- STFT centering/window mismatch;
- complex axes swapped;
- frequency/time transpose;
- chunk edge padding and overlap normalization;
- stem order mismatch;
- source peak/RMS normalization omitted;
- output rescale/clipping changes mixture consistency;
- streaming lacks future context assumed by training.

## Optimization ladder

1. Cache windows and reuse overlap buffers.
2. Compile fixed chunk shapes.
3. Use native FFT and fast attention paths.
4. Batch chunks/stems when memory permits.
5. Quantize Transformer/linear blocks separately from spectral input/output.
6. Tune chunk/overlap as a quality-latency-memory trade-off.
7. Custom kernels only for measured complex/masking/overlap bottlenecks.

## Completion gates

- STFT/iSTFT round-trip and exact output length;
- SI-SDR/SNR or task metric versus source;
- stem order and mixture reconstruction checks;
- boundary artifact listening and metrics;
- silence, transient, music, and speech fixtures as applicable;
- RTF includes chunking, FFT, overlap-add, and file conversion.
