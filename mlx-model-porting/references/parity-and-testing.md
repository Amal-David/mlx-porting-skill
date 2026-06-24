# Parity and testing

## The parity ladder

### 1. Artifact parity

Check config values, tokenizer/processor files, special IDs, sample rate, FFT/mel settings, context windows, codebooks, delay pattern, generation defaults, and normalization constants.

### 2. Weight parity

Check key coverage, shape, dtype, transformed statistics, tied parameters, and shard completeness. Randomly sample elements after each nontrivial transform.

### 3. Primitive parity

Use tiny deterministic tensors for normalization, RoPE/position encoding, attention masks, convolution padding, FFT/STFT, quantizer lookup, recurrence, cache update, and sampler logic.

### 4. Block parity

Compare one block with fixed inputs and fixed state. Capture pre-normalization, projections, attention/SSM output, residual branches, MLP/MoE output, and updated state.

### 5. Staged model parity

Compare after frontend/embedding, every N blocks, bottleneck, output projection, and postprocessor. Binary-search the first divergent boundary.

### 6. End-to-end parity

Use deterministic decode or reconstruction. Compare logits/latents before comparing decoded text or audio, because postprocessing can obscure the first error.

### 7. Stateful parity

Compare:

- full sequence versus token/chunk incremental execution;
- empty, growing, rotating, and truncated cache;
- reset/reuse behavior;
- batch merge/split if supported;
- streaming chunk boundaries and final flush.

### 8. Task-quality parity

Tensor allclose is not enough when approximations are introduced.

- LLM: perplexity or log-prob delta, task accuracy, generation agreement under deterministic decoding.
- ASR: WER/CER and timestamps where relevant.
- TTS: intelligibility via ASR, speaker similarity, duration/prosody checks, clipping, and listening tests.
- Codec/vocoder: SI-SDR/SNR where appropriate, spectral distance, PESQ/STOI where licensed and applicable, and perceptual review.
- VLM: task/benchmark sample agreement and visual-token path tests.
- Diffusion/flow: fixed-seed latent and output metrics plus perceptual/task review.

## Tolerance policy

Set tolerances per stage and dtype. Do not use one permissive global tolerance.

Typical policy structure:

| Stage | Precision | Metric | Starting threshold |
|---|---|---|---|
| FP32 primitive | FP32 | max abs / max rel | tight, operation-specific |
| FP16/BF16 block | reduced | allclose + cosine | moderate, inspect accumulation |
| Logits/latents | mixed | max abs + rank/top-k agreement | task-specific |
| Quantized model | low-bit | quality metric and distribution drift | baseline-relative |
| Audio waveform | floating | aligned spectral/perceptual metrics | architecture-specific |

Thresholds are hypotheses, not universal constants. Record why each is acceptable.

## Failure localization

When a final output fails:

1. reproduce with the smallest fixture;
2. compare preprocessing output;
3. find the first divergent checkpoint;
4. disable cache/streaming/quantization/compile;
5. compare in FP32 where possible;
6. inspect shape, axis order, masks, positions, and state update;
7. test the isolated primitive;
8. fix the first divergence only, then rerun the ladder.

## Required regression matrix

At minimum include:

- minimal valid input;
- common input;
- boundary length;
- odd/nonmultiple dimensions when legal;
- empty or reset state;
- long cache/stream;
- batch one and batch greater than one if advertised;
- every supported dtype/quantization mode;
- save/reload;
- deterministic repeatability.
