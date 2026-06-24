# Research report: an architecture-aware MLX porting skill

**Review date:** 2026-06-24  
**Artifact version:** 0.1.0

## Executive finding

There are useful pieces of the requested system, but the public landscape is fragmented:

- a general MLX development skill covers basic array, neural-network, compilation, and migration gotchas;
- narrow speech skills expose already-supported MLX STT/TTS models;
- MLX-LM, MLX-VLM, and MLX-Audio contain strong conversion and inference implementations;
- MLX serving projects add batching, paged/prefix cache behavior, and benchmark harnesses;
- inference research supplies many candidate techniques, but most papers assume CUDA and do not establish a Metal/MLX win.

What was not found as a cohesive artifact is a **cross-agent, evidence-versioned, architecture-aware runbook that starts from an arbitrary model and enforces intake → source oracle → deterministic weight conversion → staged parity → profiling → optimization → quantization → serving → publication**.

This repository fills that gap as an initial engineering corpus. It does not pretend that every indexed paper has been reproduced on MLX. Instead, it distinguishes native support, proven MLX implementations, transferable candidates, and unverified ideas.

## Existing artifacts audited

### `mlx-dev-skill`

Strengths:

- useful MLX fundamentals: lazy evaluation, `mx.eval`, array indexing differences, NHWC convolution layout, module call conventions, dtypes, and compilation state;
- compact and easy to activate.

Gap relative to this project:

- no architecture fingerprinting or arbitrary-model intake;
- no architecture-specific porting runbooks;
- no source-oracle and layerwise parity protocol;
- no deterministic weight-map schema;
- no evidence/status registry for recent inference methods;
- no audio codec, TTS, ASR, streaming, or vocoder porting methodology;
- no controlled daily research/update process.

### MLX-LM

Provides the strongest official reference for decoder-only and related language-model ports: conversion and quantization, generation caches, prompt caching, speculative decoding, training adapters, and a broad architecture catalogue. It should be the first reference for compatible language models.

### MLX-VLM

Provides active multimodal patterns and, in current code, advanced serving features including continuous batching, automatic prefix caching, quantized KV options, and multiple speculative-drafter families. These features demonstrate why the registry must be revision- and date-aware.

### MLX-Audio

Provides a broad and fast-moving implementation corpus across TTS, STT, speech-to-speech, language identification, codecs, and streaming. Its conversion path already performs model-domain/type detection, model-specific sanitization, dtype conversion, mixed recipes, multiple quantization modes, and publication. It is a critical source of proven patterns, but it is a library—not a generic methodology that can guide an agent through a never-before-ported architecture.

### MLX serving projects

Projects such as `vllm-mlx` explore continuous batching, paged/prefix caching, SSD tiers, sparse prefill, speculative paths, and benchmark storage. These are valuable serving references. They must remain distinct from framework guarantees and should be treated as third-party implementations until validated for the target workload.

## Design conclusions

### 1. Architecture comes before optimization

“Transformer” is too broad. A dense decoder, MoE decoder, Whisper-style encoder-decoder, Mamba-style state-space model, residual-vector-quantized codec, delay-pattern audio LM, flow-matching TTS model, and streaming transducer have different state, cache, shape, quantization, and correctness constraints. The skill therefore routes to a family runbook before selecting techniques.

### 2. The source oracle is the central trust boundary

Most failed ports are not caused by a missing matrix multiplication. They are caused by subtle mismatches in preprocessing, tensor layout, positional encoding, cache semantics, normalization epsilon, tied weights, padding/masking, sampling, codec delay patterns, or streaming state. The skill mandates portable fixtures and staged intermediate comparisons before performance work.

### 3. Techniques need implementation status, not hype labels

Each technique is assigned a status such as:

- `native-mlx`;
- `official-mlx-project`;
- `proven-mlx-port`;
- `research-candidate`;
- `rejected-or-superseded`.

A paper can justify an experiment, not a claim that the technique is beneficial on Apple Silicon. The registry includes a decision rule and validation gate for each technique.

### 4. Audio needs a distinct optimization grammar

Audio pipelines often contain several very different bottlenecks:

- feature extraction or waveform preprocessing;
- semantic/acoustic token generation;
- codebook scheduling;
- codec decode;
- convolutional or Fourier vocoding;
- overlap-add and streaming buffers.

Token/s alone is therefore insufficient. The runbooks use time-to-first-audio, real-time factor, chunk boundary quality, sample continuity, ASR intelligibility, speaker similarity, and peak memory alongside ordinary tensor parity.

### 5. Custom kernels are a final, measured step

MLX already exposes fused and low-level operations. The default sequence is native operation → layout/state correction → stable-region compilation → existing fast primitive → custom Metal kernel. A custom kernel must have a reference implementation, numerical tests, shape/dtype coverage, a fallback, and an end-to-end win—not merely a kernel microbenchmark.

### 6. Continuous updates must be governed

A daily job that edits recommendations automatically is unsafe. New papers can be wrong for MLX, upstream commits can regress, and third-party repositories can be compromised or abandoned. The included update workflow collects candidates, snapshots revisions, runs audits, and prepares review material. It intentionally does not auto-merge.

## Research methodology and limits

The evidence index combines:

- official framework and library source/docs;
- active MLX ecosystem implementations;
- architecture papers;
- recent work on attention, KV management, serving, quantization, speculative decoding, neural codecs, TTS, ASR, and streaming.

Review depth is explicit:

- `synthesized`: the source directly informed a rule, runbook, or registry decision;
- `screened`: abstract/readme/code path reviewed for relevance and limitations;
- `indexed`: catalogued for future review and daily-change detection.

The corpus is intentionally broader than the deeply synthesized subset. It would be misleading to claim line-by-line reproduction of every indexed paper. The skill is designed so evidence can be promoted only through review and tests.

## 2026-06-24 adversarial audit addendum

The publish-readiness audit corrected three source-identity errors before release: Orca now points to the primary USENIX OSDI 2022 page, HQQ now points to the Dropbox implementation instead of an unrelated arXiv paper, and Prompt Lookup Decoding now points to its primary repository instead of an unrelated combined-speculator paper. The MLX core API evidence was also split into concrete, URL-checkable operator docs for fast norm/RoPE, quantized matmul, gather matmul, block-masked matmul, and Hadamard transform.

Latest MLX ecosystem review on 2026-06-24 added native low-bit mode guidance for MLX `affine`, `mxfp4`, `mxfp8`, and `nvfp4`; pinned current MLX, MLX-LM, MLX-VLM, MLX-Audio, and vllm-mlx snapshots; and downgraded or scoped claims that were third-party, cache-composition-sensitive, or paper-only. New June 2026 KV-cache papers were indexed as screened research candidates, not promoted to supported techniques.

The release gate now includes structural provenance validation via `scripts/validate_sources.py`. A supported technique must cite implementation evidence, and moving synthesized sources must carry a snapshot.

## Recommended next validation phase

Before calling this a mature public skill, run a benchmark suite over at least:

1. a dense decoder LLM;
2. a sparse MoE LLM;
3. a Whisper-style ASR model;
4. a neural audio codec;
5. an autoregressive codec TTS model;
6. a flow/diffusion TTS model;
7. a streaming speech model;
8. one deliberately unsupported architecture.

For each case, evaluate whether the agent:

- selects the correct runbook;
- creates a complete weight map;
- catches seeded parity bugs;
- rejects irrelevant CUDA-only optimizations;
- chooses useful MLX-native improvements;
- reports honest, reproducible metrics;
- avoids executing remote code or publishing incompatible weights.
