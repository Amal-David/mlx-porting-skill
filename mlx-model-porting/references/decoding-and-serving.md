# Decoding and serving

## Separate objectives

- Single-user interactive latency.
- Offline single-stream throughput.
- Multi-request throughput and tail latency.
- Long-context TTFT.
- Memory-constrained context or concurrency.

An optimization can help one and hurt another.

## Baseline generation loop

Before advanced decoding, implement exact greedy/sampling generation with:

- correct tokenizer template and special tokens;
- cache/state parity;
- repetition/penalty/logit processors;
- stop criteria;
- deterministic seed behavior;
- streaming output without forced host synchronization per internal operation.

## Speculative decoding choices

### Independent draft model

Use when a compatible smaller model shares tokenizer/vocabulary and has enough acceptance to offset draft/verification cost. Benchmark acceptance length, target verification cost, and memory for both models.

### Prompt lookup or n-gram drafting

Low-memory option for repetitive prompts/documents. Useful only when matches are frequent and lookup overhead stays low.

### Multi-head / Medusa-style

Requires trained extra heads and tree verification. Do not retrofit without compatible weights/training. Quantized verification can change the cost balance.

### Feature-level EAGLE family

Requires architecture-specific drafter weights and access to appropriate target features. Treat each drafter version as a distinct implementation contract.

### Multi-token prediction (MTP)

Use native auxiliary heads when supplied by the model. Verify head ordering, acceptance algorithm, cache update, and quality-preserving sampling.

### Diffusion/block drafters such as DFlash

Use only where a tested MLX implementation and compatible drafter artifact exist. Measure drafting latency and verification geometry rather than relying on advertised theoretical speedup.

When package or model-card evidence names explicit target/drafter pairings, record those pairings as compatibility constraints. Validate acceptance rate, accepted tokens per verification step, drafter cache/window size, verification cache update, and memory overhead. A low acceptance setting or too-small drafter cache can make speculation slower than the baseline.

### Speech speculative decoding

Autoregressive speech tokens can be drafted and verified, but acceptance must respect delay/codebook structure and audio quality. Measure first-audio latency, RTF, intelligibility, and discontinuities.

### Reasoning and thinking budgets

For Qwen-style or other reasoning-budget controls, treat the budget as tokenizer and logit-processor behavior. Capture start/end delimiter token IDs, forced-transition behavior, stop criteria, and whether the source can force an end-of-thinking token. Do not generalize one model family's reasoning controls to every decoder.

## Continuous batching

A correct scheduler must handle:

- request admission during active generation;
- variable prompt and output lengths;
- per-request samplers and stop criteria;
- cache allocation/free, cancellation, and errors;
- batch compaction without state corruption;
- fairness and tail latency;
- multimodal preprocessing dependencies.

## Prefix caching

Key the cache by model revision, adapter, quantization, tokenizer/template, exact token prefix, relevant processor state, and cache format version. Track warm-memory and persistent tiers separately.

For shared or block-level caches, require tenant isolation and a privacy threat model. Non-prefix cache fusion and deterministic chunk scheduling can create timing side channels; do not enable cross-user cache reuse without explicit namespace controls, leakage tests, and an opt-out path.

For multimodal caches, include media content hashes and all processing knobs: processor revision, image/video decode path, frame indices, FPS, max frames, pixel limits, projector version, dtype, adapter, LoRA ID, and tenant salt. Reject LM-only prefix reuse when remaining tokens include media placeholders that are not covered by restored embeddings/KV state.

For audio caches, do not import text/VLM speedup numbers. Include codec/vocoder revision, speaker/reference fingerprint, chunking and flush policy, quantization, and privacy namespace. Validate first-audio latency, RTF, WER/CER or speaker/prosody quality, and boundary continuity.

## Serving validation

Test concurrency one and many, mixed lengths, cancellation, timeout, malformed requests, streaming disconnect, model reload, cache eviction, and repeated prefixes. Store raw load/TTFT/decode/throughput/memory distributions.
