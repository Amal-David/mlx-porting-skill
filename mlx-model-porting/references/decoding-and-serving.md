# Decoding and serving

## Contents

- [Separate objectives](#separate-objectives)
- [Baseline generation loop](#baseline-generation-loop)
- [Speculative decoding choices](#speculative-decoding-choices)
- [Speculative acceptance and cost gate](#speculative-acceptance-and-cost-gate)
- [Continuous batching](#continuous-batching)
- [Prefix caching](#prefix-caching)
- [Serving validation](#serving-validation)

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
[EAGLE-3](https://arxiv.org/abs/2503.01840v3) predicts tokens from fused multi-layer
features rather than retaining EAGLE's original feature-prediction objective.
Pin the feature layer indices, normalization, vocabulary mapping, and trained
target revision; a matching hidden dimension alone is insufficient.

### Multi-token prediction (MTP)

Use native auxiliary heads when supplied by the model. Verify head ordering, acceptance algorithm, cache update, and quality-preserving sampling.

### Diffusion/block drafters such as DFlash

Use only where a tested MLX implementation and compatible drafter artifact exist. Measure drafting latency and verification geometry rather than relying on advertised theoretical speedup.
[DFlash v2](https://arxiv.org/abs/2602.06036v2) uses target features to condition
parallel block drafting. Its published accelerator results establish a
candidate algorithm; they do not establish Apple Silicon speed or compatibility
with arbitrary quantized targets, adapters, or recurrent-state caches.

### Confidence-scheduled block drafters such as DSpark

DSpark combines a block drafter with lightweight semi-autoregressive refinement
and confidence-based scheduling. Treat it as a distinct target/drafter/runtime
contract, not as a drop-in DFlash or MTP switch. A base-model drafter can lose
acceptance after target fine-tuning or adapter changes; pin the target and
drafter revisions and validate them together.

Measure acceptance survival by draft position, proposed and verified block
lengths, confidence calibration, draft time, target verification time, cache
update time, memory, and end-to-end latency. Run both single-user and intended
concurrent serving loads. Warm-start or task-matched drafter training can be a
useful experiment, but it is model training and does not establish MLX serving
support by itself.

`paper-2607-05147` and `fal-ideogram-dspark-2026-07-08` are algorithm and
production-context evidence only. Their DeepSeek/SGLang/B200 results are not
Apple Silicon performance claims.

When package or model-card evidence names explicit target/drafter pairings, record those pairings as compatibility constraints. Validate acceptance rate, accepted tokens per verification step, drafter cache/window size, verification cache update, and memory overhead. A low acceptance setting or too-small drafter cache can make speculation slower than the baseline.

### Speech speculative decoding

Autoregressive speech tokens can be drafted and verified, but acceptance must respect delay/codebook structure and audio quality. Measure first-audio latency, RTF, intelligibility, and discontinuities.

### Reasoning and thinking budgets

For Qwen-style or other reasoning-budget controls, treat the budget as tokenizer and logit-processor behavior. Capture start/end delimiter token IDs, forced-transition behavior, stop criteria, and whether the source can force an end-of-thinking token. Do not generalize one model family's reasoning controls to every decoder.

## Speculative acceptance and cost gate

Use a matched local autoregressive baseline. The October 2026
[MacPaw study](https://research.macpaw.com/publications/llm-optimizations-apple-silicon)
found slowdowns for its common small, quantized chat workload and positive cases
in other configurations. Treat this as evidence to measure each regime, not a
blanket ban on speculation.

For each complete measured run, compute `total_round_time / committed_tokens`.
Count only tokens actually committed to the target output, including a verifier
fallback token when emitted; exclude rejected proposals, padding, and tokens
beyond the stop condition. Separate draft, verification, state update/rollback,
and control/synchronization time without counting overlapping work twice.
Compare the resulting cost with baseline decode at matched context lengths.
Acceptance length alone cannot pass this gate.

Freeze the target, drafter, quantization, sampler/logit processors, prompt
distribution, and output limit. Include short chat completions and the intended
long-generation task; report prefill, TTFT, decode, end-to-end time, and peak
memory separately. Sweep draft budget instead of assuming the largest block wins.

Correctness fixtures must force rejection at the first, middle, and last draft
positions, all-accepted blocks, EOS inside a block, and a following continuation.
Check positions and every cache/state component against non-speculative
execution. For recurrent/hybrid targets, restore a checkpoint and replay the
accepted prefix when state cannot be safely trimmed; never crop an opaque
recurrent state as if it were a token-axis KV array. Include that cost in timing.
For sampling, validate the actual acceptance/resampling distribution and logit
processors; identical random seeds need not consume randomness identically.

Roll back when output/state correctness fails or when the chosen end-to-end
objective regresses beyond the measured noise floor. Keep autoregressive
generation selectable. Do not silently change precision or the task to turn
a failed speculation experiment into an apparent win.

### Release-specific verification geometry

[MLX 0.32.5](https://github.com/ml-explore/mlx/releases/tag/v0.32.5)
contains [Metal split-attention kernels for intermediate verification sizes](https://github.com/ml-explore/mlx/commit/065ec4683cce3c4d3d1d846b4759ad59c13a4ffd)
on NAX and pre-NAX paths. The upstream change targets up to eight query tokens
and the intermediate row regime (query tokens times query heads per KV head).
It can change the optimal draft budget without changing the model. Sweep the
previous best, no-draft and neighboring budgets at both short and long
contexts, including different GQA ratios; validate forced rejection, EOS and
subsequent cache reuse before comparing throughput. Do not copy the upstream
contributor's throughput into a generic recommendation.

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

Separate an encoder-embedding cache from a complete language-model prefix
cache. The [vLLM-MLX paper](https://arxiv.org/abs/2601.19139v2) provides
source-reported evidence for content-based media reuse. An identical decoded
image may justify reusing encoder features under the same processor; it cannot
by itself identify LM state conditioned on different text, media order, or
positions. Test same media with changed questions and reordered media, plus
same text with changed media. Treat a cold/warm mismatch as a cache miss and
fall back to recomputation. Keep cold, encoder-hit, prefix-hit, and mixed-hit
benchmarks distinct.

### Speculative cache publication and looped-layer state

An emitted token need not yet have been consumed by the target forward pass.
[MLX-LM #2083](https://github.com/ml-explore/mlx-lm/pull/2083)
([reviewed proposal](https://github.com/ml-explore/mlx-lm/commit/a4b972f312184988fbd19b57ed10a8b41dad8630),
open on November 26) documents a cache key that includes one more token than
its KV state after speculative generation ends. At publication, bind the key
to the *verified processed prefix*, not the emitted text or a guessed physical
buffer length. Track logical offsets and per-component state; rotating and
recurrent caches cannot generally be repaired by trimming to a tensor length.
If state cannot be aligned and verified, skip publication and recompute.
Test all-accepted rounds ending exactly at the token limit, EOS, cancellation,
partial rejection and a second request sharing/diverging at the last token.
Compare warm continuation with cold greedy output and cache positions.

For weight-shared looped transformers, count **execution visits**, not unique
parameter layers, when allocating per-visit state. [Inkling 2 Mini proposal #2086](https://github.com/ml-explore/mlx-lm/pull/2086)
([reviewed source](https://github.com/ml-explore/mlx-lm/commit/e2b5bbd3bd9d186dea895e60802aa765e70b9d93),
open on November 26) repeats middle layers and uses separate cache entries for
each visit; its pipeline method rejects looped configurations. Distinguish
parameter bytes from per-visit KV/convolution/message state and intermediate
memory. Check visit order, weight identity, per-visit state, full/chunked
prefill, greedy continuation and cache reset, including image inputs where
applicable. Tensor-parallel and pipeline-parallel support must be verified
independently. Do not route this as a supported scaffold or infer pipeline
compatibility from a non-looped relative. Roll back to the source oracle on
any discrepancy.

## Serving validation

Test concurrency one and many, mixed lengths, cancellation, timeout, malformed requests, streaming disconnect, model reload, cache eviction, and repeated prefixes. Store raw load/TTFT/decode/throughput/memory distributions.
