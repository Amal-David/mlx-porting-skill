# Attention and KV/state optimization

## Contents

- [First classify attention](#first-classify-attention)
- [Native fast SDPA](#native-fast-sdpa)
- [Compressed-attention state contract](#compressed-attention-state-contract)
- [Cache designs](#cache-designs)
- [KV quantization](#kv-quantization)
- [Compressed-cache execution gate](#compressed-cache-execution-gate)
- [Cache compression and sparse attention](#cache-compression-and-sparse-attention)
- [Shared-prefix serving](#shared-prefix-serving)

## First classify attention

Record:

- MHA, GQA, MQA, MLA, cross-attention, local/window, block sparse, linear/recurrent, or hybrid;
- causal/bidirectional mask and padding semantics;
- RoPE/ALiBi/relative position behavior;
- attention sinks or global tokens;
- key/value head count and cache layout;
- prefill versus decode shapes;
- shared or cross-layer cache rules.

## Native fast SDPA

Use the MLX fast scaled-dot-product attention path only when head grouping, mask, causal behavior, scale, sinks, and dtype agree with the source. Test both prefill and single-token decode, because implementations can choose different kernels.

For GQA/MQA, do not pre-tile K/V to the query-head count unless the source semantics require it. Validate mask form and broadcasting: causal string masks and boolean/additive masks must match the intended `[B, N, T_q, T_kv]` attention geometry. Account for float32 softmax behavior and optional attention sinks when comparing with the source.

Keep sparse/window/global-token patterns separate from native fast SDPA. Sliding-window or rotating caches are supported only when the architecture semantics match; Longformer/BigBird-style global token rules, MInference/Quest/NSA-style sparse prefill, and custom FlashAttention variants are research or third-party paths until an MLX implementation and local validation gate exist.

## Compressed-attention state contract

Inspect the actual per-layer schedule before selecting a cache. For example,
[DeepSeek-V4's reference implementation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
distinguishes local attention, overlapping compressed sparse attention, and
non-overlapping heavily compressed attention. Shared K=V, partial RoPE with
an inverse output rotation, and compressor/indexer state are part of its math.
This is architecture evidence, not a claim of scaffold or MLX support.

For any such port, add a tiny source-oracle fixture with:

- lengths immediately before, at, and after each compression/window boundary;
- a prefill split inside an unfinished window, then one-token decode;
- empty compressed history, invalid-index sentinels, and fewer entries than top-k;
- source-equivalent visibility of local and compressed entries, with no future
  token entering attention through a partly filled compression block;
- state save/restore and divergent suffixes from the same prefix, including
  buffer contents, overlap, entry counts, indexer state, and absolute positions.

Compare intermediate masks, selected indices, attention outputs, and final
logits. Never substitute ordinary append-only KV or materialize a shared K=V
tensor twice merely to fit a generic interface. Retain the readable source
path until chunked prefill and cached decode pass the same parity gates.

## Cache designs

### Full KV cache

Best baseline for exact incremental parity. Preallocate when maximum context and memory budget are known, otherwise use a tested growth policy.

### Rotating or sliding-window cache

Use only when the architecture is trained/configured for a window or when eviction semantics are explicitly accepted. Preserve sink/global tokens and positional indexing.

**Regression watch (November 12, 2026):** MLX-LM's [draft PR #2041](https://github.com/ml-explore/mlx-lm/pull/2041) ([reviewed source commit](https://github.com/ml-explore/mlx-lm/commit/acddc4f35dbf31574ce7d73d5cf112d16557ca96)) proposes a fix for a rotating-cache attention mask that could hide retained sink slots during chunked prefill. This is an **unmerged proposal**, not proof that a particular released version has or lacks the defect. Before enabling rotating/sink caches, compare source/reference masks, per-layer attention, logits and greedy output with the full-cache path across unchunked versus split prefill, prefill boundaries at window capacity, repeated eviction, multiple sink counts and single-token decode. Confirm sink entries remain visible while out-of-window entries are masked; keep absolute RoPE positions and state-restore semantics unchanged. Restore full KV/cache fallback if any mask, parity or long-context quality gate fails.

### Prompt/prefix cache

Reuse immutable prefix state across calls only when tokenizer template, model revision, quantization, adapter, positions, and relevant generation configuration match. Include cache namespace/versioning.

For multimodal prefixes, add content hashes and processing knobs for media. Compare cold and warm logits, not only final text. Some cache policies are batch-noninvariant by construction; test mixed batches, eviction, cache reset, and suffixes containing new media placeholders.

### Paged/cache-block designs

Useful for concurrent serving and prefix sharing, but add scheduler and block-table complexity. Validate fragmentation, block boundaries, request cancellation, merge/split, and mixed-length batches.

For multi-tenant RAG or non-prefix cache fusion, include cache-privacy review. Cache timing and chunk-boundary behavior can leak information, so namespace isolation, padding/constant-time policy, tenant controls, and opt-out behavior belong in the serving gate.

For batched generation, add an isolation regression that would fail if KV or prompt-cache state crosses requests. Include mixed prompts, shared prefixes, divergent suffixes, cancellation, and cache reset/reuse in the same batch. Treat release notes or package metadata about cache contamination as a reason to add tests, not as root-cause proof.

### Disk/SSD tiering

Treat as a serving policy, not a free latency win. Benchmark serialization, restore, checksum/versioning, eviction, and storage pressure. Never deserialize untrusted executable objects.

## KV quantization

Start with native/proven uniform KV quantization where supported. Measure long-context quality, because errors accumulate and keys/values can differ in sensitivity. Mixed precision, rotations, nonuniform codebooks, or sub-4-bit methods remain research candidates unless an MLX implementation and workload result exist.

Recent research candidates include layerwise/key-value sensitivity search, RoPE-aware key bit allocation, hardware-aware grouping, and agent-workload-oriented 4-bit KV paths. Do not copy GPU-specific speedups into MLX claims. Promote one only after an MLX implementation proves quality, bytes/token/layer, quantize/dequantize overhead, decode latency, and fallback behavior on the target workload.

Do not assume KV quantization composes with every cache class or serving mode. Validate prompt-cache save/load, rotating cache, batching, trimming, `max_kv_size`, and cache reset/reuse interactions before advertising a combined mode.

For TurboQuant-style or packed custom-kernel KV paths, keep the batch caveat attached: third-party package metadata and papers are not native MLX-core support, and source-reported gains must be rechecked for batch-aware kernels, cache composition, and fallback behavior.

Required checks:

- perplexity/logit drift across context lengths;
- retrieval and reasoning at long context;
- cache bytes/token/layer;
- quantize/dequantize overhead;
- decode latency and throughput;
- interaction with fast attention and speculative verification.

## Compressed-cache execution gate

Treat storage format and execution path as separate decisions. The
[TurboQuant paper](https://arxiv.org/abs/2504.19874v1) motivates online rotated
quantization; the [vLLM evaluation](https://vllm-project.github.io/2026/05/11/turboquant.html)
shows that additional capacity can coexist with slower execution. An October
2026 [Apple Silicon preprint](https://arxiv.org/abs/2610.33076v1) reports
long-context failures when its compressed caches are repeatedly reconstructed.
These are source-reported observations, not local results or a rejection of
all low-bit KV methods.

Before recommending a packed-cache candidate, record:

| Check | Required evidence |
| --- | --- |
| Actual storage | Packed K/V plus scales, norms, rotations/codebooks, residuals, padding, allocator capacity, and any full-precision recent-token region. |
| Attention path | Whether attention consumes packed blocks, reconstructs one bounded tile, or expands the entire history; include all conversions in timing. |
| Live memory | Persistent state, reconstruction workspace, and process/MLX peak measurements, with accounting definitions that avoid double counting. |
| Growth | Context sweep through allocation and packing boundaries up to the target maximum; separate reserve/growth time from steady-state decode. |
| Quality | Same source checkpoint and prompts; long-context retrieval, generation, and per-layer/logit checks, including zero-norm and outlier fixtures. |
| Composition | Batched shapes, partial groups, non-contiguous inputs, trimming, prefix restore, and rejection rollback in the intended cache class. |

Dense attention still reads history; compression alone does not make its
computation sublinear. Prefer a bounded reconstruction path only after parity
and profiling show a benefit. Roll back if the target quality limit fails,
the peak exceeds the memory budget, or the selected latency/throughput objective
regresses beyond noise. A capacity gain may still be useful when explicitly
recorded as a memory-for-latency trade-off. Do not multiply compression factors
from sequential transforms; measure their combined packed representation.

## Cache compression and sparse attention

Methods such as heavy-hitter eviction, attention-based pruning, low-rank projection, sparse prefill, or retrieval-based KV are architecture/workload changes. They are not lossless defaults. Add only with a task benchmark and an exact fallback.

## Shared-prefix serving

Prefix-aware batching can improve arithmetic intensity and memory reuse. Group requests only when the prefix is byte/token identical under the same model/template and cache namespace. Report TTFT and throughput separately; an approach that helps throughput can hurt an isolated interactive request.
