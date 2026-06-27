# Attention and KV/state optimization

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

## Cache designs

### Full KV cache

Best baseline for exact incremental parity. Preallocate when maximum context and memory budget are known, otherwise use a tested growth policy.

### Rotating or sliding-window cache

Use only when the architecture is trained/configured for a window or when eviction semantics are explicitly accepted. Preserve sink/global tokens and positional indexing.

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

## Cache compression and sparse attention

Methods such as heavy-hitter eviction, attention-based pruning, low-rank projection, sparse prefill, or retrieval-based KV are architecture/workload changes. They are not lossless defaults. Add only with a task benchmark and an exact fallback.

## Shared-prefix serving

Prefix-aware batching can improve arithmetic intensity and memory reuse. Group requests only when the prefix is byte/token identical under the same model/template and cache namespace. Report TTFT and throughput separately; an approach that helps throughput can hurt an isolated interactive request.
