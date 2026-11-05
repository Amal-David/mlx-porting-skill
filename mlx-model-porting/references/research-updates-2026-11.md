# Research decisions reviewed 2026-11-05

Use this reference after source-to-MLX parity to select a falsifiable experiment.
Publication dates below identify the reviewed work; a recent paper is not an
MLX support declaration. No results in this review were reproduced locally.
Keep experimental candidates opt-in and preserve the existing exact fallback.

## Contents

- [1. Match the workload before transferring a kernel](#1-match-the-workload-before-transferring-a-kernel)
- [2. Account for compressed-cache execution](#2-account-for-compressed-cache-execution)
- [3. Require speculative break-even and state parity](#3-require-speculative-break-even-and-state-parity)
- [4. Treat hybrid state as a first-class porting artifact](#4-treat-hybrid-state-as-a-first-class-porting-artifact)
- [5. Test compressed attention at boundaries](#5-test-compressed-attention-at-boundaries)
- [6. Separate media reuse from LM prefix reuse](#6-separate-media-reuse-from-lm-prefix-reuse)
- [7. Evaluate learned low-bit weights as a conversion contract](#7-evaluate-learned-low-bit-weights-as-a-conversion-contract)
- [8. Use kernel search with an independent oracle](#8-use-kernel-search-with-an-independent-oracle)
- [9. Use native runtimes as controlled external baselines](#9-use-native-runtimes-as-controlled-external-baselines)
- [10. Keep diffusion and audio approximations task-specific](#10-keep-diffusion-and-audio-approximations-task-specific)
- [11. Identify what a distributed system actually distributes](#11-identify-what-a-distributed-system-actually-distributes)

## 1. Match the workload before transferring a kernel

**Evidence:** MacPaw's [author report](https://research.macpaw.com/publications/llm-optimizations-apple-silicon)
(September 30, 2026) and [paper](https://arxiv.org/abs/2610.31479v1)
(October 22, 2026) report that KV
pruning, speculative decoding, quantization, and ported kernels behave
differently across Apple devices and workload regimes. Source-reported
Apple measurements do not establish a universal winning configuration.

**Decision:** Separate batch-one decode, small verification matrices, prefill,
and concurrent serving. For each proposed CUDA idea, record the bottleneck,
the required hardware operation, the available Metal equivalent, and whether
the runtime already removes the targeted cost. Check device capabilities;
do not infer them from a chip-family label or a blog's fixed limits.

**Gate / rollback:** Pass source parity and a paired end-to-end benchmark for
the chosen regime. Reject that experiment when its complete workload regresses;
do not blacklist the technique for unmeasured regimes. Apply the dispatch gate
in [the MoE runbook](runbook-moe-transformer.md).

## 2. Account for compressed-cache execution

**Evidence:** [TurboQuant](https://arxiv.org/abs/2504.19874v1), April 2025 / ICLR
2026, is a vector-quantization algorithm. The [vLLM authors' study](https://vllm-project.github.io/2026/05/11/turboquant.html),
May 2026, reports capacity/latency trade-offs in its NVIDIA implementation.
[Double KV Cache Optimization on Apple Silicon](https://arxiv.org/abs/2610.33076v1),
October 2026, reports long-context reconstruction and allocation problems in
its experimental MLX paths. The latter's abstract was reviewed; its results
remain an unreplicated preprint observation.

**Decision:** Record packed storage, metadata, recent-token regions, and live
reconstruction workspace separately. Identify whether attention consumes
packed data, bounded tiles, or a full reconstructed history. Reduced persistent
bytes do not imply lower peak memory or less total execution time.

**Gate / rollback:** Follow [the compressed-cache execution gate](attention-and-kv.md#compressed-cache-execution-gate).
Sweep context and allocation boundaries, enforce the task quality budget, and
include every conversion in timing. Revert on quality, memory, or objective
failure; retain a slower capacity mode only as an explicit measured trade-off.

## 3. Require speculative break-even and state parity

**Evidence:** [EAGLE-3 v3](https://arxiv.org/abs/2503.01840v3), April 2025, and
[DFlash v2](https://arxiv.org/abs/2602.06036v2), May 2026 / ICML 2026, introduce
different trained target-feature drafting contracts. Their server-GPU gains
are algorithm evidence. The MacPaw report above supplies contrasting
source-reported local-workload outcomes, including configuration-specific wins.

**Decision:** Pin the target, feature taps, drafter, tokenizer mapping, precision,
and sampling policy together. Compute elapsed round time per committed output
token, including rejection recovery and stop handling. Sweep block size; do
not use accepted-token count as a proxy for wall-time gain.

**Gate / rollback:** Exercise forced rejection at every block position, EOS,
and continuation after rejection. Compare all KV/recurrent state with the
baseline. Require valid sampling semantics and a matched end-to-end improvement;
otherwise keep autoregressive generation. Use [the serving gate](decoding-and-serving.md#speculative-acceptance-and-cost-gate).

## 4. Treat hybrid state as a first-class porting artifact

**Evidence:** [Qwen3-Next's author release](https://qwen.ai/blog?id=qwen3-next),
September 2025, combines recurrent linear attention with standard attention.
[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791v1), May 2026, changes the
erase/write gate structure. These are architecture contracts, not interchangeable
implementations or proof of the repository's scaffold coverage.

**Decision:** Build a layer-indexed state manifest and a readable recurrence.
Preserve gate dimensions, update order, accumulator precision, convolution
history, attention offsets, and the actual configured layer schedule.

**Gate / rollback:** Compare full prefill, one-token execution, unequal chunks,
nonzero initial state, reset, save/restore, and divergent suffixes. Check terminal
state as well as logits. Revert an optimized scan on any failed schedule;
retain the source-equivalent recurrence. See [the hybrid state gate](runbook-ssm-hybrid.md#modern-hybrid-state-gate).

## 5. Test compressed attention at boundaries

**Evidence:** DeepSeek-V4 was [released by its authors](https://api-docs.deepseek.com/news/news260424/)
on April 24, 2026. The [Transformers reference documentation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
describes heterogeneous attention/cache state and hash-selected versus
learned-routed MoE layers. This establishes source semantics, not MLX support.

**Decision:** Preserve local and compressed branches, partial compression
buffers, indexer state, position transforms, and routing tables. Derive masks
from the source rather than treating compressed entries as ordinary tokens.

**Gate / rollback:** Use tiny fixtures around every window/stride boundary,
including no completed compressed block and invalid top-k sentinels. Verify
future-token exclusion, chunked continuation, and selection/scoring separately.
Keep the readable reference on failure. Use [attention](attention-and-kv.md#compressed-attention-state-contract)
and [MoE](runbook-moe-transformer.md) runbooks together.

## 6. Separate media reuse from LM prefix reuse

**Evidence:** [Native LLM and MLLM Inference at Scale on Apple Silicon v2](https://arxiv.org/abs/2601.19139v2),
January 2026, reports content-based image reuse and continuous batching in the
third-party vLLM-MLX runtime. Its repeated-media workload is not evidence for
arbitrary text prefixes or generic audio reuse.

**Decision:** Key encoder features by processed content and processor/model
identity; key LM state by the complete ordered token/media prefix and its
position/configuration contract. Keep the two cache-hit types visible.

**Gate / rollback:** Test identical media with changed text/order, changed media
with identical text, and cold/warm equality. Report cold, encoder-hit,
prefix-hit, and mixed-request timing. Treat an incompatible identity as a miss;
recompute when parity fails. Follow [prefix caching](decoding-and-serving.md#prefix-caching).

## 7. Evaluate learned low-bit weights as a conversion contract

**Evidence:** [GSQ v2](https://arxiv.org/abs/2604.18556v2), May 2026, jointly
optimizes scalar grid assignments and scales. This motivates an optional
quality experiment; scalar-grid compatibility is not an MLX checkpoint ABI.

**Decision:** Inspect the exported packing, signedness, group axis, scale and
bias convention, per-layer bits, and actual target kernel. Preserve the teacher
and calibration provenance. Compare with a native MLX baseline from the same
floating-point checkpoint before introducing a custom decoder.

**Gate / rollback:** Check dequantized tensors and held-out task quality, not
just calibration loss. Count calibration/export cost separately from inference.
Reject a format that requires silent reinterpretation or unbounded unpacking;
revert on quality or the chosen deployment objective. Keep it experimental
until the complete conversion and execution path is reproduced.

## 8. Use kernel search with an independent oracle

**Evidence:** [K-Search v2](https://arxiv.org/abs/2602.19128v2), February 2026,
separates plan search from implementation. The [July 29 MLX author study](https://bair.berkeley.edu/blog/2026/07/29/cuda-to-mlx-k-search/)
reports attention approaching its native baseline and Mamba prefill gains
against a sequential implementation. Neither establishes a general advantage
over the current MLX kernels.

**Decision:** Describe canonical math, layouts, legal target operations, and
the measured bottleneck before generating candidates. Preserve an independent
oracle and a bounded search log with one hypothesis per trial. Treat source
mapping tables and fast-math identities as hypotheses to verify against Metal
documentation and numerical tests; do not copy fixed hardware limits.

**Gate / rollback:** Test held-out shapes, non-contiguous inputs, partial tiles,
dtypes, and long reductions/scans. Reject incorrect candidates before timing.
Compare with the best installed native path and measure full-model impact.
Keep a native fallback and bounded search budget. Apply [the scan gate](runbook-ssm-hybrid.md#parallel-scan-promotion-gate)
and [kernel checks](compile-and-kernels.md).

## 9. Use native runtimes as controlled external baselines

**Evidence:** [BaseRT's author report](https://huggingface.co/blog/basecompute/basert-release),
July 2026, describes a separate native Metal runtime, reusable buffers, and
inline dequantization. It is not an MLX backend or an implementation supplied
by this skill; its rankings are source-reported for its benchmark setup.

**Decision:** Use allocation/movement/dispatch profiling to form a local MLX
hypothesis. If comparing runtimes, match checkpoint lineage, quantization
recipe, prompt tokens, output policy, context length, and quality budget.

**Gate / rollback:** Include load/compile costs separately and retain raw timing
and memory. Mark an unmatched comparison as exploratory. Revert a proposed
fusion or preallocation policy when parity, peak memory, or the target metric
fails; do not replace the runtime merely because an external headline is newer.

## 10. Keep diffusion and audio approximations task-specific

**Evidence:** [SVDQuant v4](https://arxiv.org/abs/2411.05007v4), November 2025,
combines low-bit computation and a high-precision low-rank branch, with NVIDIA
kernel results. [TeaCache v2](https://arxiv.org/abs/2411.19108v2), March 2025 /
CVPR 2025, estimates denoising change to reuse outputs. These remain useful
candidate designs; neither establishes an MLX speed or audio-quality claim.

**Decision:** Preserve an exact scheduler/denoiser baseline. Separate storage
quantization, low-precision arithmetic, and approximate timestep reuse.
Reusing outputs is lossy even when no retraining is required.

**Gate / rollback:** Fix prompts, seeds, schedules, guidance, shapes, and model
revision. Evaluate full-generation quality and latency, including encoder and
decoder/vocoder. For audio add first-audio time, RTF, and continuity/intelligibility
checks. Disable the candidate when the predeclared task budget is exceeded;
never substitute text perplexity for perceptual quality.

## 11. Identify what a distributed system actually distributes

**Evidence:** Apple's [Minions paper](https://arxiv.org/abs/2610.12970v1), October
9, 2026, distributes independent inference subtasks across intermittently
available Macs. It is task scheduling evidence, not a shared-memory model
parallelism result or an MLX distributed API release.

**Decision:** Distinguish complete-request distribution from tensor/pipeline
partitioning before choosing a design. Record per-machine model capacity,
transport, retries, and data/result identity; pooled RAM does not imply one
address space accessible to a kernel.

**Gate / rollback:** For a multi-Mac experiment, test single-node parity,
interrupted/retried work, duplicate result handling, and end-to-end transfer
costs. Preserve a local execution path when coordination costs exceed the
chosen objective. Consult [distributed execution](runtime-compatibility.md#distributed-execution)
for MLX APIs.
