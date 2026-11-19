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
- [November 12 upstream watchlist (unmerged)](#november-12-upstream-watchlist-unmerged)
- [November 19 upstream follow-up (release-aware)](#november-19-upstream-follow-up-release-aware)

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

## November 12 upstream watchlist (unmerged)

These primary-source MLX-LM changes were **open, not merged or included in a
verified release** on November 12, 2026. Preserve the review-time commit, avoid
promoting performance numbers, and do not silently enable them as supported
features. The latest verified stable source releases remain
[MLX 0.32.4 (October 14)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
and [MLX-LM 0.32.2 (October 16)](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2).

- **Cache correctness:** [draft #2041](https://github.com/ml-explore/mlx-lm/pull/2041)
  (opened November 7; [source snapshot](https://github.com/ml-explore/mlx-lm/commit/acddc4f35dbf31574ce7d73d5cf112d16557ca96))
  targets retained attention-sink visibility during chunked rotating-KV
  prefill. Add split-prefill, sink/window-eviction and cached-vs-full parity
  fixtures before claiming that a local configuration is safe. See the
  [rotating cache gate](attention-and-kv.md#rotating-or-sliding-window-cache).
- **Grouped low-bit experts:** [#2044](https://github.com/ml-explore/mlx-lm/pull/2044)
  (opened November 7; [source snapshot](https://github.com/ml-explore/mlx-lm/commit/42e5d30061a959c888c58d6ff5d0b9f898a62359))
  proposes grouped-expert Metal kernels for 4-bit MoE. Its author-reported
  throughput is specific to an implementation and concurrency regime, not
  reproduced MLX-porting-skill evidence. Before experimenting, compare expert
  selection, dequantization/layout, fused output and logits to the existing
  MLX reference, then measure matched prompts, expert shapes, precision,
  batch/concurrency and cold/warm paths; roll back on quality or cost regressions.
- **DSA indexing:** [#2046](https://github.com/ml-explore/mlx-lm/pull/2046)
  (opened November 9; [source snapshot](https://github.com/ml-explore/mlx-lm/commit/e2115d96ce7e68ff3551379a8352e220ae29c7f5))
  proposes a fused indexer head sum. Treat as a candidate only when the exact
  DSA indexer and intermediate state are present. Compare selected indices,
  scores, masks and logits against the unfused reference; benchmark the full
  selection-and-attention path rather than a microkernel alone.

This update records reviewed upstream *candidate status*, not a fresh research
collector run, successful model port, device execution or promoted claim.

## November 19 upstream follow-up (release-aware)

**Source-reviewed only; not a new collector run, model port or reproduced MLX
measurement.** As checked November 19, 2026, GitHub's latest tagged releases
are [MLX 0.32.4 (October 14)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
and [MLX-LM 0.32.2 (October 16)](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2).
A fix merged on upstream `main` is not yet contained in either tagged release.
The reviewed 40-character commit links below are source snapshots, not promises
about what an installed wheel supports.

- **New, separate batched rotating-cache correctness issue:**
  [MLX-LM #2076](https://github.com/ml-explore/mlx-lm/pull/2076) (opened
  November 18; [reviewed commit](https://github.com/ml-explore/mlx-lm/commit/88df6e796caeaf59eb98af3f4564c8b119fac177);
  **unmerged at review**) reports that after the ring wraps,
  `BatchRotatingKVCache.make_mask` can use the ring cursor to trim padding
  even though the keys were reordered chronologically. This can hide the
  oldest *real* key in a left-padded row during a multi-token update.
  Compare batched against independent single-row masks, logits and outputs
  across repeated wraps, mixed-length rows, 1- and 3-token updates, and
  speculative verification. This differs from #2041's attention-sink bug.
  See [rotating cache](attention-and-kv.md#rotating-or-sliding-window-cache).
- **Half-precision backward safety:** [MLX #4910](https://github.com/ml-explore/mlx/pull/4910)
  (opened November 19; [reviewed commit](https://github.com/ml-explore/mlx/commit/1c117492fd66047c0bfe5433f4a5a47f6119d214);
  **unmerged at review**) reports fused SDPA VJP wrong gradients for large
  bf16/fp16 scores despite a correct forward path. Verify dQ/dK/dV against
  float32/eager across causal/unmasked, GQA and irregular tiles, and measure
  the higher-precision score intermediate's memory/latency cost. An
  inference-only parity test cannot establish training correctness.
  See [backward gate](compile-and-kernels.md#half-precision-sdpa-backward-gate).
- **Recurrent checkpoint training memory:** [MLX #4907](https://github.com/ml-explore/mlx/pull/4907)
  (opened November 17; [reviewed commit](https://github.com/ml-explore/mlx/commit/bbc8c5d0c17b375bc4534e53ed8c740e573400ac);
  **unmerged at review**) proposes cotangent-arrival and VJP-completion
  barriers for chunked recurrence backward scheduling. The author reports
  CPU checks only; Metal remains unvalidated. Compare full gradients and
  growing-length peak memory, including per-step and initial-state gradients,
  before claiming a bounded-memory training path. See
  [training memory gate](runbook-ssm-hybrid.md#checkpointed-training-memory-gate).
- **Distributed NVFP4 expert scales:** [MLX-LM #2077](https://github.com/ml-explore/mlx-lm/pull/2077)
  (opened November 18; [reviewed commit](https://github.com/ml-explore/mlx-lm/commit/77b5eb61800785a7ec219e75cccd8890e0e5b864);
  **unmerged at review**) keeps per-expert `global_scale` vectors replicated
  across tensor-parallel ranks while ordinary scales/biases follow weight
  sharding. Test varying expert counts, gate/up/down projections, rank
  reconstruction, checkpoint reload and single-rank parity; the author's
  two ranks ran on one physical GPU, **not** multiple Macs.
- **Merged on upstream main, not yet in a tagged MLX-LM release:**
  [MLX-LM #1788](https://github.com/ml-explore/mlx-lm/pull/1788) merged
  **November 16** ([reviewed commit](https://github.com/ml-explore/mlx-lm/commit/e4c4af820905a2c660d7e4879a1cce5cc912b1f4))
  for Qwen3.8-Flash-Next (`qwen4_exp`). Its conversion convention folds the
  source RMSNorm `+1` **once** into selected non-gated norm weights from the
  raw Hugging Face layout; converted checkpoints must not be re-folded or
  have the offset subtracted. Check idempotent conversion, intermediate norms,
  padded-batch and cached-continuation parity. Previously converted weights
  with the old convention can silently degrade and require re-conversion.
  This upstream result is not a completed port by this repository.

On a failed correctness, task-quality or memory gate, revert to the tested
full-cache, eager-gradient, single-rank or readable recurrence reference.
Benchmark only matched model/input/software/hardware conditions after those
gates pass. No numeric result or runtime capability is promoted here.
