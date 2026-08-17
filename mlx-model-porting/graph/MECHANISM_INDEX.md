# Mechanism index

Generated from `graph/compiled/evidence-graph.json` by `graph/tools/render_graph_summary.py`. Do not hand-edit; change a shard under `graph/shards/` and regenerate.

Every effect below is an integer basis-point delta against the specific baseline the run was measured on (100 bp = 1 percent). A verdict of `inconclusive` means the measured delta did not exceed the evaluator's own observed same-content rerun spread, so the run does not establish a sign in either direction. Node ids are given so every claim can be traced to its evidence.

## Corpus

| Node kind | Count |
| --- | ---: |
| applied_result | 38 |
| constraint | 6 |
| external_reference | 124 |
| finding | 27 |
| hardware | 3 |
| hypothesis | 12 |
| mechanism | 64 |
| model | 4 |
| trait | 37 |
| workload | 4 |
| **edges** | 575 |

## Mechanisms with measured effects

### `mechanism:qmv-row-regrouping`

**Shape-specific QMV row regrouping** — status `mixed-evidence`, exactness `needs_parity_proof`, provenance `official_verified`.

At one exact quantized shape, change how output rows are partitioned across threadgroups (4+3 to 3+2+2, 4+4 to 3+3+2, or a direct-nibble variant) while leaving every other shape on the promoted path. Arithmetic is intended to be unchanged, but the regrouping is only legal after an exactness proof at that shape. This family produced both the largest measured positive and several regressions, and the sign depends on the exact width, not on the idea.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +2.41% | improved | `result:qwen38-pr414-m6m9-direct-nibble-qmv` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| +0.25% | inconclusive | `result:qwen38-pr256-m8-qmv-3-3-2` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -0.92% | inconclusive | `result:qwen38-e019-m8-qmv-repair` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:bf16-precision-island-draft-head`

**BF16 precision islands in a Q4 draft head** — status `promoted`, exactness `approximate_legal`, provenance `official_verified`.

Preserve every tensor of an incumbent Q4/group-64 draft head byte for byte and add a small number of bare BF16 proposal-only tensors covering the highest-error Q, K, and V rows. K and V are fully corrected while only the worst Q rows are, recovering about 29 percent of the incumbent Q reconstruction error at a bounded size cost. The target verifier, emitted-token authority, rollback, and serial leg are untouched. This is the largest clean positive in the campaign and it is a proposal-quality intervention, not a kernel one.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +1.72% | improved | `result:qwen38-e027-bf16-precision-island-head` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:async-eval-ladder`

**Asynchronous evaluation ladder** — status `promoted`, exactness `exact_by_construction`, provenance `official_verified`.

Submit work asynchronously in graded rungs across verification widths so the GPU is not idle between dependent stages. Scheduling only: no arithmetic changes. Both densifying and removing rungs measured negative, so the promoted rung set is a local optimum rather than a monotone knob.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +0.27% | inconclusive | `result:qwen38-pr211-eight-rung-async-ladder` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -0.09% | inconclusive | `result:qwen38-pr362-async-ladder-densification` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -0.55% | inconclusive | `result:qwen38-pr394-verify-ladder-rung-removal` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -1.28% | inconclusive | `result:qwen38-pr212-kickoff-only-async` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:one-launch-packed-gdn-prework`

**One-launch packed GDN prework** — status `promoted`, exactness `needs_parity_proof`, provenance `official_verified`.

Replace the multi-launch gated-delta prework chain for verify widths three through nine with one packed Metal launch, keeping the graph sigmoid for beta so the FP32 recurrent boundary is preserved. Needs a parity proof because the recurrence feeds trusted state.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +0.09% | inconclusive | `result:qwen38-pr194-one-launch-gdn-prework` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:recurrent-replay-prefetch`

**Recurrent replay prefetch** — status `promoted`, exactness `exact_by_construction`, provenance `official_verified`.

Prefetch the replay tape used to reconstruct recurrent state after a partial rejection, so the rollback path does not stall on a cold read. Five lines; exact because it only moves when the same data is fetched.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +0.03% | inconclusive | `result:qwen38-e020-recurrent-replay-prefetch` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:dv64-gdn-recurrence`

**Dv64 tiled GDN recurrence** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Retile the gated-delta recurrence at value dimension 64 for two specific widths. Locally the isolated recurrence ratios improved; on the ranked run the whole-run effect vanished into rerun spread. A clean example of an isolated kernel win that does not transfer.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.02% | inconclusive | `result:qwen38-e024-t8t9-dv64-gdn` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:qk-shape-gate`

**Short Q/K preparation shape gate** — status `retired`, exactness `exact_by_construction`, provenance `official_verified`.

Gate the fused Q/K normalization and RoPE preparation on a short-width shape test so narrow steps skip the wide path. Measured within rerun spread and not promoted.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.05% | inconclusive | `result:qwen38-pr200-qk-shape-gate` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:proposal-head-cache-preallocation`

**Proposal-head KV cache preallocation** — status `retired`, exactness `exact_by_construction`, provenance `official_verified`.

Pre-size only the proposal head's KV cache to the known scored window so the 256-token block growth never triggers a capacity copy during scored decode. Logically exact and logically free; measured slightly negative and never recovered.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.15% | inconclusive | `result:qwen38-pr105-proposal-head-cache-prealloc` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:grouped-hv-wide-gdn-recurrence`

**Grouped-Hv wide GDN recurrence** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Group the value-head dimension in the wide gated-delta recurrence so several heads share one pass. Parity-clean and measured negative; a later Dv64 variant of the same idea also measured negative.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.22% | inconclusive | `result:qwen38-e018-grouped-hv-wide-gdn` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:prefix-replay-rollback`

**Prefix replay as rollback authority** — status `promoted`, exactness `exact_by_construction`, provenance `official_verified`.

Make prefix replay the sole authority for restoring recurrent state after a rejection instead of keeping wide snapshots. Trades memory traffic for recompute and removes a fail-stop surface, but the measured effect on the ranked path was within rerun spread.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.23% | inconclusive | `result:qwen38-pr231-prefix-replay-sole-authority` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:weight-error-minimised-draft-head`

**Weight-error-minimised draft head** — status `retired`, exactness `approximate_legal`, provenance `official_verified`.

Rebuild the quantized draft head to minimize weight reconstruction error (lower MSE, activation-aware, or output-aware) while keeping the runtime layout. Measured three times, each negative, once catastrophically. Weight reconstruction error is not proposal quality, and proposal quality is not score.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.37% | inconclusive | `result:qwen38-pr208-lower-mse-q4-head` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -3.40% | regressed | `result:qwen38-pr386-activation-aware-q4-head` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -13.28% | regressed | `result:qwen38-pr359-error-minimised-q4-head` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:unchanged-rerun-control`

**Unchanged candidate rerun (null control)** — status `control`, exactness `exact_by_construction`, provenance `official_verified`.

Resubmit byte-identical candidate content to the ranked evaluator with no change. It is not an optimization; it is the control that measures evaluator variation, and it is the only way to calibrate how large a delta has to be before it means anything.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.38% | inconclusive | `result:qwen38-pr236-byte-equivalent-rerun` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -1.30% | inconclusive | `result:qwen38-e014-byte-identical-rerun` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:verify-final-normalization-reuse`

**Verify final-normalization reuse** — status `retired`, exactness `exact_by_construction`, provenance `official_verified`.

Reuse the already-computed post-final-norm verify hidden state for the pending and history rows instead of rebuilding one-row RMSNorm graphs. Exact, strictly less work, and measured slightly negative: the added output liveness outweighed the saved rows.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.40% | inconclusive | `result:qwen38-pr395-verify-final-norm-reuse` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:eval-root-trim`

**Evaluation-root trimming** — status `retired`, exactness `exact_by_construction`, provenance `official_verified`.

Remove cache-state tensors from the blocking evaluation root list so the graph flush waits on fewer outputs. Exact and one line; measured negative on two different bases.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -0.46% | inconclusive | `result:qwen38-pr290-eval-root-trim-newer-base` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -1.54% | regressed | `result:qwen38-pr237-eval-root-trim` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:head-step-cost-retune`

**Draft head-step cost retune** — status `retired`, exactness `approximate_legal`, provenance `official_verified`.

Change the scalar or per-row vector that prices one proposal-head step against one verification step in the round-cost scheduler. Officially measured at 0.14, 0.18, and 0.24 against a promoted 0.20, and later as a per-row vector on a much newer base. Every measurement on both sides of the incumbent was negative, twice reproduced. This is the most thoroughly disproven family in the campaign.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -1.32% | regressed | `result:qwen38-pr146-head-step-cost-018` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -1.51% | regressed | `result:qwen38-pr149-head-step-cost-024` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -1.70% | regressed | `result:qwen38-pr391-per-row-head-cost-vector` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -1.72% | regressed | `result:qwen38-pr392-per-row-head-cost-vector-rerun` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |
| -3.85% | regressed | `result:qwen38-pr235-head-step-cost-014` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:wide-qmv-threshold-retune`

**Wide-QMV dispatch threshold retune** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Lower the output-width threshold at which the quantized matvec switches to its wide kernel. Shape-dispatch thresholds are hardware-tuned upstream; moving one without a profile regressed.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -1.36% | regressed | `result:qwen38-pr201-wide-qmv-threshold-2048` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:semantic-mtp-width-repair`

**Semantic MTP width repair** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Pad and repair a mid-width verification shape so it dispatches on the faster wider kernel. The local dispatch-cliff saving was large; the ranked score regressed, because the ranked prompt pool does not sit on that width.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -1.42% | regressed | `result:qwen38-pr363-semantic-m7-to-m9-repair` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:compiled-proposal-head-history`

**Compiled proposal-head K/V history layout** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Compile the proposal head's K-RMS normalization and the K/V transpose/append layout for the history path. Passed parity and regressed; reopened on a newer base without a better result.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -1.53% | regressed | `result:qwen38-pr209-compiled-history-kv-layout` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:lazy-replay-convolution-boundary`

**Lazy replay-convolution boundary** — status `retired`, exactness `exact_by_construction`, provenance `official_verified`.

Remove the wide convolution carrier from the successful-verification replay tape so the tape is materialized lazily. Exact and less work; measured negative.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -1.75% | regressed | `result:qwen38-pr393-lazy-replay-convolution` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:mixed-precision-draft-head-layout`

**Mixed-precision draft head layout** — status `retired`, exactness `approximate_legal`, provenance `official_verified`.

Promote the draft head's fc projection to BF16 while leaving the rest quantized. Proposal quality improved but did not repay its runtime cost. Contrast with narrow precision islands, which did.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -1.77% | regressed | `result:qwen38-pr360-fc-bf16-mixed-head` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:grouped-sdpa-kv-reuse`

**Grouped SDPA with shared K/V reads** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Group query heads that share a KV head so multi-row verification attention reads each K/V stream once instead of once per query head. Serial single-row attention stays on the stock path. Terminally negative on the ranked path at two group widths.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -2.57% | regressed | `result:qwen38-e016-gh2-grouped-sdpa` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:compiled-mtp-fusion-front`

**Compiled MTP fusion front** — status `retired`, exactness `needs_parity_proof`, provenance `official_verified`.

Compile the proposal head's pre-FC sequence (embedding RMSNorm, hidden RMSNorm, concatenate, quantized fc) into one region. Passed exact parity and regressed four percent; the adjacent history-only variant also regressed. Two independent negatives.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| -4.08% | regressed | `result:qwen38-pr197-compiled-mtp-front` | `model:qwen3.8-27b-mtp` | `hardware:yukon-mlxfast-runner` | `workload:qwen-mtp-ranked-decode` |

### `mechanism:continuous-batching-serving`

**Continuous batching** — status `proven-mlx-port`, exactness `exact_by_construction`, provenance `code_verified`.

Use only for concurrent serving where scheduler complexity is justified. Validation gate: concurrency=1 and many. Rollback: tail latency regression. Observed source band 1.0x-4.3x on metric batch-throughput; the repository holds this claim (promotion_state=withheld) and emits no effective range.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +270.00% | improved | `result:v1-qwen3-0.6b-continuous-batching-16` | `model:qwen3-0.6b` | `hardware:undisclosed-source-reported` | `workload:concurrent-serving-16-streams` |
| +160.00% | improved | `result:v1-qwen3-8b-continuous-batching-16` | `model:qwen3-8b` | `hardware:undisclosed-source-reported` | `workload:concurrent-serving-16-streams` |

### `mechanism:bf16-weight-cast`

**BF16 weight casting for MLX inference** — status `native-mlx`, exactness `unclassified`, provenance `code_verified`.

Cast a controlled F32 MLX port to BF16 only when the exact target workload retains its declared quality window; keep timing results observation-only until every promotion gate, including external signed attestation, passes. Validation gate: deterministic F32-to-BF16 conversion manifest. Rollback: quality output leaves the declared window. Observed source band 1.0x-1.8122x on metric wall-time; the repository holds this claim (promotion_state=withheld) and emits no effective range.

| Effect | Verdict | Result | Model | Hardware | Workload |
| ---: | --- | --- | --- | --- | --- |
| +30.37% | improved | `result:v1-qwen25-05b-bf16-cast-walltime` | `model:qwen2.5-0.5b-instruct` | `hardware:m4-pro` | `workload:six-token-greedy-decode` |

## Mechanisms with no measured effect in this graph

These carry no `applied_result`. Some were rejected on structural grounds before measurement, some have never been scored because a submission gate failed first, and some are registry entries whose evidence is documentation rather than a run. Absence of a measurement here is an open question, never a negative result.

| Mechanism | Status | Provenance |
| --- | --- | --- |
| `mechanism:adaptive-kv-quantization` | research-candidate | research_inference |
| `mechanism:adaptive-round-cost-scheduler` | promoted | official_verified |
| `mechanism:audio-reference-conditioning-cache` | proven-mlx-port | code_verified |
| `mechanism:audio-streaming-and-cache` | proven-mlx-port | code_verified |
| `mechanism:block-weight-streaming` | proven-mlx-port | code_verified |
| `mechanism:cache-privacy-and-isolation` | research-candidate | research_inference |
| `mechanism:compact-draft-vocabulary-selector` | promoted | official_verified |
| `mechanism:compile-stable-region` | native-mlx | code_verified |
| `mechanism:compiled-swiglu` | promoted | official_verified |
| `mechanism:content-prefix-cache-vlm` | rejected-or-superseded | research_inference |
| `mechanism:cuda-graphs-decode-capture` | rejected-or-superseded | research_inference |
| `mechanism:declared-quantized-draft-head` | promoted | official_verified |
| `mechanism:draft-model-speculation` | official-mlx-project | code_verified |
| `mechanism:eagle-medusa-mtp-drafters` | research-candidate | research_inference |
| `mechanism:entropy-coded-weight-stream` | unresolved | code_verified |
| `mechanism:exact-hierarchical-top-two` | promoted | official_verified |
| `mechanism:fast-sdpa` | native-mlx | code_verified |
| `mechanism:fused-qkv-projection` | promoted | official_verified |
| `mechanism:fused-target-lmhead-top-two` | rejected-before-implementation | code_verified |
| `mechanism:generic-audio-prefix-cache` | research-candidate | research_inference |
| `mechanism:lazy-eval-boundaries` | native-mlx | code_verified |
| `mechanism:lazy-verification` | promoted | official_verified |
| `mechanism:moe-expert-dispatch-and-quantization` | native-mlx | code_verified |
| `mechanism:moe-gate-up-fusion` | research-candidate | research_inference |
| `mechanism:moe-gather-and-expert-batching` | native-mlx | code_verified |
| `mechanism:multimodal-content-prefix-cache` | proven-mlx-port | code_verified |
| `mechanism:native-low-bit-weight-quantization` | native-mlx | code_verified |
| `mechanism:packed-gdn-beta-fold` | held | code_verified |
| `mechanism:paired-qmv` | promoted | official_verified |
| `mechanism:prompt-lookup-ngram-speculation` | research-candidate | research_inference |
| `mechanism:prompt-prefix-cache` | official-mlx-project | code_verified |
| `mechanism:qwen3-tts-batch-generation` | proven-mlx-port | code_verified |
| `mechanism:spatial-grid-sample-kernel` | proven-mlx-port | code_verified |
| `mechanism:token-conditioned-lowrank-draft-adapter` | unfrozen | author_claim |
| `mechanism:top-two-confidence-margin-cap` | promoted | official_verified |
| `mechanism:two-bit-compact-draft-readout` | unresolved | code_verified |
| `mechanism:uniform-kv-quantization` | official-mlx-project | code_verified |
| `mechanism:video-input-budgeting` | proven-mlx-port | code_verified |
| `mechanism:vision-feature-cache` | proven-mlx-port | code_verified |
| `mechanism:visual-token-pruning-or-merge` | research-candidate | research_inference |

## Models and traits

- `model:qwen2.5-0.5b-instruct` — Qwen2.5-0.5B-Instruct
  - traits: `trait:dense-decoder-transformer`
  - measured results: 1
- `model:qwen3-0.6b` — Qwen3-0.6B
  - traits: `trait:dense-decoder-transformer`
  - measured results: 1
- `model:qwen3-8b` — Qwen3-8B
  - traits: `trait:dense-decoder-transformer`
  - measured results: 1
- `model:qwen3.8-27b-mtp` — Qwen 3.8 27B with native MTP head
  - traits: `trait:affine-q4-group-64`, `trait:dense-decoder-transformer`, `trait:fp32-recurrent-state`, `trait:gated-delta-net`, `trait:hybrid-attention`, `trait:native-mtp-head`, `trait:shared-token-embeddings`, `trait:speculative-self-draft`
  - measured results: 35

## Generalizable findings and constraints

These are the parts of the corpus that are expected to survive a change of model, hardware, or workload. Each cites its own node id.

### Constraints

- **A declared remote weight artifact must match a freshly downloaded digest** (`constraint:declared-artifact-digest-identity`, official_verified)
  When a candidate declares a remote weight artifact, the official runner resolves the declared revision and compares digests. In this campaign a candidate died at that step with no score and no performance conclusion because the manifest digest did not match the digest of the fetched pinned tree. Publish an immutable revision and recompute the digest from a clean download of that exact revision before submitting.
- **Only harness-declared editable paths are archived** (`constraint:editable-surface-is-gate-zero`, code_verified)
  A benchmark harness that archives a candidate archives only the paths the benchmark manifest declares editable. A change that depends on any other file is invalid before compilation, correctness, or timing is even discussed. In this campaign several locally correct and locally faster candidates were unreproducible for exactly this reason, and several public submissions terminally failed unscored after the frontier moved under them. Enumerate every required path and audit it against the manifest before building a worktree.
- **Local timing is directional; only the official receipt promotes** (`constraint:official-evaluator-is-the-only-authority`, code_verified)
  No local result may promote a candidate. Promotion requires an official receipt bound to the identical candidate identity. This is not bureaucratic: the campaign repeatedly produced isolated local wins that reversed on the ranked run, because the ranked pool uses hidden prompts, a different host, and a median rather than a mean.
- **Model-resident measurement is serial on one GPU** (`constraint:single-model-resident-measurement-lane`, code_verified)
  Setup, model runs, local iterate, and local submit contend for one GPU and must be serialized behind a lease. Static research, history analysis, candidate shaping, and model-free tests can run fully in parallel with a measurement in flight, which is why queue latency is exploitable research time rather than dead time.
- **Target side must be bit-exact; the proposal side only moves acceptance** (`constraint:target-side-bit-exactness`, code_verified)
  In a target-verified speculative decoding system the verifier owns emitted tokens, so every target-side arithmetic change must reproduce identical logits, identical ordered top-two ids and values, identical tie-breaking, identical cache state, and identical rejection rollback. The proposal side has no such obligation: it may be requantized, reshaped, or replaced entirely, because a worse draft costs only acceptance rate. This asymmetry is the single most useful structural fact for anyone optimizing a speculative-decoding stack. It partitions the search space into a cheap high-degrees-of-freedom half and an expensive parity-proof half, and it explains why proposal-quality interventions in this campaign were legal but usually unprofitable while target-side kernel rewrites were the ones that had to survive an exactness proof.

### Findings

- **The same kernel regrouping is positive on one base and negative on another** (`finding:base-dependent-mechanism-sign`, official_verified)
  An M8 QMV row regrouping scored above its base on one promoted main and below its base on a later one, both with full parity. A mechanism's sign is a property of the mechanism-plus-base pair, not of the mechanism. Re-measure after every rebase; do not carry a positive receipt forward across a frontier move.
- **Bundled candidates cannot attribute a component sign** (`finding:confounded-bundles-cannot-attribute`, code_verified)
  Several public submissions advertised a one-line scheduler experiment while their actual diffs spanned four to eight files and removed promoted mechanisms. Whatever those runs scored, they cannot validate the advertised change, and a positive composite score does not license adopting any component. Read the diff, not the note.
- **Removing exact work is not automatically faster** (`finding:exact-work-removal-can-still-regress`, official_verified)
  Four separate exact, strictly-less-work changes measured negative on the ranked run: trimming the evaluation root (twice), reusing an already-computed final-normalized hidden state, lazily materializing the replay convolution carrier, and pre-sizing the proposal head's cache. Under a lazy graph runtime, removing an eval boundary or extending a tensor's liveness changes scheduling and allocation in ways that can cost more than the arithmetic saved. 'It does strictly less' is a hypothesis, not a result.
- **Isolated kernel gains routinely fail to reach the ranked score** (`finding:isolated-kernel-gain-does-not-transfer`, official_verified)
  Three independent instances: a Dv64 recurrence retile whose isolated recurrence ratios improved produced a null whole-run result; a semantic width repair with a large local dispatch-cliff saving scored 1.4 percent negative; and a proposal QMV output reuse that was faster in a compact microbenchmark regressed once serialized. Microbenchmark deltas on a hot kernel do not survive contact with a mixed hidden prompt distribution, because the ranked pool does not spend its time in the shape the microbenchmark measured.
- **Byte-identical candidates score up to 130 basis points apart** (`finding:official-rerun-variation`, official_verified)
  Three official runs of byte-identical or byte-equivalent candidate content scored 2.9042110287045, 2.8930596055435216, and 2.86645713502766. That is a 130-basis-point spread with no code difference at all. Any single-run delta smaller than this is not a measurement of the intervention, and roughly two thirds of the officially scored candidates in this campaign fall inside it. Every applied_result in this graph records its verdict against this spread, which is why many results the campaign called 'rejected' are recorded here as inconclusive: the campaign's decision to retire them was correct policy, but the number alone did not establish harm.
- **Promotion and measured effect are different things** (`finding:promotion-is-not-effect-resolution`, official_verified)
  A leaderboard promotes whatever ranks above the incumbent on one run. Two of the mechanisms in the promoted stack were promoted on deltas of +3 and +9 basis points, one to two orders of magnitude below the evaluator's own same-content spread. They are worth keeping because they are exact and cheap, not because they were shown to help. Reading a promoted stack as a stack of proven wins overstates what the receipts say.
- **Proposal-side changes are cheap to try and rarely pay; target-side changes are the reverse** (`finding:speculative-search-space-is-asymmetric`, official_verified)
  Because the verifier owns emitted tokens, proposal-side interventions need no parity proof and can be attempted quickly, while target-side kernel changes need an exactness argument before they can even be built. In this campaign the cheap half produced many legal attempts and one large win, and the expensive half produced most of the promoted stack. Budget accordingly: prototype freely on the proposal side, but expect the durable structural wins to require the parity work.
- **Unscored surface failures leave whole mechanism families unmeasured** (`finding:surface-failure-yields-no-evidence`, official_verified)
  Two of the most obviously promising ideas in this problem -- a lossless entropy-coded weight stream and a two-bit compact draft readout -- have no official performance evidence in either direction after six combined submissions, because every one of them failed the editable-surface or moving-base gate before it was measured. Resubmitting an identical patch after a surface failure produces another surface failure, not evidence. Treat 'attempted and failed unscored' as an open question, never as a negative.
- **Minimising draft-head weight error does not buy acceptance** (`finding:weight-error-is-not-proposal-quality`, official_verified)
  Three officially scored draft heads built to minimise weight reconstruction error (lower MSE, activation-aware, and an fc-BF16 mixed layout) all measured negative, one by more than thirteen percent. The head that best reconstructs the base weights is not the head that best agrees with the base model's next token, and a draft head's only job is agreement. The one head intervention that did gain corrected specific high-error Q, K, and V rows rather than minimising a global norm.
- **The quantized weight stream dwarfs the logits it produces** (`finding:weight-stream-dominates-readout`, code_verified)
  At the target vocabulary projection the quantized weight stream is roughly 682 MiB per row while the logits tensor is roughly 0.47 MiB, a ratio near 1450 to 1. Any fusion justified by 'we avoid materializing the logits' is optimizing three hundredths of a percent of the traffic while taking on an exact global affine-4 reduction. Check the byte ratio before designing the kernel.
