# Compound optimization stacks

## Stack purpose

An optimization stack is an advisory sequence of compatible experiments for one
model family and workload class. It is not a promise of speedup and it is not a
shortcut around profiling, parity, or quality gates.

Use a stack when a user asks for a speedup plan, asks how techniques combine, or
asks what compound gain might be possible. Keep the answer in planning-band
language until local receipts exist.

The registry lives in `assets/optimization_stacks.yaml`. The numeric method
bands live in `assets/optimization_guidance.yaml`. The advisor derives compound
bands at runtime through `compose_stack_band`; do not store compound ranges in
the stack registry.

## Shipped stacks

`dense-decoder-inference` covers `dense-decoder-transformer` and
`moe-decoder-transformer`. Its registered steps are:

- `native-low-bit-weight-quantization` - conditionally-lossy weight memory and
  throughput mode;
- `fast-sdpa` - lossless native attention path when semantics match;
- `compile-stable-region` - lossless repeated-region compile;
- `prompt-prefix-cache` - lossless exact-prefix reuse;
- `uniform-kv-quantization` - conditionally-lossy KV memory mode;
- `draft-model-speculation` - lossless speculative verification when the draft
  and target pair is compatible.

`moe-serving` covers `moe-decoder-transformer`. Its registered steps are:

It is selected only for a controlled `concurrent-serving` or `server` workload
and a matching serving objective.

- `moe-expert-dispatch-and-quantization` - conditionally-lossy expert dispatch
  and precision path;
- `continuous-batching-serving` - lossless serving scheduler path;
- `prompt-prefix-cache` - lossless exact-prefix reuse under serving load.

`vlm-repeated-media` covers `vision-language-omni`. Its registered steps are:

It is selected only for a controlled `repeated-media` or `multimodal-serving`
workload and a matching TTFT, prefill, or concurrent-throughput objective.

- `vision-feature-cache` - lossless repeated image/video feature reuse;
- `multimodal-content-prefix-cache` - lossless media-aware block prefix cache;
- `continuous-batching-serving` - lossless concurrent serving scheduler;
- `cache-privacy-and-isolation` - lossless safety gate, not a speed multiplier.

`tts-batch` covers `flow-diffusion-tts` and `autoregressive-audio-lm`. Its
registered steps are:

It is selected only for a controlled concurrent-TTS, repeated-reference, or
streaming-audio workload and a matching batch, first-audio, or RTF objective.

- `qwen3-tts-batch-generation` - lossless compatible concurrent TTS batching;
- `audio-reference-conditioning-cache` - lossless repeated reference reuse;
- `audio-streaming-and-cache` - conditionally-lossy streaming/cache mode.

## Derived band rule

`compose_stack_band` derives the advisory compound band. The code is the source
of truth; this prose must stay synchronized with it.

Each stack declares `primary_metric`. The hypothesis floor stays `1.0x`; per-step
floors do not raise the compound floor, even when a source reports a non-1.0
floor.

The generated `assets/effective_claims.json` is the only numeric authority.
Before composition, the advisor replaces every raw guidance band with that
catalogue's effective state. A step can enter a hypothesis ceiling only when its
catalog-authorized range matches the stack `primary_metric`, its provenance is
eligible for numeric use, and its band carries
`numeric_authority: effective_claims`. Raw source prose and handwritten
receipt sidecars never enter the product.

Bands for a different metric are never multiplied into the primary metric
ceiling. They are returned as `other_metric_upside` and rendered as
workload-conditional upside for that other metric. For example, a TTFT proxy
prompt-cache ratio cannot raise a decode-throughput ceiling.

Profile-required steps are still important. They are returned as
`unmeasured_upside`, meaning they need a profile and local benchmark before they
can contribute a numeric range.

Every pair among retained stack steps must be explicitly
`validated-composable` before a numeric hypothesis is emitted. Unlisted pairs
default to `unknown`; `known-conflicting` and `mutually-exclusive` pairs remain
visible as exclusions. Numeric steps must also have nonempty, unique evidence
lineage ids so one experiment cannot be counted twice.

If `compound.measured_together` is false, the derived provenance is
`multiplicative_hypothesis` and the advisor must carry this exact flag:
`unmeasured composition - multiplicative hypothesis, not a claim`.

The hypothesis ceiling provenance is always `multiplicative_hypothesis`, even
when `compound.measured_together` is true. A measured headline appears only
when the generated receipt assessment is promotion-ready, the receipt hashes
and comparable baseline verify, the primary metric matches, and the exact
ordered enabled-method list covers the registered stack. The measured value
carries `local_reproduced`; the hypothesis product stays secondary and flagged.

## Composition validity

A stack can include `composition_notes` for pairs whose interaction matters.
Pairs not listed default to `unknown`.

Use these validity values:

- `validated-composable`: the pair has evidence that the two methods can run
  together under the named gates. This can also mark a required safety gate that
  enables a speed step, such as cache privacy plus a shared cache.
- `unknown`: the pair is plausible but unmeasured together. Keep separate gates
  and do not imply the product has been proven.
- `known-conflicting`: the pair is unsafe, semantically incompatible, or known
  to invalidate the benchmark. The compound derivation excludes both methods.

To add a composition note, edit the relevant stack in
`assets/optimization_stacks.yaml`:

- set `pair` to exactly two method ids already present in that stack;
- set `validity` to one of the values above;
- write `why` as the shortest useful reason, anchored to a reference, source,
  benchmark receipt, or failed validation gate;
- run `scripts/validate_sources.py` so missing methods, invalid validity values,
  and forbidden stored compound ranges fail loudly.

Do not add a note just because two steps appear near each other. Add one when
the interaction changes semantics, measurement, risk, or eligibility.

## Receipt promotion

Use the hypothesis-led promotion bar before changing stack evidence or method
bands. A speed claim needs source provenance, a reproducible MLX path, a
correctness or quality check, benchmark metadata, a rollback condition, and the
affected registry or runbook.

To promote a method band from `profile_required` to `local_reproduced`, attach a
local benchmark artifact that proves the method on the target workload. The
receipt must identify:

- chip and Mac class;
- MLX and MLX-LM, MLX-VLM, or MLX-Audio versions as applicable;
- exact model revision and artifact revision;
- config labels for dtype, quantization, cache, compile, batch, prompt/media,
  generation, and serving mode;
- baseline receipt id and candidate receipt id;
- correctness or quality gate result;
- target metric, repetitions, median/tail or distribution summary, and raw
  artifact path;
- a canonical fingerprint binding the candidate receipt SHA-256, aggregates,
  every measured result and raw-output descriptor/digest, and the quality
  artifact/result digest;
- rollback condition and caveats.
- an external signature covering the repository commit/tree, challenge,
  reviewed dependency manifest, raw output, promotion policy, and timing,
  verified against a maintainer-controlled trust anchor outside the submitted
  receipt/evidence and repository.

The generic external-command and legacy MLX-LM lanes do not pass that final
gate. The narrow `attested-mlx-port-wall-time` lane preserves digest-bound
reproducibility evidence for its exact reviewed model/workload, but it cannot
promote without the external signer and trust root. It does not promote a stack
or authorize another runner.

Benchmark receipts live under `assets/benchmarks/`. Generate and validate their
assessments with `scripts/validate_benchmarks.py`; never hand-edit the generated
assessment, index, or report to manufacture promotion.

Advisor use requires more than copying that fingerprint's digest into a generic
profile. The `TargetProfile` must carry the full canonical fingerprint, exact
receipt-derived models, target, and workload objects, exact non-empty hardware
and software descriptors, and a non-empty controlled workload set.

To flip `compound.measured_together`, measure the full registered stack together
against the same baseline receipt, with the same model revision, workload,
config labels, and quality gates. The receipt must state which steps were
enabled, in what order, which composition notes were satisfied, and whether any
registered step was skipped.

Do not mark `compound.measured_together: true` from separate per-step receipts.
Separate receipts justify candidate steps; only an end-to-end stack receipt
adds a measured compound headline. It does not promote the hypothesis product to
`local_reproduced`.

Rollback requirements belong beside the evidence. If quality drifts, parity
fails, tail latency regresses, memory exceeds budget, cache isolation breaks, or
the win disappears outside a narrow microbenchmark, keep the method or stack in
profile-required or experimental status and record why.

## Ordering discipline

Execute lossless runtime steps before conditionally-lossy or model-changing
steps. Start with parity-preserving paths: layout cleanup, fast native kernels,
compile of stable regions, cache correctness, batching, and serving state
isolation.

Only after those are measured should you try conditionally-lossy steps such as
weight quantization, KV quantization, streaming approximation, pruning,
distillation, layer skipping, or reduced diffusion/flow steps.

Conditionally-lossy and otherwise unvalidated approaches require explicit
experimental consent under SKILL rule 11. Say: "This is an experimental
approach. Do you want to try it?" Continue only after the user opts in.

Keep the registry order as a planning scaffold, not permission to skip the
discipline. If a stack lists a lossy step early, apply it only after the relevant
lossless baseline and gates are in place unless a measured dependency requires a
different order.
