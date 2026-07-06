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

- `moe-expert-dispatch-and-quantization` - conditionally-lossy expert dispatch
  and precision path;
- `continuous-batching-serving` - lossless serving scheduler path;
- `prompt-prefix-cache` - lossless exact-prefix reuse under serving load.

`vlm-repeated-media` covers `vision-language-omni`. Its registered steps are:

- `vision-feature-cache` - lossless repeated image/video feature reuse;
- `content-prefix-cache-vlm` - lossless repeated-media prefix reuse;
- `multimodal-content-prefix-cache` - lossless media-aware block prefix cache;
- `continuous-batching-serving` - lossless concurrent serving scheduler;
- `cache-privacy-and-isolation` - lossless safety gate, not a speed multiplier.

`tts-batch` covers `flow-diffusion-tts` and `autoregressive-audio-lm`. Its
registered steps are:

- `qwen3-tts-batch-generation` - lossless compatible concurrent TTS batching;
- `audio-reference-conditioning-cache` - lossless repeated reference reuse;
- `audio-streaming-and-cache` - conditionally-lossy streaming/cache mode.

## Derived band rule

`compose_stack_band` derives the advisory compound band. The code is the source
of truth; this prose must stay synchronized with it.

The floor stays `1.0x` until the stack is measured together and a compound
receipt includes `measured_floor` or `floor`. Before that receipt, per-step
floors do not raise the compound floor, even when a source reports a non-1.0
floor.

The ceiling is the product of the ceilings from steps whose method
`improvement_band.provenance` is `source_reported` or `local_reproduced`. Steps
without a band, steps whose provenance is `profile_required`, and steps with any
other provenance are not multiplied into the ceiling.

Profile-required steps are still important. They are returned as
`unmeasured_upside`, meaning they need a profile and local benchmark before they
can contribute a numeric range.

Known-conflicting pairs are excluded before multiplication. If a composition
note has `validity: known-conflicting`, both methods in that pair are omitted
from the compound product and the pair appears in `excluded_conflicts`.

If `compound.measured_together` is false, the derived provenance is
`multiplicative_hypothesis` and the advisor must carry this exact flag:
`unmeasured composition - multiplicative hypothesis, not a claim`.

If `compound.measured_together` is true, the derived provenance becomes
`local_reproduced`. That only means the stack has a local compound receipt; keep
the receipt caveats attached and keep the floor rule above.

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
- rollback condition and caveats.

Benchmark receipts should live under `assets/benchmarks/` once the benchmark
receipt harness exists. Until then, describe the contract in the review packet
and do not fabricate receipt files.

To flip `compound.measured_together`, measure the full registered stack together
against the same baseline receipt, with the same model revision, workload,
config labels, and quality gates. The receipt must state which steps were
enabled, in what order, which composition notes were satisfied, and whether any
registered step was skipped.

Do not mark `compound.measured_together: true` from separate per-step receipts.
Separate receipts justify candidate steps; only an end-to-end stack receipt
promotes the compound band.

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
