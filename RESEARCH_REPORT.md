# Research report: an architecture-aware MLX porting standard

**Review date:** 2026-07-11

**Artifact version:** 0.6.1

## Executive finding

The MLX ecosystem contains strong components, but they solve different parts of
the problem:

- Apple-maintained MLX and MLX-LM define the framework and a broad set of
  language-model implementation patterns;
- pinned third-party projects such as MLX-VLM, MLX-Audio, and vllm-mlx provide
  useful multimodal, audio, and serving prior art;
- papers and non-MLX implementations supply architecture and algorithm
  candidates; and
- model repositories supply configuration, preprocessing, checkpoint, and
  license truth for a particular port.

What remains uncommon is a portable engineering standard that joins those
pieces into one conservative flow:

`static intake → architecture route → source oracle → eager MLX graph → deterministic weight map → parity → profiling → evidence-gated advice → packaging`

This repository provides that standard. Dense-decoder Transformers now have an
executable end-to-end path proven on one real checkpoint. The other 16 routed
families retain executable intake, planning, generic validation, and evidence
gates, with architecture-module implementation supplied by their runbooks
rather than a generator. No indexed architecture or technique is treated as
locally reproduced without its own evidence.

## Foundation reconciled in 0.4.0

The full-project review found that the previous implementation mixed useful
research with several unsafe or misleading boundaries:

- the public Worker duplicated repository logic and could drift from the
  checked-in skill and documentation;
- architecture routing could overcommit on weak evidence and did not represent
  known hybrid models as composable routes;
- recommendation matching was too broad to support target-specific numeric
  advice, while generated and handwritten claim surfaces could disagree;
- historical benchmark ratios looked more authoritative than their lineage,
  workload, quality, stability, and execution evidence justified;
- model intake, installers, research collectors, archive readers, subprocesses,
  and network redirects contained unbounded or insufficiently attested paths;
- generated documentation, site data, benchmark assessments, and the
  distribution manifest lacked one complete dependency/drift contract; and
- the nightly curator conflated arXiv base identity with exact revisions and
  did not consistently link known update candidates to prior evidence.

Version 0.4.0 removes the Worker, makes routing confidence-aware and
composable, centralizes numeric authority, withholds every unproved claim,
hardens the executable surfaces, makes generated views deterministic, and
reconciles nightly evidence without silently promoting research. The remaining
limitations are listed at the end of this report; they are proof work, not
missing prose.

Version 0.5.0 adds the dense-decoder execution chain and the first checked-in
real port packet, introduces one narrow benchmark reproducibility adapter with
a resealed promotion boundary, and wires the reviewed knowledge graph into
the advisor as a separate non-executable research queue. The deterministically
reconciled backlog and drift check preserve the T-7007 knowledge-layer state:
697 nodes, 499 edges, bounded advisor consumption, and no numeric authority
outside `effective_claims.json`.

Version 0.6.0 adds the learning-first portal: an MLX mental-model curriculum,
source-to-MLX translation guide, four honest guided journeys, a motion-guided
hero proof loop, and a parity/profile-gated optimization atlas. The atlas keeps
all 28 methods in eight measured-bottleneck families, preserves model-specific
proof boundaries and runbooks in exports, and refuses indexed evidence,
unscoped numbers, research without opt-in, or rejected methods as actionable
guidance.

## Current corpus and proof boundary

The 0.6.1 snapshot contains:

| Surface | Count | Boundary |
|---|---:|---|
| Architecture routes | 17 | Synthetic golden route and guard scenarios; one dense-decoder model has a completed real port. |
| Evidence sources | 361 | Every source has review depth; 35 carry classified support scope and claim types, while 326 remain intentionally unclassified. |
| Technique records | 70 | Status describes evidence maturity, not a portable performance guarantee. |
| Guidance methods | 28 | Target matching and evidence gates determine whether advice can be surfaced. |
| Optimization stacks | 4 | Planning structures; no positive compound number is implied. |
| Python scripts | 33 | Inspectable intake, execution, validation, evidence, and packaging tools. |
| Benchmark receipts | 13 | 12 performance observations, 0 promotion-ready, 1 rejected. |
| Effective claims | 10 | All ten are withheld. |
| Knowledge graph | 708 nodes / 501 edges | Review-only research memory with a reconciled backlog and bounded advisor projection. |
| Offline tests | 473 | Contract, security, determinism, portability, generated-drift, learning-atlas, and gated execution coverage. |

The `bf16-weight-cast` measurement is a reproducible observation for the
captured Qwen load-plus-six-token workload. It is not promotion-ready because
no external signer or out-of-repository trust root exists. All other positive
measurements also remain observations, and missing or failed lineage, workload,
runner, quality, stability, rollback, compatibility, or trust gates continue to
prevent promotion.

## Execution architecture

### 1. Two explicit entry paths

`inspect_model.py` statically inspects source-model metadata, safetensors
headers, source-format manifests, licenses, and risk signals. Network access is
opt-in, and remote model code is never an intake requirement.

`inspect_mlx_project.py` handles an existing MLX project or converted
checkpoint. It inventories runtime surfaces, proof gaps, likely improvement
paths, and contribution candidates without importing or executing the target
project. A truncated inventory blocks a clean conclusion.

### 2. Confidence-aware and composable routing

`assets/architectures.yaml` is the route authority. A family must clear both an
absolute evidence threshold and a lead over the runner-up; weak or tied signals
become an ambiguity blocker instead of a guess. Known hybrids such as recurrent
plus attention or MoE plus recurrent models can emit one primary family and
multiple required component runbooks.

Each declared family has a synthetic golden scenario. That fixture checks
routing, expected weight-key coverage, detection of a seeded parity failure,
and correct optimization inclusion/exclusion. It does not prove that a real
checkpoint for that family has been ported.

### 3. Source oracle before implementation

`capture_oracle.py` freezes the source implementation into reproducible inputs,
intermediate captures, state/cache behavior, and outputs. This is the central
trust boundary: most port failures come from layout, masks, positions,
normalization, tied weights, preprocessing, cache semantics, codec timing, or
streaming state rather than from the matrix multiplication itself.

For dense decoders, `scaffold_port.py` re-inspects the artifact and generates a
minimal eager MLX package. Other families still require their runbook-defined
module implementation. Quantization, compilation, custom kernels, batching,
caching, and speculative execution wait until the smallest useful graph passes
parity.

### 4. Deterministic weight conversion and parity ladder

`convert_checkpoint.py` drafts and applies schema-2 maps of key and shape
transforms. Every rename, transpose, permutation, reshape, split, merge,
squeeze, and tie is reviewable; permissive loading cannot hide missing weights.

`run_parity.py` invokes source and `capture_mlx.py` capture in one command and
stops at the first input, embedding, layer, final-norm, logit, or exact-token
divergence. The Qwen2.5-0.5B-Instruct packet passed all 29 rungs and matched
eight greedy token IDs across Torch, standalone MLX, and offline MLX-LM. This
proves that model and family path only; a tensor tolerance or exact token match
is not a general domain-quality evaluation.

### 5. Target-aware advisor

The advisor matches exact controlled identifiers for family, model type,
capability, workload, objective, and relevant software versions. Missing target
evidence creates a hold rather than an implicit match.

Results are separated into five buckets:

1. `validated-locally` — a local parity, test, or promotion gate supports the
   result for the stated scope;
2. `validated-source-theory` — official APIs, pinned implementations, or
   primary theory justify the path, but target confirmation remains necessary;
3. `benchmark-required` — safe to consider after parity, with no numeric target
   claim yet;
4. `experimental-approach` — unpromoted research requiring explicit user
   opt-in; and
5. `rejected-do-not-use` — incompatible, unsafe, contradicted, superseded, or
   otherwise blocked.

The bucket is not itself permission to skip parity, quality, or licensing work.

### 6. Receipt assessment and numeric-claim authority

`validate_benchmarks.py` independently assesses checked-in receipts. Historical
schema-1 receipts are immutable observations. A schema-2 candidate must bind a
controlled runner to pinned model lineage and checked-in workload artifacts,
preserve raw-output digests, pass a controlled built-in quality contract, match
a compatible baseline, meet stability and noise thresholds, and declare
rollback before it can be promotion-ready. The current controlled quality
contract is exact-output parity over distinct digest-bound artifacts; arbitrary
external JSON values and legacy Python evaluators remain observation-only.
The measured runner may be the MLX-LM generation adapter or the
family-neutral `external-command-wall-time` adapter. The latter resolves a
no-shell argv template that executes a digest-pinned Python runner, binds the
target model and revision, workload evidence, semantic variant, and label-owned
output, and hashes the resolved interpreter plus sanitized environment into the
target identity. It starts Python with `-I -B`, statically rejects symlink
components, and freezes the quality contract before execution. Only wall time
measured by the parent harness counts, never a value printed by the child
command, and every measured run must recreate the candidate artifact used by
the exact-output quality gate.
Neither generic lane is promotion-capable: the external runner can ignore its
arguments, while the MLX-LM lane does not attest imported package bytes and
per-run generated output. The repository-owned
`attested-mlx-port-wall-time` adapter retains a parent challenge, reviewed
runner, loaded dependencies, model/workload identity, and per-run output for
the Qwen worked-port workload. Re-hashing those author-supplied bytes proves
internal consistency, not authenticity; SHA-256 is not a signature.

Promotion requires a protected Apple-Silicon signer to sign the repository
commit/tree, challenge, reviewed dependency manifest, raw output, promotion
policy, and timing. The validator must verify that signature against a
maintainer-controlled trust anchor outside the receipt, evidence tree, and
repository. That signer is future work, so every checked-in receipt has
`execution_attested=false`.

`generate_claim_catalog.py` combines the canonical guidance, source evidence,
and generated receipt assessments into `assets/effective_claims.json`. The
advisor consumes that generated catalogue as its sole numeric authority. A
custom or handwritten assessment sidecar cannot manufacture a local claim
because the advisor recomputes the colocated receipt evidence.

Any future local promotion must also carry a canonical experiment fingerprint from
receipt to assessment to claim. It covers exact model/source revisions, target,
workload, experiment, metric, methods, and baseline binding. Heterogeneous
semantic identities cannot collapse into one range. Compatible repetitions
must share the exact baseline file/digest, warmup/run/timeout protocol, and exact-output
quality contract; they use the conservative minimum and retain that receipt's
full fingerprint. The advisor requires the complete canonical
fingerprint plus exact model/source, target, workload, hardware, software, and
controlled-workload descriptors in the `TargetProfile` before showing the
number. A copied digest or generic profile cannot unlock a local claim.

Stacks are conservative by construction. Unknown pairs stay unknown; duplicate
lineage cannot be multiplied; a regressing member withholds the local claim;
and a measured-together claim must cover the exact ordered methods. A product of
individual ceilings is a hypothesis, never a measured compound result.

### 7. Packaging and publication

A publishable port records source and revision, license, tokenizer or processor,
conversion recipe, weight-map result, parity and quality results, software and
hardware versions, benchmark workload, limitations, and rollback. Converted
weights are not ready merely because they load.

## Evidence architecture

Evidence is layered so that one kind of fact cannot silently impersonate
another:

1. `assets/sources.yaml` records source identity, snapshot, and review depth for
   every entry. Classified records additionally carry support scope, claim type,
   and boundaries; an absent classification grants no implicit claim.
2. `assets/techniques.yaml` records the broader technique decision registry.
3. `assets/optimization_guidance.yaml` records actionable method matching,
   gates, tradeoffs, rollback, and any scoped observation.
4. `assets/recommendation-taxonomy.yaml` defines controlled identifiers,
   TargetProfile requirements, statuses, and the five advisor buckets.
5. `assets/benchmarks/*.json` records experiments; generated assessments decide
   whether any receipt may support promotion.
6. `assets/effective_claims.json` resolves the numeric claim actually visible to
   the advisor.

At this release boundary, all 10 catalogued numeric records are withheld.
Three preserve source-reported benchmark ranges as observations, but none is
profile-eligible: their exact hardware, model, precision, workload geometry,
cache state, concurrency or batch shape, and baseline cannot all be expressed
and matched by the current `TargetProfile` contract. The observations remain
useful research evidence without becoming portable performance promises. The
generator also enforces the durable rule: source-reported numbers remain held
until an equivalent local target-workload run passes every promotion gate.

Review depth is deliberately separate from support scope:

- `synthesized` means a source directly informed a rule, runbook, or decision;
- `screened` means the relevant material and its limitations were reviewed;
- `indexed` means catalogue-only evidence awaiting deeper review.

The controlled support scopes are `context_only`, `local_reproduced`,
`official_mlx`, `official_mlx_project`, `paper_only`, and
`third_party_pinned`. In particular, a pinned non-Apple MLX implementation is
reproducible prior art, not an Apple framework guarantee.

Method status is now mechanically tied to that classification. `native-mlx`
requires synthesized pinned `official_mlx` API or implementation evidence;
`official-mlx-project` requires an Apple-project implementation; and
`proven-mlx-port` requires a synthesized pinned third-party or locally
reproduced path. A one-line status promotion without compatible evidence fails
validation.

The primary ecosystem anchors at this review are MLX v0.32.0 and MLX-LM
v0.31.3 from Apple-maintained projects. The synthesized MLX-Audio anchor remains
v0.4.4; v0.4.5 was collected on 2026-07-10 as a review-only candidate and is not
silently treated as equivalent evidence. MLX-VLM and vllm-mlx are also useful
pinned third-party references and remain labelled accordingly.

The nightly-curator conflict was reconciled without rewriting its July 8
evidence: all 28 files from the 2026-07-08 PR run remain byte-identical. Eleven
older June 27/July 7 research artifacts received one mechanical privacy-only
redaction: machine-specific checkout prefixes became `<repo-root>`. No finding,
source, timestamp, command meaning, claim, or outcome changed. The reviewed
2026-07-10 graph now contains 697 nodes and 499 non-dangling edges, including
lineage edges for every known repository candidate. arXiv base identity is
separate from immutable `vN` revision; three current `v2` papers are held as
`updated_candidate` with explicit before/after comparison state because the
older source records did not pin a paper revision.

## Generated views and dependency flow

Canonical data flows one way into human- and site-facing views:

### Research-to-advisor review queue

`update-candidates.json`, the automated `contributor-refresh.json` source
selection receipt, hand-reviewed contributor learnings, model outcomes, and the
generated backlog feed `knowledge_curator.py`. The curator preserves those
inputs as review-only nodes and edges in `knowledge_graph.json`; it also
reconciles `research_backlog.json` deterministically from the current graph and
update-candidate state. `knowledge_curator.py --check-backlog` fails when that
derived backlog drifts.

`recommend_optimizations.py` reads the graph with a fixed byte limit and a
fail-closed schema check. For the routed architecture families it shows bounded
`candidate_relevant_to`, `evidence_for`, `evidence_for_outcome`, and
`candidate_version_of` provenance under a separate **Unreviewed research
signals (experimental/review queue)** section. Every item remains
non-executable and carries node ids, available source URLs, and review states.
It is not a sixth advisor bucket and cannot provide a numeric claim;
`effective_claims.json` remains the sole numeric authority. The former top-model
popularity snapshot and collector were removed because no architecture-safe
runtime decision consumed them.

```text
sources.yaml
  └─ generate_evidence_index.py ─> EVIDENCE_INDEX.md

benchmarks/*.json
  └─ validate_benchmarks.py ─> receipt_assessments.json
                              ├> receipts_index.json
                              └> BENCHMARK_REPORT.md

optimization_guidance.yaml + sources.yaml + receipt_assessments.json
  └─ generate_claim_catalog.py ─> effective_claims.json

update-candidates.json + contributor-refresh.json + reviewed research assets
  └─ knowledge_curator.py ─> knowledge_graph.json
                            └> research_backlog.json (reconcile/check)

knowledge_graph.json + effective_claims.json
  └─ recommend_optimizations.py ─> five advice buckets
                                   + separate unreviewed research signals

VERSION + canonical registries + references/*.md + generated assessments/claims
  └─ generate_site_data.py ─> site/data.js

final distributed tree
  └─ manifest.py ─> MANIFEST.json
```

Generated files are review artifacts, not editing surfaces. Their drift checks
make a registry, receipt, version, or report change fail until every dependent
view is regenerated. `MANIFEST.json` is regenerated last.

## Extension flow

### Add or change an architecture family

1. Add detection signals, runbook path, routing constraints, and any explicit
   hybrid composition to `assets/architectures.yaml`.
2. Add or update the family runbook with state/cache semantics, weight
   transforms, parity checkpoints, optimization order, quantization exclusions,
   failure modes, and a minimal validation matrix.
3. Add a synthetic scenario that exercises routing, weight coverage, a seeded
   parity failure, and positive/negative optimization selection.
4. Verify that every declared family still has golden coverage and that weak
   evidence fails closed.

This extends routing knowledge. A real-model port still requires its own source
oracle, conversion implementation, and validation packet.

### Add or change a technique

1. Register primary or pinned evidence in `sources.yaml` with honest review
   depth, support scope, and claim types.
2. Add the technique status and decision rule to `techniques.yaml`.
3. Add actionable guidance only when family/capability/workload targeting,
   correctness and quality gates, benchmark protocol, tradeoffs, and rollback
   are explicit.
4. Add stack composition notes only for the exact pair and order supported by
   evidence; unknown remains the default.
5. Add recommendation and source-validation tests, then regenerate dependent
   views.

A paper-only or CUDA implementation can justify `research-candidate`, not
supported MLX guidance.

### Add a benchmark candidate

1. Record a schema-2 experiment with immutable target and source lineage,
   checked-in workload artifacts, either a controlled MLX-LM invocation or the
   digest-pinned family-neutral wall-time adapter, and bounded raw outputs.
2. Pair a candidate with an exactly compatible baseline and declare the enabled
   methods, primary metric, stability limit, controlled exact-output quality
   contract, and rollback condition. Lossy or task-specific candidates remain
   observations until a suitable built-in evaluator exists.
3. Regenerate benchmark assessments; do not edit the assessment, index, or
   report by hand.
4. Regenerate the effective claim catalogue. A failed gate remains visible as
   an observation or rejection and cannot be rewritten as a success in prose.

### Add research evidence

Automated research may collect, normalize, rank, and prepare review material.
It may not upgrade `indexed` to `synthesized`, promote a technique, modify the
claim catalogue, or auto-merge a recommendation without human review and the
required gates.

### Change public documentation or version data

Edit the authored Markdown or canonical asset, then regenerate the evidence
index, benchmark views, claim catalogue, and site data as their inputs require.
The static site has no worker dependency; `site/data.js` is derived locally so
the published documentation and offline file view share the same corpus counts.

## What remains deliberately unresolved

- No arbitrary-model architecture generator is shipped.
- Synthetic route coverage is not end-to-end checkpoint support; only one
  dense-decoder checkpoint has completed the full executable chain.
- The captured Qwen BF16 timing remains a reproducible observation, not a
  promoted claim or portable model, workload, or metric guarantee.
- Source-reported performance remains constrained to the cited revision,
  hardware, model, inputs, workload, and metric.
- MLX techniques inferred from CUDA or papers remain experimental until a
  reproducible MLX path passes parity and target-workload measurement.
- Domain evaluations beyond exact output remain future work for language,
  audio, vision, diffusion, scientific, long-context, and streaming ports.

The next evidence milestone is not a larger unqualified number. It is a small
set of real, source-pinned ports with complete source oracles, deterministic
weight maps, task-quality gates, and compatible schema-2 baseline/candidate
receipts.
