# Validation status

**Release snapshot:** 0.5.0

**Review date:** 2026-07-11

This file separates repository-level proof from target-model proof. Offline
tests can demonstrate deterministic routing, safety controls, evidence
boundaries, and artifact consistency. They cannot demonstrate that an unseen
checkpoint has been converted correctly, that a model is useful, or that an
optimization improves a particular Mac workload.

## Canonical snapshot

| Surface | Current checked-in truth | Canonical source |
|---|---:|---|
| Architecture-family routes | 17 | `mlx-model-porting/assets/architectures.yaml` |
| Evidence sources | 350 | `mlx-model-porting/assets/sources.yaml` |
| Technique records | 66 | `mlx-model-porting/assets/techniques.yaml` |
| Optimization-guidance methods | 28 | `mlx-model-porting/assets/optimization_guidance.yaml` |
| Optimization stacks | 4 | `mlx-model-porting/assets/optimization_stacks.yaml` |
| Python scripts | 29 | `mlx-model-porting/scripts/*.py` |
| Benchmark receipts | 13 | `mlx-model-porting/assets/benchmarks/receipt_assessments.json` |
| Performance observations | 11 | generated benchmark assessment |
| Promotion-ready receipts | 1 | generated benchmark assessment |
| Rejected receipts | 1 | generated benchmark assessment |
| Effective claims | 10 | `mlx-model-porting/assets/effective_claims.json` |
| Promoted / withheld claims | 1 / 9 | generated effective-claim catalogue |
| Knowledge-graph nodes / edges | 697 / 499 | `mlx-model-porting/assets/knowledge_graph.json` |
| Offline tests | 423 | `python3 -m unittest discover -s tests` |

The 17 routes are synthetic golden scenarios. They prove that every declared
family has a fixture exercising route selection, expected weight coverage, a
seeded parity failure, and optimization inclusion/exclusion. They do **not**
represent 17 completed real-model ports.

## What the offline gates cover

| Contract | Evidence |
|---|---|
| Weak, unknown, and tied architecture signals stop for manual review; compatible hybrid routes remain explicit. | `tests/test_scenarios.py`, `tests/test_routing_contract.py` |
| Intake is static by default, remote code is not executed, hostile model artifacts are read through bounded no-follow paths, partial shards block recommendations, local paths are portable by default, and truncation blocks clean conclusions. | `tests/test_model_intake_hardening.py`, `tests/test_tooling.py`, `tests/test_hardening_filesystem_contract.py`, `tests/test_hardening_project_inspection_contract.py` |
| Weight-map transforms are explicit and tensor comparison fails on NaN/Inf, shape drift, tolerance failure, or cosine drift. | `tests/test_tooling.py`, `tests/test_scenarios.py` |
| Dense-decoder source capture, fail-closed scaffold generation, schema-2 conversion, MLX capture, and first-divergence parity have dependency-free contracts plus gated Torch/MLX execution tests. | `tests/test_capture_oracle_contract.py`, `tests/test_scaffold_port_contract.py`, `tests/test_convert_checkpoint_contract.py`, `tests/test_parity_runner_contract.py` |
| Recommendations match controlled family, capability, workload, objective, and version identifiers exactly. | `tests/test_recommendation_contract.py` |
| The five advisor buckets are enforced, experimental approaches require opt-in, blocked intake forbids execution, and rejected methods stay rejected. | `tests/test_recommendation_contract.py`, `tests/test_tooling.py` |
| Compound numbers require compatible measured-together coverage and unique evidence lineage; regressions and duplicate composition are not promoted. | `tests/test_recommendation_contract.py`, `tests/test_claim_catalog_contract.py` |
| Historical schema-1 receipts remain observations. Schema-2 promotion requires every runner, execution-attestation, lineage, workload, raw-output, quality, stability, rollback, baseline, and noise gate. | `tests/test_benchmark_evidence_contract.py`, `tests/test_promotion_validation_contract.py` |
| Benchmark assessments, the receipt index, human report, claim catalog, evidence index, and site data are deterministic generated artifacts with drift checks. | `tests/test_benchmark_evidence_contract.py`, `tests/test_claim_catalog_contract.py`, `tests/test_evidence_index_contract.py`, `tests/test_site_data_contract.py` |
| Supported evidence is pinned and typed; review depth cannot silently imply MLX support or local reproduction. | `mlx-model-porting/scripts/validate_sources.py`, `tests/test_evidence_index_contract.py` |
| Research campaigns are review-only, bounded, path-confined, provenance-preserving, and fail on malformed or stale results. | `tests/test_hardening_campaign_contract.py`, `tests/test_hardening_contract.py`, `tests/test_tooling.py` |
| Network origins, redirects, pagination, process trees, output capture, archives, structured inputs, and filesystem traversal are bounded or fail closed. | `tests/test_hardening_network_process_contract.py`, `tests/test_hardening_benchmark_command_contract.py`, `tests/test_hardening_common_contract.py`, `tests/test_hardening_filesystem_contract.py` |
| Distribution text is checkout-agnostic; copy installation is an exact manifest-attested allowlist with a complete in-payload license; symlink installation is mode-aware and idempotent; the retired public worker is absent; and the static site has local runtime dependencies and accessible fallbacks. | `tests/test_distribution_portability.py`, `tests/test_installer_manifest_contract.py`, `tests/test_worker_retirement.py`, `tests/test_site_contract.py` |

## Benchmark and claim boundary

The generated assessment in
`mlx-model-porting/assets/benchmarks/receipt_assessments.json` classifies the
current receipt set as:

| Classification | Count | Meaning |
|---|---:|---|
| `performance_observation` | 11 | A measurement is preserved, but one or more promotion gates are missing or failed. It is not a reusable speed or memory claim. |
| `promotion_ready` | 1 | The attested Qwen BF16 receipt passes every current gate for its exact fingerprint. |
| `rejected` | 1 | The measured configuration regressed or otherwise fails the claim boundary. |

The checked-in observations include older Apple M4 Pro runs, but they use
legacy receipt contracts or lack required lineage, workload, output, quality,
rollback, stability, compatibility, or controlled-runner evidence. Their raw
ratios may help design a future experiment; they must not be advertised as
reliable wins.

A schema-2 candidate becomes `promotion_ready` only when the validator can
independently establish all of the following:

- aggregate metrics recompute from bounded raw evidence;
- either the controlled `python -m mlx_lm generate` invocation matches the
  declared target model and workload, or an external/attested wall-time lane resolves
  an exact safe argv template that executes a digest-pinned Python runner at
  argv position 1 and binds `models.target.id`, `models.target.revision`,
  workload evidence, semantic variant arguments, and a label-owned output. The
  resolved interpreter and sanitized ambient environment are part of the
  target identity, while `-I -B` disables current-directory, user-site, and
  environment-injected imports;
- external wall time comes only from the parent-measured, size-bounded
  `benchmark_command` schema-1 report; copied reports, command output metrics,
  failed/timed-out runs, command/template drift, shells, dynamic code, wrappers,
  secrets, and private ephemeral paths fail closed;
- external warmup count, run count, and a finite positive timeout (maximum 3600
  seconds) are bound into the experiment protocol;
- immutable target, source, and optional draft lineage are pinned;
- checked-in workload artifacts and normalized target/workload hashes match;
- artifact and receipt-output paths are statically rejected when they contain
  symlink components, and the quality contract is snapshotted before execution
  and verified unchanged;
- a schema-2 built-in exact-output-parity contract independently compares a
  digest-bound reference with the candidate artifact recreated and recorded by
  every measured run;
- candidate and baseline experiments are exactly compatible, and the baseline
  path, registered root receipt, digest, metrics, and fingerprint all identify
  the same artifact;
- both runs meet the stability threshold and the gain exceeds the noise floor;
- enabled methods match the invocation; and
- an explicit rollback condition exists; and
- `execution_attested` proves independently that the exact runner and dependency
  bytes exercised the declared model/workload and generated the measured output.

The generic external-command and legacy MLX-LM lanes remain deliberately
unattested. A digest-pinned generic script can ignore its arguments, and the
legacy MLX-LM lane trusts package imports and printed metrics without binding
their bytes. Self-reported attestation fields do not change either result.

The repository-owned `attested-mlx-port-wall-time` adapter is the one narrow
attestable lane. Its trust and evidence design is:

- the receipt executes the checked-in runner directly at argv position 1 under
  the already-bound isolated interpreter (`-I -B`), and both the workload
  artifact and validator re-hash those exact runner bytes;
- before each child process, the parent benchmark harness writes a fresh
  content-random challenge bound to the receipt label, phase, run index,
  command, and snapshotted quality contract. The trusted runner must consume
  that challenge, so evidence copied from another run fails closed;
- the runner validates the declared model, revision, input, workload, and
  variant, hashes the model artifact from disk before loading it, performs the
  fixed MLX workload, writes the quality output, then emits a canonical
  content-signed evidence bundle;
- that bundle binds the challenge, logical argv, checked-in input bytes,
  normalized workload, on-disk model digest and size, generated output, and
  every loaded `mlx`, `_mlx`, and generated port-package file exposed through
  `sys.modules`;
- the runner copies those small loaded dependency files into a bounded
  content-addressed evidence store. The parent snapshots each measured run's
  challenge, bundle, and output. The validator independently re-hashes every
  checked-in snapshot and dependency byte, re-derives the runner, argv, input,
  workload, model-identity, output, challenge, and evidence-set digests, and
  sets `execution_attested=true` only when every measured run agrees; and
- wall time remains exclusively the parent's process measurement. Child
  stdout, stderr, and reported timing values remain non-metrics.

The child necessarily computes the model and dependency digests, but it is no
longer an arbitrary child: the validator anchors that behavior in the reviewed
runner's exact checked-in bytes. A runner that lies has different bytes and
fails the runner digest gate. The large model weights are not duplicated in the
repository; their runtime digest must equal the receipt's target revision and
declared size. Imported Python/extension dependency bytes are retained as
bounded evidence and re-hashed directly.

Residual trust remains explicit. This adapter does not attest firmware, the
macOS kernel, Metal driver/framework dynamic libraries below the imported
Python extension, hardware correctness, Python interpreter semantics, or files
that a native extension opens without exposing them through `sys.modules`.
Those surfaces remain target/environment provenance, not content-attested
dependencies. The adapter is intentionally model/workload-specific; the two
generic lanes remain observations until they gain equally narrow reviewed
adapters. Parent wall time includes model hashing, dependency capture, and
evidence writing performed by the trusted adapter. Promoted ratios therefore
apply only to this attested end-to-end lane and are not pure decode-speed claims.

Legacy Python evaluators, schema-1 declarative scores, and handwritten quality
attestations are recorded only as provenance; the validator does not execute
contributor code and does not accept contributor-selected JSON values,
comparators, or thresholds as promotion proof. Exact-output parity is currently
the only controlled built-in task metric. Lossy or task-specific candidates
remain observations until an equally controlled evaluator is implemented.

The generated `mlx-model-porting/assets/effective_claims.json` is the sole
numeric authority consumed by the advisor. It promotes one fingerprint-scoped
local claim, keeps the other local observations withheld, and withholds every
source-reported range because the exact
source experiment is not representable by the existing `TargetProfile` schema.
Source evidence can justify trying a method, but only a locally reproduced
receipt set can promote its number. The catalog never treats a multiplied stack
ceiling as a measured compound result. A locally promoted range carries one
canonical experiment fingerprint through receipt assessment and claim
generation. Heterogeneous semantic identities cannot be pooled. Compatible
repetitions require the same exact baseline file/digest, warmup/run/timeout protocol,
and exact-output quality contract; they use the conservative minimum ratio and
carry that repetition's complete fingerprint. The advisor exposes the range only when
the `TargetProfile` carries the full canonical fingerprint plus exact
receipt-derived model, target, workload, hardware, and software descriptors and
a non-empty controlled workload set. The fingerprint binds the candidate
receipt digest, aggregates, measured runs, raw-output descriptors and digests,
and quality artifact/result digest; a copied fingerprint digest alone is never
enough.

## What still requires real Apple Silicon execution

- The checked-in Qwen2.5-0.5B-Instruct packet proves one real model in one
  family: `dense-decoder-transformer`. It passed 29 source-to-MLX parity rungs,
  exact greedy-token comparison, and an independent offline MLX-LM cross-check.
- That run does not prove another dense-decoder config or any of the other 16
  routed families. Each still needs its own source oracle, architecture-module
  implementation, complete checkpoint conversion, and parity packet.
- Exact-output parity is the only controlled built-in task quality gate. Domain
  evaluation remains required for language quality, vision, audio, speech,
  diffusion, streaming, scientific tasks, and any lossy change.
- One `bf16-weight-cast` claim is promoted only for the complete attested Qwen
  load-plus-six-token fingerprint. Other models, workloads, hardware/software
  profiles, metrics, and optimization claims still require compatible real
  Apple Silicon receipts.
- Target-chip latency, throughput, memory, energy, compilation, and thermal
  measurements.
- End-to-end validation of a measured-together optimization stack.
- License and publication review for a particular converted checkpoint.

Static source-format inspection of ONNX, GGUF, Flax/Orbax, TensorFlow/Keras,
Core ML, or safetensors artifacts is metadata triage only. The repository does
not lower arbitrary graphs from those formats into executable MLX.

## What requires network access

- Live source-link checks:
  `python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting --check-urls`.
- Upstream revision drift detection and collection of new research candidates.
- Live GitHub contributor collection and external researcher execution.

Offline tests exercise deterministic fakes and contract fixtures for these
paths. That does not prove a live upstream service, token, repository, or result
is available.

## Offline release gates

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/knowledge_curator.py --check-backlog
python3 mlx-model-porting/scripts/validate_benchmarks.py check
python3 mlx-model-porting/scripts/generate_claim_catalog.py --check
python3 mlx-model-porting/scripts/generate_evidence_index.py --check
python3 mlx-model-porting/scripts/generate_site_data.py --check
node --check site/data.js
node --check site/app.js
python3 mlx-model-porting/scripts/manifest.py check
git diff --check
```

`check` modes are non-mutating drift gates. Regeneration commands and ownership
rules are documented in `CONTRIBUTING.md`; `MANIFEST.json` must be regenerated
last after all distributed files are final.
