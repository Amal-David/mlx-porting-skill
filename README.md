# MLX model porting, inspection, and evidence-gated optimization

`mlx-model-porting` is a portable Agent Skill for guiding an unfamiliar
PyTorch or Hugging Face model into a correct MLX implementation, or for
inspecting an existing MLX project before improving it. It provides static
intake, architecture routing, source-oracle and weight-map guidance, parity
gates, profiling tools, and conservative optimization advice.

This repository is an engineering workflow, **not a generic converter that can
turn any checkpoint into a working MLX model**. The 17 architecture routes are
covered by synthetic golden scenarios; they are routing and guard fixtures, not
17 completed real-model ports.

The public runbook is available at
[mlx-porter.pages.dev](https://mlx-porter.pages.dev/), and its offline source is
in [`site/`](site/).

## What is in the repository

| Layer | Purpose |
|---|---|
| [`mlx-model-porting/SKILL.md`](mlx-model-porting/SKILL.md) | Compact agent contract and trigger map. |
| [`mlx-model-porting/references/`](mlx-model-porting/references/) | Porting method, failure atlas, optimization guides, and 17 family runbooks. |
| [`mlx-model-porting/assets/`](mlx-model-porting/assets/) | Canonical architecture, technique, guidance, stack, source, benchmark, and claim data. |
| [`mlx-model-porting/scripts/`](mlx-model-porting/scripts/) | Non-destructive inspection, planning, parity, benchmarking, evidence, and packaging tools. |
| [`tests/`](tests/) | Synthetic scenarios and contract, security, determinism, and portability tests. |

Current 0.4.0 corpus snapshot:

- 17 architecture-family routes with synthetic golden coverage;
- 350 evidence sources with explicit review depth; 23 currently carry classified
  support scope and claim types, while 327 remain intentionally unclassified;
- 65 technique records, 27 optimization-guidance methods, and 4 stack
  definitions;
- 11 checked-in benchmark receipts: 10 `performance_observation`, 0
  `promotion_ready`, and 1 `rejected`.

## How the workflow fits together

1. **Inspect without executing model code.** Inventory configuration,
   safetensors metadata, source formats, licenses, and remote-code risks. Local
   intake binds every file into a portable SHA-256 tree identity; incomplete
   identity or missing artifact-bound license evidence blocks advice.
2. **Route to a family runbook.** Confidence and ambiguity gates can stop a
   weak route; hybrid models can require more than one runbook.
3. **Build a source oracle and the smallest eager MLX graph.** Freeze fixtures
   and intermediate tensors before changing performance behavior.
4. **Convert weights deterministically.** Every rename, transpose, reshape,
   split, merge, and tie belongs in an inspectable weight map.
5. **Pass staged parity.** Compare weights, blocks, state/cache behavior,
   outputs, and task-specific quality before optimization.
6. **Profile, then advise.** Recommendations match controlled family,
   capability, workload, objective, and software identifiers exactly. A local
   numeric promotion additionally requires the exact receipt-derived experiment
   fingerprint, not a look-alike generic profile.
7. **Measure one change at a time.** Benchmark receipts are observations until
   lineage, workload, runner, stability, quality, rollback, and compatibility
   gates all pass.
8. **Package with provenance.** Preserve source revision, license, conversion
   recipe, validation results, limits, and rollback conditions.

See [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) for the architecture and extension
flows, and [`VALIDATION.md`](VALIDATION.md) for exactly what the checked-in gates
do and do not prove.

## Install

The distributable skill is the `mlx-model-porting/` directory. For a supported
client preset:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex
```

For an explicit destination:

```bash
python3 mlx-model-porting/scripts/install_skill.py \
  --dest ~/.agents/skills \
  --mode symlink
```

Copy mode is a release-artifact operation: run it from the repository checkout
that contains `MANIFEST.json`. It installs only the `mlx-model-porting/`
allowlist from that manifest, verifies every file hash and executable bit,
rejects unlisted source content and unsafe links, and then verifies the staged
tree is an exact identity match. Known local cache/build noise is never copied.
The installed skill includes its own complete Apache-2.0 `LICENSE`.

Symlink mode points the client at the checkout instead of materializing an
attested copy. Both modes refuse unsafe source/destination nesting and are
idempotent for an unchanged target. See
[`adapters/README.md`](adapters/README.md) for client-specific discovery notes.

### Runtime dependencies

| Surface | Requirement |
|---|---|
| Static intake, routing, planning, registries, and validation | Python 3.10+ standard library |
| Tensor parity with `compare_tensors.py` | NumPy |
| Explicit `--allow-network` Hugging Face intake | `huggingface-hub` |
| Model execution | Apple Silicon plus the exact MLX/framework packages required by the chosen port |

Create a dedicated environment for the optional tools rather than relying on
machine-global packages:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --require-hashes \
  -r mlx-model-porting/requirements-tools.lock
```

`requirements-tools.txt` is the small authored input; the universal lock pins
transitive versions and artifact hashes. Refresh it with `uv pip compile
mlx-model-porting/requirements-tools.txt --universal --python-version 3.10
--generate-hashes --output-file mlx-model-porting/requirements-tools.lock`.
CI additionally pins its minimal platform wheels in
`.github/requirements-ci.txt`. The skill installer deliberately does not install
Python packages or execute model code.

## Start from a source model

The default inspection path is offline and static. Network metadata access must
be enabled explicitly.

```bash
python3 mlx-model-porting/scripts/inspect_model.py MODEL \
  --output inspection.json \
  --markdown inspection.md

python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json \
  --output recommendations.json \
  --markdown OPTIMIZATIONS.md

python3 mlx-model-porting/scripts/make_port_plan.py inspection.json \
  --artifact-root MODEL \
  --recommendations recommendations.json \
  --output PORT_PLAN.md
```

Run the recommender only after inspection is clean. If inspection reports any
blocker, generate the port plan without recommendations; it will contain
remediation steps only, and `--family` cannot turn it into an actionable plan.
Every actionable plan requires `--artifact-root`: the planner reruns the static
inspector against those local bytes and rejects any artifact, routing, license,
or safety drift. It also recomputes the complete recommendation report before
embedding advice, so an edited intermediate report is not trusted.

Inspection reports use portable basename-only local references by default.
Pass `--include-local-paths` only when an absolute path is deliberately needed
for local debugging; do not publish that form.

For a local synthetic smoke path that does not download a model:

```bash
python3 mlx-model-porting/scripts/inspect_model.py \
  tests/fixtures/models/decoder \
  --output /tmp/inspection.json

python3 mlx-model-porting/scripts/make_port_plan.py \
  /tmp/inspection.json \
  --artifact-root tests/fixtures/models/decoder \
  --output /tmp/PORT_PLAN.md
```

The fixture is expected to route to `dense-decoder-transformer`. That result
demonstrates the router and guard path only; it does not demonstrate an
end-to-end checkpoint conversion.

## Start from an existing MLX project

Use inspector mode for local MLX code, an already-converted checkpoint, or a
served MLX application. It inventories files and proof surfaces without running
the project:

```bash
python3 mlx-model-porting/scripts/inspect_mlx_project.py PROJECT \
  --model LOCAL_MODEL \
  --output inspection.json \
  --markdown MLX_INSPECTION.md
```

This inspector also emits basename-only local references unless
`--include-local-paths` is explicitly requested for private debugging output.

If an inventory is truncated, the inspector blocks a clean or recommendation-
ready conclusion instead of treating unseen files as safe.

## Validate a port

Weight coverage and tensor parity are explicit steps, not side effects of
loading a checkpoint:

```bash
python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source source.json \
  --target target.json \
  --mapping WEIGHT_MAP.json \
  --output weight-map-report.json

python3 mlx-model-porting/scripts/compare_tensors.py \
  source.npz target.npz \
  --mapping mapping.json \
  --atol 1e-5 \
  --rtol 1e-4 \
  --cosine-min 0.99 \
  --output parity.json
```

Tolerance and quality gates must be chosen for the model and task. Passing a
generic tensor threshold alone is not sufficient for language, vision, audio,
speech, diffusion, or streaming correctness.

## Request optimization advice

Supply a `TargetProfile` when asking for version- or workload-sensitive advice:

```bash
python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json \
  --target-profile target-profile.json \
  --objective peak-memory \
  --output recommendations.json \
  --markdown OPTIMIZATIONS.md
```

For a future `local-promotion` claim, project the profile from the full promoted
fingerprint instead of copying its digest. This example deliberately copies the
canonical objects without reconstructing or normalizing them:

```python
import json

catalog = json.load(open("mlx-model-porting/assets/effective_claims.json"))
claim = next(row for row in catalog["claims"] if row["method_id"] == "METHOD_ID")
fingerprint = claim["experiment_fingerprint"]
payload = fingerprint["payload"]
target = payload["target"]
profile = {
    "schema_version": 1,
    "hardware": target["descriptor"]["hardware"],
    "software": target["descriptor"]["software"],
    "capabilities": [],
    "workloads": ["server"],  # choose the exact controlled workload tag in scope
    "models": payload["models"],
    "target": target,
    "workload": payload["workload"],
    "experiment_fingerprint": fingerprint,
}
print(json.dumps(profile, indent=2))
```

The fingerprint binds the candidate receipt SHA-256, model and source revisions,
target and workload descriptors, enabled methods, metric, experiment contract,
baseline receipt, aggregates, every measured result and raw-output digest, and
the quality result digest. Multiple receipts pool only through a separate stable
identity that retains the exact baseline file/digest, warmup/run/timeout protocol, and
exact-output quality contract; the lowest compatible ratio supplies the public
range and full fingerprint. The advisor also exact-matches the profile's model,
target, workload, hardware, and software objects. A copied digest, empty profile,
or merely similar Mac cannot unlock the number.

The advisor separates results into five controlled buckets:

1. `validated-locally`;
2. `validated-source-theory`;
3. `benchmark-required`;
4. `experimental-approach` — requires explicit user opt-in;
5. `rejected-do-not-use`.

Numeric authority comes from the generated
[`effective_claims.json`](mlx-model-porting/assets/effective_claims.json), not
from prose or an unverified receipt sidecar. All current source-reported ranges
remain withheld observations: the existing `TargetProfile` cannot encode their
complete benchmark scope closely enough to make them profile-eligible.
Source-reported numbers are never advisor-visible until an equivalent local
target-workload benchmark passes the promotion contract.
Historical and newly generated local receipts remain performance observations
because neither runner lane independently attests the executed dependency bytes
and model/workload semantics. Exact argv and quality digests are necessary but
not sufficient to promote a number.
Compound numbers are withheld unless compatible steps were validated together
with unique evidence lineage; multiplying per-step ceilings is never a measured
claim.

## Evidence and benchmark boundaries

[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) is generated from the canonical
[`sources.yaml`](mlx-model-porting/assets/sources.yaml). Review depth means:

- `synthesized`: directly informed a rule, runbook, or registry decision;
- `screened`: relevant material and limits were reviewed;
- `indexed`: catalogued for later review, with no implied support.

Support scope and review depth are separate from benchmark promotion. A pinned
third-party implementation can support an implementation path while still
requiring local parity and workload measurement.

Supported guidance status is also evidence-gated: native methods require
classified official MLX evidence, Apple-project methods require classified
official-project source, and third-party ports require a synthesized pinned or
locally reproduced implementation reference.

The benchmark truth is generated in
[`BENCHMARK_REPORT.md`](mlx-model-porting/assets/BENCHMARK_REPORT.md). The current
11-receipt set contains useful historical measurements, but **zero local speed
or memory claims are promotion-ready**. Do not describe individual ratios as
reliable or portable wins.

Measurement validation has two controlled runner lanes. `mlx-lm-generate`
retains its token-throughput observations. `external-command-wall-time` covers the other
architecture families with direct, no-shell argv: a digest-pinned Python runner
at argv position 1, `models.target.id`, `models.target.revision`, checked-in
workload inputs, and semantic `variant_config` must all be bound through an
exact argv template. The resolved interpreter and sanitized ambient environment
are hashed into the target identity, and the runner starts with `-I -B` to
exclude current-directory, user-site, and environment-injected imports. Static
symlink components are rejected and the quality contract is frozen before
execution. Its only performance metric is
`wall_seconds`, measured by the parent `benchmark_command.py` process and
checked against the complete bounded schema-1 raw report. Every measured run
must recreate the exact label-owned quality output; values printed by the model
command are never accepted as metrics. This generic lane is observation-only:
a digest does not prove that arbitrary runner code used its bound arguments.
The MLX-LM lane is also observation-only until its imported package bytes and
per-run model output are independently bound.

## Release gates

Run these checks from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/validate_benchmarks.py check
python3 mlx-model-porting/scripts/generate_claim_catalog.py --check
python3 mlx-model-porting/scripts/generate_evidence_index.py --check
python3 mlx-model-porting/scripts/generate_site_data.py --check
node --check site/data.js
node --check site/app.js
python3 mlx-model-porting/scripts/manifest.py check
git diff --check
```

These are offline method and artifact gates. They do not replace a real
Apple-Silicon port, task-quality evaluation, or target-workload benchmark.

## Extending the skill

Do not hand-edit generated reports or indexes. Add or change the canonical
registry, runbook, receipt, or version input, regenerate in dependency order,
and review the resulting diff. The ownership table and exact regeneration
commands are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Research automation is review-only: it may collect and rank candidates and
prepare a change, but it may not promote guidance or auto-merge research-derived
recommendations.

## Versioning

The 0.4.0 release snapshot is dated 2026-07-10. The package version is recorded
in [`VERSION`](VERSION) and the skill frontmatter; release changes are recorded
in [`CHANGELOG.md`](CHANGELOG.md).
