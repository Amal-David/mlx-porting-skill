# Contributing

Treat this repository as an executable engineering standard. A contribution is
complete only when its source, scope, validation gate, rollback, generated
views, and tests agree.

## Where to start

Good first contributions are small, verifiable, and grounded in the real backlog:

- **Classify an evidence source.** Pick one of the ~328 sources that remain
  intentionally unclassified in
  [`mlx-model-porting/assets/sources.yaml`](mlx-model-porting/assets/sources.yaml),
  add an honest support scope and claim type, then regenerate
  [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md).
- **Extend a family runbook** in
  [`mlx-model-porting/references/`](mlx-model-porting/references/) with a
  verified, pinned reference for a technique or failure mode it currently omits.
- **Add a golden-scenario fixture variant** to
  [`tests/test_scenarios.py`](tests/test_scenarios.py) so a routing or
  weight-coverage edge case is covered without loosening an existing gate.
- **Run a worked example** from
  [`mlx-model-porting/examples/`](mlx-model-porting/examples/) end to end and
  report any divergence from its checked-in parity report.

Each keeps the source, scope, and gate in agreement — the standard the rest of
this document formalizes.

## Evidence vocabulary

Recommendation status is controlled:

- `native-mlx`: implemented in the Apple-maintained MLX framework;
- `official-mlx-project`: implemented in an Apple-maintained project built on
  MLX;
- `proven-mlx-port`: implemented in a pinned MLX ecosystem project with
  relevant implementation evidence;
- `research-candidate`: supported by a paper, non-MLX implementation, or
  unpromoted experiment, but not established as supported MLX guidance;
- `rejected-or-superseded`: ineffective, unsafe, incompatible, contradicted,
  or replaced for the stated scope.

Support scope is separate from status and uses only these controlled values:

- `context_only`;
- `local_reproduced`;
- `official_mlx`;
- `official_mlx_project`;
- `paper_only`;
- `third_party_pinned`.

Use `third_party_pinned` for a non-Apple MLX implementation and cite an
exact commit-addressed repository or source file, or a release artifact with a
recorded digest. A movable branch or tag is not an immutable pin. The scope
means reproducible prior art, not an official framework guarantee or a
target-model performance claim.

Review depth is also independent:

- `synthesized`: directly informed a rule, runbook, or registry decision;
- `screened`: relevant material and limits were reviewed;
- `indexed`: catalogue-only evidence awaiting deeper review.

Never promote `indexed` to `synthesized` without identifying the rule, runbook,
or registry decision the source informed.

## Architecture extension flow

A new family or material routing change must include:

1. exact detection signals and route policy in
   `mlx-model-porting/assets/architectures.yaml`;
2. a runbook covering state/cache semantics, weight transforms, parity
   checkpoints, optimization ordering, quantization exclusions, failure modes,
   and a minimal validation matrix;
3. explicit hybrid components when one family contract is insufficient;
4. a synthetic scenario that verifies the route, expected weight coverage, a
   seeded parity failure, and correct optimization inclusion/exclusion; and
5. ambiguity behavior showing weak or tied evidence fails closed.

A route fixture extends the decision system. Do not describe it as a completed
real-model port without a separate source oracle, MLX implementation, weight
map, parity results, and task-quality evidence.

## Technique and guidance extension flow

A technique change must identify:

1. primary or pinned source evidence;
2. exact architecture, capability, workload, objective, and version scope;
3. bottleneck or correctness problem addressed;
4. implementation status and support scope;
5. correctness and task-quality gates;
6. benchmark protocol and target metric;
7. tradeoffs and explicit rollback condition; and
8. review date.

Add broad decision records to `assets/techniques.yaml`. Add an actionable method
to `assets/optimization_guidance.yaml` only when its matching constraints,
validation gate, evidence lineage, and rollback are complete. Stack pairs not
explicitly validated together remain `unknown`; do not infer composability or
multiply per-method ceilings.

An experimental contributor, blog, package, paper, or repository approach must
stay in the `experimental-approach` advisor bucket and require the prompt:
“This is an experimental approach. Do you want to try it?”

### Learning and optimization atlas extension flow

The public study guide is a generated consumer of the same decision system, not
a second recommendation registry. When a technique becomes a teachable method:

1. register the technique and reviewed evidence, then add complete applicability,
   status, trade-off, validation, rollback, and evidence references to
   `assets/optimization_guidance.yaml`;
2. assign the method to exactly one `optimization_families` branch in
   `assets/learning_paths.json` and only to representative journeys where it is
   useful teaching context—journey membership is not model support;
3. add method-level entries to `method_quality_gates` for every lossy or
   conditionally-lossy path, including any method marked that way in
   `optimization_stacks.yaml`;
4. keep numeric outcomes exclusively in generated `effective_claims.json`;
   screened, indexed, withheld, rejected, or source-reported ranges must not be
   copied into learning data or public method guidance;
5. run `generate_site_data.py`, then the learning, optimization-taxonomy,
   evidence-boundary, interaction, no-JavaScript, responsive, and export-link
   tests; and
6. regenerate `MANIFEST.json` only after every authored and generated file is
   final.

The site generator rejects methods without reviewed canonical evidence and
rejects `indexed` sources from public method links. Research candidates retain
their explicit user-opt-in advisor policy; rejected or superseded methods are
never selectable. A local-promotion number must retain its metric, exact target
constraints, experiment fingerprint, and effective range as one scoped record.

## Benchmark extension flow

Use `mlx-model-porting/scripts/benchmark_generation.py` only for legacy
MLX-LM observation receipts. For a family-neutral schema-2 receipt, use
`benchmark_command.py --receipt-spec ... --quality-contract ...`; for a
controlled MLX-LM schema-2 candidate, follow the receipt contract in
`references/benchmarking.md`. A candidate intended for future promotion needs:

- immutable target and source model identity and lineage;
- checked-in workload inputs and canonical workload parameters;
- either a controlled `python -m mlx_lm generate` invocation, or the
  `external-command-wall-time` adapter with an exact literal/source argv
  template that executes a digest-pinned Python runner at argv position 1 and
  binds `models.target.id`, `models.target.revision`, checked-in workload
  evidence, a semantic variant, and a label-owned quality-output path. The
  resolved interpreter and sanitized ambient environment are hashed into the
  target identity, and Python starts with `-I -B` so current-directory,
  user-site, and environment-injected imports cannot alter the run;
- digest-bound raw outputs;
- a schema-2 controlled quality contract. Today the only controlled built-in
  metric is exact-output parity between a digest-bound reference and the exact
  candidate output recreated by every measured run;
- an exactly compatible root-level baseline referenced by its canonical file
  name and digest. Candidate generation rejects nested, unsafe, or incomplete
  baselines before starting the measured command;
- at least the validator's required run count and stability threshold, plus an
  explicit finite timeout no greater than 3600 seconds for external receipts;
- enabled method IDs and primary metric; and
- an explicit rollback condition; and
- an external signature over the repository commit/tree, challenge, reviewed
  dependency manifest, raw output, promotion policy, and timing, produced by a
  protected Apple-Silicon signer and verified against a maintainer-controlled
  trust anchor outside the submitted receipt/evidence and this repository. The
  generic lanes do not satisfy this gate. The repository-owned
  `attested-mlx-port-wall-time` lane preserves internally consistent evidence
  for reproduction, but its unkeyed SHA-256 digests are not signatures and it
  remains observation-only until that external signer exists.

Legacy Python evaluators, schema-1 arbitrary JSON scores, or handwritten quality
artifacts can be retained as observation provenance, but the validator does not
execute or trust them and they cannot make a receipt promotion-ready. Add a new
task metric only as a controlled built-in schema that independently recomputes
its result from bounded checked-in artifacts.

A promotion-ready assessment emits one receipt-specific canonical experiment
fingerprint. It binds exact model/source revisions, target
descriptor, workload artifacts and parameters, enabled methods, primary metric,
experiment contract, baseline digest, raw measurements, and quality result.
Compatible repetitions may group only when their stable semantic identity also
matches: the exact baseline file/digest, warmup/run/timeout protocol, and exact-output
quality contract must be identical. Claim generation withholds heterogeneous
identities and uses the conservative minimum repetition together with that
repetition's full fingerprint. Do not replace
this receipt-derived identity with hand-authored target constraints: consumers
must match the full fingerprint and its exact receipt-derived model, target,
workload, hardware, and software descriptors, not only the fingerprint digest.

After adding or changing a receipt, regenerate the benchmark assessment. A
missing or failed gate must remain visible as `performance_observation` or
`rejected`; never replace the generated classification with prose.

## Execution-tool test expectations

Changes to `capture_oracle.py`, `scaffold_port.py`, `convert_checkpoint.py`,
`capture_mlx.py`, `run_parity.py`, or `_capture_common.py` must preserve two
test layers:

- the base contract runs without importing Torch, Transformers, or MLX and
  covers CLI validation, bounded parsing, deterministic manifests/output,
  unsupported-config blockers, map coverage, scaffold drift, and
  first-divergence behavior. Conversion may use the repository's required
  NumPy dependency, but cannot require MLX for draft, validation, or fallback
  safetensors tests;
- real execution tests use `unittest.skipUnless` with an explicit dependency
  probe. Torch/Transformers gates source-model capture; MLX gates generated
  package execution and native safetensors loading; the full parity test runs
  only when both stacks are usable. A missing optional runtime must produce an
  honest skip, while missing required dependencies in an invoked CLI must fail
  with an actionable message.

Every new supported dense-decoder semantic needs a contract that would fail if
its generated computation, target tensor set, conversion rule, or parity rung
regressed. A runbook-only family must not gain an execution test that implies a
shipped architecture module.

## Generated-file ownership

Do not hand-edit generated artifacts.

| Authored input | Generated output | Owner command |
|---|---|---|
| Explicit live official-release and paper queries | `mlx-model-porting/assets/update-candidates.json` | `python3 mlx-model-porting/scripts/update_sources.py --output mlx-model-porting/assets/update-candidates.json` |
| Explicit GitHub contributor query parameters | `mlx-model-porting/assets/contributor-refresh.json` | `python3 mlx-model-porting/scripts/collect_contributors.py --repo ml-explore/mlx --requested-count 1000 --output mlx-model-porting/assets/contributor-refresh.json` |
| Reviewed source/contributor candidates plus canonical registries | `mlx-model-porting/assets/knowledge_graph.json` and a new dated nightly delta/run directory | `python3 mlx-model-porting/scripts/nightly_knowledge_curator.py` |
| Current `knowledge_graph.json` plus `update-candidates.json` | `mlx-model-porting/assets/research_backlog.json` | `python3 mlx-model-porting/scripts/knowledge_curator.py --reconcile-backlog` (verify with `--check-backlog`) |
| `mlx-model-porting/requirements-tools.txt` | `mlx-model-porting/requirements-tools.lock` | `uv pip compile mlx-model-porting/requirements-tools.txt --universal --python-version 3.10 --generate-hashes --output-file mlx-model-porting/requirements-tools.lock` |
| `mlx-model-porting/assets/sources.yaml` | `EVIDENCE_INDEX.md` | `python3 mlx-model-porting/scripts/generate_evidence_index.py` |
| Receipt JSON under `mlx-model-porting/assets/benchmarks/` | `receipt_assessments.json`, `receipts_index.json`, and `mlx-model-porting/assets/BENCHMARK_REPORT.md` | `python3 mlx-model-porting/scripts/validate_benchmarks.py generate` |
| `optimization_guidance.yaml`, `sources.yaml`, and generated receipt assessments | `mlx-model-porting/assets/effective_claims.json` | `python3 mlx-model-porting/scripts/generate_claim_catalog.py` |
| `VERSION`, canonical registries, `references/*.md`, generated assessments, and effective claims | `site/data.js` | `python3 mlx-model-porting/scripts/generate_site_data.py` |
| Final distributed repository tree | `MANIFEST.json` | `python3 mlx-model-porting/scripts/manifest.py generate` |

The first three rows are explicit network collectors, not deterministic release
generators. Review their receipts and errors before committing them. Never
overwrite a historical `research-runs/<date-or-run-id>/` directory for a new
collector result; a new execution requires a new run ID/date. The sole exception
is an owner-reviewed, mechanically verifiable security or privacy redaction that
changes no finding, claim, source, timestamp, or outcome. Record every affected
file and the exact transformation in the release report and changelog. Outside
that exception, old nightly/campaign receipts remain immutable evidence. The
deterministic dependency order below starts only after any intended collector
refresh is reviewed.

Regenerate in dependency order:

```bash
python3 mlx-model-porting/scripts/knowledge_curator.py --reconcile-backlog
python3 mlx-model-porting/scripts/knowledge_curator.py --check-backlog
python3 mlx-model-porting/scripts/validate_benchmarks.py generate
python3 mlx-model-porting/scripts/generate_claim_catalog.py
python3 mlx-model-porting/scripts/generate_evidence_index.py
python3 mlx-model-porting/scripts/generate_site_data.py
python3 mlx-model-porting/scripts/manifest.py generate
```

Run only the generators whose authored inputs changed, but always regenerate
`site/data.js` after a version, corpus-count, benchmark-assessment, effective-
claim change, reference addition/removal, or architecture-runbook heading
change. Generate `MANIFEST.json` last, after every distributed file is final.

## Installation and license boundary

`mlx-model-porting/` is the installable payload. Keep the complete Apache-2.0
text at `mlx-model-porting/LICENSE`; the repository-root license alone does not
satisfy a copied skill distribution. Copy mode reads the repository-root
`MANIFEST.json`, verifies the exact `mlx-model-porting/` allowlist and source
hashes, and stages only those records. Aside from explicitly excluded local
cache/build noise, an unlisted source entry must fail the install rather than
silently enter the copied skill. Generate the root manifest only after the
payload and installer tests are final.

## Pull request requirements

Keep the change reviewable and include:

- the canonical authored inputs;
- generated diffs produced from those inputs;
- tests that would fail if the intended rule regressed;
- provenance and immutable revisions where required;
- the correctness and quality gate;
- benchmark metadata for any optimization change; and
- a rollback condition.

Research automation may collect candidates and prepare a review change. It may
not auto-merge, upgrade review depth, promote guidance, or rewrite generated
claim state without review.

## Release checks

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

The offline suite proves repository contracts, not a real-model conversion or a
portable speedup. If a required Apple-Silicon run, network check, quality
evaluation, or publication review was not performed, say so explicitly in the
pull request.

## Release and deployment automation

Two GitHub Actions workflows automate release side effects on `main`:

- `.github/workflows/deploy-site.yml` publishes the static site to Cloudflare
  Pages when a push to `main` touches `site/` or `VERSION`. It requires the repo
  secrets `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.
- `.github/workflows/release.yml` cuts the git tag and GitHub Release
  automatically when `VERSION` changes on `main`.

Both are side effects of an already-merged, already-validated change: they
publish what the release checks above have proven and relax no gate.
