# Deep Research Loop

Use this loop when the skill needs broad MLX ecosystem learning rather than a
single model port. The loop is review-only: it collects and synthesizes
candidate evidence, but it does not promote a technique or rewrite a runbook
without a concrete validation gate.

## Source Lanes

The loop samples beyond GitHub:

- official docs: MLX and Apple APIs, runtime behavior, packaging, and support boundaries;
- papers: architecture semantics, metrics, and algorithmic constraints;
- Hugging Face: model cards, library tags, task demand, licenses, and artifact shapes;
- packages: release cadence, install surfaces, optional dependencies, and version drift;
- technical blogs: implementation details and benchmark caveats;
- community discussions: pain points and leads only;
- repositories: source implementation, tests, conversion scripts, and benchmarks.
- repo-local audit: gaps in existing runbooks, validators, fixtures, and tests.

GitHub can be useful source evidence, but it must not be the only discovery
axis for comprehensive skill maintenance.

## Agent Assignments

The harness generates bounded assignments from
`assets/research_loop_config.json`. A live operator can hand those assignments
to subagents. Tests and offline automation use fixture findings instead of live
agents or network.

Each assignment must tell the researcher:

- which source lanes to sample;
- what adoption bar applies;
- that remote model code must not be executed;
- that all findings need URLs, access dates, validation gates, and affected
  skill surfaces;
- that community evidence can only create leads.

## Blog Contract

Each agent writes a markdown blog with:

- assignment summary;
- sources sampled;
- candidate findings;
- decision notes;
- open validation.

The blog is an audit artifact, not marketing copy. It should show what was
sampled, what was learned, what was held, and what remains uncertain.

## Decision Model

Findings use one of four states:

- `adopted`: strong enough to change a concrete skill asset after validation;
- `held`: useful lead, insufficient for supported guidance;
- `rejected`: not relevant, unsafe, contradicted, or incompatible;
- `needs-validation`: likely important, but requires a local MLX path, parity
  test, benchmark, or license review.

An adopted finding must have source provenance, review depth, validation gate,
rollback or hold condition, and an affected asset/runbook. If any of those are
missing, keep it `needs-validation` or `held`.

## CLI

Generate assignments and synthesize fixture-backed findings:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --offline-fixture tests/fixtures/research_loop/offline_findings.json \
  --output-dir research-runs/manual
```

Without `--offline-fixture`, the script writes assignments and an empty
review-only synthesis that says no findings were ingested. It does not fetch
network resources or spawn subagents by itself; the operator or host agent owns
live delegation and then feeds returned findings back into the harness.

## Promotion Rule

The synthesis report is not a recommendation merge. To promote a finding, edit
the relevant asset or reference separately and run the normal validators:

- `scripts/validate_sources.py`;
- `scripts/audit_skill.py`;
- targeted unit tests;
- manifest check.
