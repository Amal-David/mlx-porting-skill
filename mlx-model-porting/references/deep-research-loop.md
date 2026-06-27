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

## Contributor-Scale Sweeps

When researching top contributors, record the requested count and the actual
API-returned count. For example, the top-1000 MLX contributor sweep requested
`ml-explore/mlx` contributors pages 1-10 and GitHub returned 256 linked
contributors, or 262 author buckets with `anon=true`, at retrieval time. Follow
GitHub `Link` headers rather than assuming all capped pages exist, and store
status codes, ETags, Last-Modified headers, rate-limit state, the login set,
matched repositories, inspected snapshots, query errors, and promoted/held
decisions in `assets/contributor_learnings.json`.

Use the collector when refreshing the contributor source set:

```bash
python3 scripts/collect_contributors.py \
  --repo ml-explore/mlx \
  --requested-count 1000 \
  --output assets/contributor-refresh.json
```

The collector is still only a source-selection tool: it follows `Link` headers,
stores page receipts, and records `anon=true` aggregate counts, but it does not
screen repositories or promote implementation guidance.

Contributor-owned repositories are source-selection evidence only. Promote a
learning only when a pinned source path, tests or local validation, affected
skill surface, validation gate, and rollback condition are recorded. If code or
repository search is rate-limited, incomplete, or capped, add a backlog item
rather than treating the long tail as complete. Do not persist raw anonymous
author identities.

## Agent Assignments

The harness generates bounded assignments from
`assets/research_loop_config.json`. By default it keeps deterministic config
order for repeatability. Pass gap hints or `--assignment-mode dynamic` when the
loop should choose the worker roster from the current objective, source-lane
keywords, and known coverage gaps. A live operator can hand those assignments to
subagents. Tests and offline automation use fixture findings instead of live
agents or network.

Each assignment must tell the researcher:

- which source lanes to sample;
- the planned sample targets, lane title, and evidence role for each lane;
- what adoption bar applies;
- that remote model code must not be executed;
- that all findings need URLs, access dates, validation gates, and affected
  skill surfaces;
- that community evidence can only create leads.

The generated `assignments.json` includes `sample_plan` objects for every
persona plus an `assignment_planner` receipt that records selected and held
personas, source-lane counts, matched objective or gap terms, and selection
reasons. `assignments.json` also includes `sampling_coverage` for every
assignment: planned targets, matched targets, unmatched planned targets, and
unplanned returned sources. `synthesis.json` repeats the planner receipt and
reports planned source-lane counts, planned sample-target counts, planned
non-GitHub sample targets, and aggregate sampling coverage separately from
returned findings. This proves both intended breadth and actual sampled
coverage, even when a run is only scaffolded and no agent has returned findings
yet.

When a loop is intended to feed skill updates, add an explicit review gate:
`--min-sampled-targets`, `--min-non-github-lanes`, and repeated
`--require-source-lane` options record the minimum evidence expected before the
run can be considered ready. `synthesis.json` and `synthesis.md` include the
gate status, checks, and blocked reasons. With `--fail-on-review-gate`, the
harness still writes assignments, blogs, and synthesis receipts, then exits
non-zero if the gate fails. A passing gate only says the research sampled the
requested breadth; it does not promote a technique without the usual source,
validation, test, and rollback evidence.

For `--iterations 2` or higher, the harness writes each pass under
`iterations/NN/` and emits top-level `loop.json` and `loop.md` receipts. Each
iteration records the gap hints it used and the next hints derived from
`held`/`needs-validation` findings. If a review gate fails, the next hints also
include failed checks and unmatched planned sampling targets. The next iteration
receives those hints, which lets the planner switch from deterministic config
order to dynamic coverage-driven assignment selection without rewriting
recommendation assets. Fixed loop receipts aggregate the per-iteration review
gates and remain failed until every iteration passes its own gate.

Use `--until-review-gate` when `--iterations` should be a cap rather than a
fixed count. In that adaptive mode, the loop stops as soon as an iteration
passes its review gate; otherwise it records `iteration_cap_exhausted`.
Adaptive loop readiness is based on the final completed iteration, so earlier
failed iterations are treated as review-only sampling passes rather than
permanent failures.

## Blog Contract

Each agent writes a markdown blog with:

- assignment summary;
- planned sampling targets;
- sources sampled;
- sampling coverage, including missed planned targets and substitutions;
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
review-only synthesis that says no findings were ingested while still recording
the planned source-lane and sample-target coverage.

Run an explicit local worker command for each assignment:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --agent-count 6 \
  --executor-workers 3 \
  --executor-command "python3 local_researcher.py" \
  --output-dir research-runs/manual-executor
```

Use `--executor-workers N` to run up to `N` local worker processes at once.
Each assignment still gets its own prompt, result JSON, stdout, stderr, exit
code, worker-count receipt, and execution state. Output order remains
assignment-order deterministic even when workers finish out of order.

Choose agents dynamically for known gaps:

```bash
python3 scripts/research_loop.py \
  --objective "Investigate ranking, recommender, and package-release gaps" \
  --agent-count 3 \
  --gap-hint ranking \
  --gap-hint recommender \
  --gap-hint package \
  --output-dir research-runs/manual-dynamic
```

Dynamic planning is deterministic: score selected personas from matched
objective terms and higher-weighted gap terms, keep config order for ties, and
write selected/held receipts before any executor runs.

Run multiple review-only iterations:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --iterations 2 \
  --offline-fixture tests/fixtures/research_loop/offline_findings.json \
  --output-dir research-runs/manual-iterative
```

For a live local worker command, use `--iterations` with `--executor-command`.
The first pass may run in config order; later passes inherit derived gap hints
from prior findings and can dynamically reshape the worker roster.

Stop a bounded campaign once the review gate passes:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --iterations 5 \
  --until-review-gate \
  --executor-workers 3 \
  --executor-command "python3 local_researcher.py" \
  --min-sampled-targets 8 \
  --min-non-github-lanes 4 \
  --output-dir research-runs/manual-adaptive
```

Require evidence breadth before treating a run as review-ready:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --offline-fixture tests/fixtures/research_loop/offline_findings.json \
  --min-sampled-targets 2 \
  --min-non-github-lanes 3 \
  --require-source-lane hugging_face \
  --fail-on-review-gate \
  --output-dir research-runs/manual-gated
```

Executor mode is opt-in and mutually exclusive with `--offline-fixture`. The
command receives these environment variables:

- `MLX_RESEARCH_PERSONA_ID`;
- `MLX_RESEARCH_PROMPT_PATH`;
- `MLX_RESEARCH_RESULT_PATH`;
- `MLX_RESEARCH_RUN_ID`;
- `MLX_RESEARCH_OUTPUT_DIR`;
- `MLX_RESEARCH_REVIEW_ONLY=1`.

The worker must write one JSON object to `MLX_RESEARCH_RESULT_PATH` with
`persona_id`, `decision_notes`, `findings`, and optional `limitations`. The
harness stores each prompt, result JSON, stdout log, stderr log, exit code, and
execution state under `agents/` in the run directory. A failed worker exits the
loop loudly; inspect the saved receipts before retrying.

The script does not fetch network resources by itself or modify recommendation
assets. In executor mode, the explicit worker command owns any live delegation
or browsing and must still obey the review-only policy.

## Promotion Rule

The synthesis report is not a recommendation merge. To promote a finding, edit
the relevant asset or reference separately and run the normal validators:

- `scripts/validate_sources.py`;
- `scripts/audit_skill.py`;
- targeted unit tests;
- manifest check.
