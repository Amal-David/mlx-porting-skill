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

Returned source objects may include `sampled_target_title` and/or
`sampled_target_locator` to declare which planned sample target the source was
intended to satisfy. The harness validates those explicit receipts against the
assignment sample plan, records valid, missing, and invalid receipt counts in
`sampling_coverage`, and surfaces invalid claims in generated blogs. Use
`--require-explicit-sampling-receipts` when matched planned targets must be
backed by worker-declared target receipts rather than inferred URL/title
matching alone.

Every run also writes a root `subagents.json` dispatch manifest plus per-agent
packets under `agents/*.assignment.json` and `agents/*.prompt.md`. These files
are the stable handoff surface for a human operator or external orchestrator to
spawn one researcher per selected persona. Each packet includes the mission,
sample plan, expected result JSON path, blog path, result contract, and
review-only constraints so workers do not need to scrape `assignments.json`.
If a worker writes markdown to `MLX_RESEARCH_BLOG_PATH`, that file is preserved
as the primary per-agent blog and the harness stores its generated fallback at
`agents/*.generated-blog.md`. If no worker-authored blog exists, the harness
writes the generated blog to the primary blog path and records `source:
generated` in `assignments.json`, `synthesis.json`, `subagents.json`, and the
per-agent packet. Blog receipts record required, present, and missing sections
plus a `contract_status`. Use `--require-worker-blog-contract` for high-stakes
campaigns where an incomplete worker-authored blog should fail after receipts
are written.

Every run also writes `campaign.json` and `campaign.md` at the run root. The
campaign receipt is the orchestrator-facing surface: it lists each wave, the
selected agents, assignment and prompt paths, expected result and blog paths,
safe launch constraints, and the exact per-wave `command_args` for rerunning
ingestion with `--ingest-subagent-results`. External dispatchers should consume
the campaign receipt instead of inferring launch state from internal file names.
For non-final iterative waves, `synthesis.json` and the matching campaign wave
also include a `next_wave_scaffold` receipt. That handoff preserves the
review-gate, sampling, blog-contract, agent-count, and dynamic-assignment flags,
then points the next wave at the current wave's derived `next_gap_hints`. Run
the scaffold command only after the current wave's worker result files have
been ingested, so follow-up assignments are based on observed gaps rather than
stale planned coverage.
When the dispatcher is an explicit local command, use
`scripts/run_research_campaign.py` to execute the campaign receipt wave by wave.
The runner starts one process per listed agent up to `--workers`, passes the
standard `MLX_RESEARCH_*` environment, preserves stdout/stderr/exit-code logs
under `campaign-run-logs/`, invokes the recorded ingest command by default,
and writes `campaign-run.json` plus `campaign-run.md`. Add
`--follow-next-wave-scaffold` when a local campaign should turn post-ingest
`next_wave_scaffold` receipts into generated follow-up waves. The runner
validates that the scaffold command is a local `scripts/research_loop.py`
scaffold, rejects executor or ingest commands inside the scaffold, records each
follow-up in the campaign-run receipt, and stops at `--max-followed-waves` or
when a wave has no next scaffold.

When a loop is intended to feed skill updates, add an explicit review gate:
`--min-sampled-targets`, `--min-non-github-lanes`, and repeated
`--require-source-lane` options record the minimum evidence expected before the
run can be considered ready. `synthesis.json` and `synthesis.md` include the
gate status, checks, and blocked reasons. With `--fail-on-review-gate`, the
harness still writes assignments, blogs, and synthesis receipts, then exits
non-zero if the gate fails. A passing gate only says the research sampled the
requested breadth; it does not promote a technique without the usual source,
validation, test, and rollback evidence.
Add `--require-explicit-sampling-receipts` to fail the review gate when a
sampled planned target lacks an explicit source receipt or when a returned
source claims a target outside that assignment's sample plan.

Each `synthesis.json` also includes `promotion_review`, and `synthesis.md`
renders the same split. This ledger classifies findings as `promotion_ready`,
`validation_backlog`, or `rejected`. It is review-only: a `promotion_ready`
entry means the finding has enough provenance and validation metadata to be
considered for a skill asset or runbook edit, not that the edit has happened.
An `adopted` finding that lacks required next validation or rollback/caveat
metadata is held in `validation_backlog` with explicit blockers.

Each `synthesis.json` also includes an `evidence_matrix` receipt. The matrix
deduplicates returned source locators across agents, links each source back to
finding IDs and persona IDs, counts source-lane citations and finding decisions,
and flags planned source lanes that remained thin or uncited. Use it to review
cross-agent coverage and corroboration context before deciding what to sample
next. Repeated citation is not promotion evidence by itself; promotion still
requires source provenance, a concrete MLX validation gate, tests, and rollback
or caveat metadata.

For `--iterations 2` or higher, the harness writes each pass under
`iterations/NN/` and emits top-level `loop.json` and `loop.md` receipts. Each
iteration records the gap hints it used and the next hints derived from
`held`/`needs-validation` findings. If a review gate fails, the next hints also
include failed checks and unmatched planned sampling targets. The next iteration
receives those hints, which lets the planner switch from deterministic config
order to dynamic coverage-driven assignment selection without rewriting
recommendation assets. Fixed loop receipts aggregate the per-iteration review
gates and remain failed until every iteration passes its own gate. Loop
receipts also aggregate each iteration's `promotion_review` entries and tag
them with iteration, run id, and output directory so a multi-wave campaign has
one final ready/backlog/rejected handoff.

Use `--until-review-gate` when `--iterations` should be a cap rather than a
fixed count. In that adaptive mode, the loop stops as soon as an iteration
passes its review gate; otherwise it records `iteration_cap_exhausted`.
Adaptive loop readiness is based on the final completed iteration, so earlier
failed iterations are treated as review-only sampling passes rather than
permanent failures.

For external multi-wave campaigns, treat each `campaign.json` wave as the unit
of orchestration. Agents within a wave may run in parallel, but dynamic later
waves should be scaffolded after prior wave ingestion when returned findings
need to drive the next gap hints. The top-level campaign receipt marks these
wave dependencies so the dispatcher can avoid launching stale follow-up
assignments. When a wave includes `next_wave_scaffold`, use that command as the
dispatcher input for wave N+1 after the recorded ingest command has succeeded.
Final waves set `next_wave_expected` to false and omit the scaffold receipt.

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
`synthesis.json` includes `blog_contract` counts, and `synthesis.md` annotates
each blog receipt with pass/fail status. Generated fallback blogs are expected
to pass the contract; worker-authored blogs are preserved even when they fail
so reviewers can inspect what the researcher actually wrote.

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

The promotion ledger applies a stricter review surface over returned findings:
`adopted` findings are promotion-ready only when they include source URL/access
date, affected asset/runbook, validation gate, required next validation, and a
rollback or caveat record. `held` and `needs-validation` findings remain
validation backlog. `rejected` findings stay visible but cannot drive skill
updates. Broad sweeps such as top-1000 contributor research should use this
ledger as the handoff from research to implementation, then still update
assets manually with tests, source validation, manifest checks, and rollback
conditions.

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

Collect results from externally spawned subagents:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --agent-count 6 \
  --output-dir research-runs/manual-dispatch

# External orchestrators should read research-runs/manual-dispatch/campaign.json,
# launch the listed wave agents, and wait for every result_path.

# After workers read subagents.json and write agents/*.result.json:
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --agent-count 6 \
  --ingest-subagent-results \
  --output-dir research-runs/manual-dispatch
```

The same command arguments are recorded under
`campaign.json -> waves[].ingest.command_args`, including run id, objective,
selected agent count, assignment mode, gap hints, review-gate requirements, and
the wave output directory.

Run a campaign with an explicit local researcher command:

```bash
python3 scripts/run_research_campaign.py \
  --campaign research-runs/manual-dispatch/campaign.json \
  --agent-command "python3 local_researcher.py" \
  --workers 6 \
  --execution-timeout 300
```

Follow adaptive next-wave scaffolds after each successful ingest:

```bash
python3 scripts/run_research_campaign.py \
  --campaign research-runs/manual-dispatch/iterations/01/campaign.json \
  --agent-command "python3 local_researcher.py" \
  --workers 6 \
  --execution-timeout 300 \
  --follow-next-wave-scaffold \
  --max-followed-waves 4
```

The runner does not fetch network resources by itself and does not execute
remote model code. The explicit researcher command owns any live browsing or
delegation and must still write one result JSON per agent. Use `--dry-run` to
write a launch plan without executing workers, or `--skip-ingest` when a human
operator wants to inspect returned result files before running the recorded
ingest command.

Require evidence breadth before treating a run as review-ready:

```bash
python3 scripts/research_loop.py \
  --objective "Broaden MLX porting evidence beyond GitHub" \
  --offline-fixture tests/fixtures/research_loop/offline_findings.json \
  --min-sampled-targets 2 \
  --min-non-github-lanes 3 \
  --require-source-lane hugging_face \
  --require-explicit-sampling-receipts \
  --fail-on-review-gate \
  --output-dir research-runs/manual-gated
```

Executor mode is opt-in and mutually exclusive with `--offline-fixture`. The
command receives these environment variables:

- `MLX_RESEARCH_PERSONA_ID`;
- `MLX_RESEARCH_ASSIGNMENT_PATH`;
- `MLX_RESEARCH_PROMPT_PATH`;
- `MLX_RESEARCH_RESULT_PATH`;
- `MLX_RESEARCH_BLOG_PATH`;
- `MLX_RESEARCH_RUN_ID`;
- `MLX_RESEARCH_OUTPUT_DIR`;
- `MLX_RESEARCH_REVIEW_ONLY=1`.

The worker must write one JSON object to `MLX_RESEARCH_RESULT_PATH` with
`persona_id`, `decision_notes`, `findings`, and optional `limitations`. It may
also write a markdown blog to `MLX_RESEARCH_BLOG_PATH`. The harness stores each
prompt, result JSON, stdout log, stderr log, exit code, blog provenance, and
execution state under `agents/` in the run directory. A failed worker exits the
loop loudly; inspect the saved receipts before retrying.

The script does not fetch network resources by itself or modify recommendation
assets. In executor mode, the explicit worker command owns any live delegation
or browsing and must still obey the review-only policy.

External ingestion mode is also opt-in and mutually exclusive with
`--offline-fixture` and `--executor-command`. It reads the `agents/*.result.json`
paths declared in the handoff packets, validates every selected persona result,
and fails loudly if any file is missing, malformed, or has the wrong
`persona_id`.

## Promotion Rule

The synthesis report is not a recommendation merge. To promote a finding, edit
the relevant asset or reference separately and run the normal validators:

- `scripts/validate_sources.py`;
- `scripts/audit_skill.py`;
- targeted unit tests;
- manifest check.
