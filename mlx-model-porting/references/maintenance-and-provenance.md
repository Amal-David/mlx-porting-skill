# Maintenance, provenance, and daily updates

## Threat model

A skill can execute scripts and influence an agent’s engineering decisions. Treat external skills, repositories, releases, model code, and generated update content as untrusted until reviewed.

Risks include:

- malicious or compromised scripts;
- abandoned repository takeover;
- prompt injection in model cards/issues/docs;
- dependency confusion or install hooks;
- unsafe pickle/model loading;
- silently changed tags or unpinned default branches;
- performance claims copied without reproducible context;
- recommendations that become wrong after MLX API changes.

## Source provenance

Each source entry records URL, source type, owner, revision/date when known, review depth, and affected topics. Prefer immutable commit or paper identifiers.

Review priority:

1. official MLX source and release notes;
2. official MLX-LM/examples source;
3. active MLX-VLM/MLX-Audio source and tests;
4. active third-party MLX ports with reproducible code;
5. primary papers and official architecture repositories;
6. technical blogs only for implementation context, never as the sole correctness authority.

## Daily update workflow

The scheduled job:

1. checks allowlisted repository releases/heads;
2. queries recent primary-paper metadata using keyword groups;
3. writes candidate changes to `assets/update-candidates.json`;
4. runs the skill audit and tests;
5. creates a review branch/PR only when configured;
6. never edits technique status or runbooks automatically.

Use:

```bash
python3 scripts/update_sources.py --output assets/update-candidates.json
```

Network access must be explicit. Tokens are read from environment and never written to artifacts.

Before distribution, run the structural audit and provenance validator:

```bash
python3 scripts/audit_skill.py --strict .
python3 scripts/validate_sources.py .
```

Use `validate_sources.py --check-urls` only when network access is explicitly allowed. It checks HTTPS reachability and evidence wiring; it never imports model code or executes repository content.

## Golden scenario gate

`tests.test_scenarios` must cover every family in `assets/architectures.yaml`. When adding or renaming an architecture family, update `tests/fixtures/generate_fixtures.py`, add a tiny scenario under `tests/fixtures/scenarios/`, and keep the full-family scorecard at full marks for routing, runbook selection, weight-key coverage, seeded parity failure, and optimization inclusion/exclusion. Synthetic scenarios prove skill routing only; they do not prove real-model conversion support.

## Deep research loop

Use `scripts/research_loop.py` for broad ecosystem learning that should sample beyond GitHub. The loop generates researcher assignments, accepts returned findings, writes per-agent blogs, and produces a review-only synthesis. By default it does not fetch network resources, spawn subagents, or modify recommendation assets. Executor mode can run an explicit local worker command for each assignment, but it must preserve prompts, stdout, stderr, result JSON, exit code, and execution state as review receipts.

Every deep loop should record which personas were only scaffolded and which returned findings were ingested. Treat package registries, Hugging Face model cards, technical blogs, papers, and community discussions as distinct evidence classes with different promotion bars. Package and model-card metadata can justify prioritization and linting rules, but not benchmark or implementation support by itself.

## Candidate promotion

A candidate moves from `indexed` to `screened` after relevance and source integrity review. It moves to `synthesized` only after a rule/runbook is updated with:

- applicability;
- MLX implementation status;
- expected bottleneck;
- correctness and quality gate;
- benchmark protocol;
- rollback condition.

Techniques marked `native-mlx`, `official-mlx-project`, or `proven-mlx-port` must cite at least one implementation source such as official docs, a repository path, source file, or release note. Papers alone can justify only `research-candidate` until a reproducible MLX path and validation gate exist.

For non-Apple MLX implementations, add `support_scope: third-party-pinned`. That scope means the technique is useful prior art in a pinned MLX ecosystem implementation, not a framework guarantee; reproduce it locally before recommending it for a port.

## Deprecation

Do not delete historical decisions. Mark superseded entries with replacement, date, and reason. This preserves why an agent should not resurrect an old recommendation.

## Release cadence

Use semantic versions for the skill. Architecture detection or behavior changes require at least a minor version. Source additions with no recommendation change may be patch releases.
