# Research Campaign 2026-07-07-nightly-knowledge-curator-research-loop

Objective: Nightly MLX knowledge curator: top contributors, papers, blogs, package releases, model outcomes, speedup ranges, and app/CLI skill deltas
Review only: True
Waves: 1
Iteration cap: 1
Stop reason: single_iteration

## Orchestration
- Dispatch: one review-only subagent per campaign wave agent
- Parallelism: agents within a wave may run in parallel; dynamic waves should be sequenced through ingestion
- Promotion rule: campaign receipts are review-only and never promote findings without source, validation, tests, and rollback evidence

## Waves
- Wave 1: 2026-07-07-nightly-knowledge-curator-research-loop (6 agents)
  - output: .
  - subagents: subagents.json
  - assignment mode: dynamic
  - gap hints: quantization, adaptive, cache, content, context, inference, long, multimodal
  - next gap hints: https, packages, papers, repositories, github, search, explore, ml, hugging_face, metadata
  - ingest command args: python3 scripts/research_loop.py --run-id 2026-07-07-nightly-knowledge-curator-research-loop --objective 'Nightly MLX knowledge curator: top contributors, papers, blogs, package releases, model outcomes, speedup ranges, and app/CLI skill deltas' --agent-count 6 --assignment-mode dynamic --gap-hint quantization --gap-hint adaptive --gap-hint cache --gap-hint content --gap-hint context --gap-hint inference --gap-hint long --gap-hint multimodal --min-sampled-targets 6 --min-non-github-lanes 4 --require-source-lane papers --require-source-lane repositories --require-source-lane repo_local_audit --ingest-subagent-results --output-dir <repo-root>/mlx-model-porting/research-runs/2026-07-07-nightly-knowledge-curator/research-loop
  - dependency: Single-wave campaign.
  - coverage-skeptic: agents/coverage-skeptic.assignment.json -> agents/coverage-skeptic.result.json
  - package-registry-scout: agents/package-registry-scout.assignment.json -> agents/package-registry-scout.result.json
  - official-docs-cartographer: agents/official-docs-cartographer.assignment.json -> agents/official-docs-cartographer.result.json
  - paper-architecture-scout: agents/paper-architecture-scout.assignment.json -> agents/paper-architecture-scout.result.json
  - huggingface-ecosystem-sampler: agents/huggingface-ecosystem-sampler.assignment.json -> agents/huggingface-ecosystem-sampler.result.json
  - practitioner-blog-miner: agents/practitioner-blog-miner.assignment.json -> agents/practitioner-blog-miner.result.json
