# Nightly MLX Knowledge Curator

- Run id: `2026-07-07-nightly-knowledge-curator`
- Started: `2026-07-07T22:02:39+00:00`
- Finished: `2026-07-07T22:03:01+00:00`
- Graph: `mlx-model-porting/assets/knowledge_graph.json`
- Delta: `mlx-model-porting/research-runs/2026-07-07-nightly-knowledge-curator/knowledge-delta.json`
- Research loop: `mlx-model-porting/research-runs/2026-07-07-nightly-knowledge-curator/research-loop`

## Commands

- `python3 mlx-model-porting/scripts/collect_contributors.py --repo ml-explore/mlx --requested-count 1000 --output mlx-model-porting/assets/contributor-refresh.json` -> 0
- `python3 mlx-model-porting/scripts/update_sources.py --output mlx-model-porting/assets/update-candidates.json --fail-on-network-error` -> 0
- `python3 mlx-model-porting/scripts/knowledge_curator.py --run-id 2026-07-07-nightly-knowledge-curator --update-candidates mlx-model-porting/assets/update-candidates.json --previous-graph mlx-model-porting/assets/knowledge_graph.json --graph-output mlx-model-porting/assets/knowledge_graph.json --delta-output mlx-model-porting/research-runs/2026-07-07-nightly-knowledge-curator/knowledge-delta.json --markdown-output mlx-model-porting/research-runs/2026-07-07-nightly-knowledge-curator/knowledge-delta.md` -> 0
- `python3 mlx-model-porting/scripts/research_loop.py --run-id 2026-07-07-nightly-knowledge-curator-research-loop --objective Nightly MLX knowledge curator: top contributors, papers, blogs, package releases, model outcomes, speedup ranges, and app/CLI skill deltas --assignment-mode dynamic --agent-count 6 --min-sampled-targets 6 --min-non-github-lanes 4 --require-source-lane papers --require-source-lane repositories --require-source-lane repo_local_audit --output-dir mlx-model-porting/research-runs/2026-07-07-nightly-knowledge-curator/research-loop --gap-hint quantization --gap-hint adaptive --gap-hint cache --gap-hint content --gap-hint context --gap-hint inference --gap-hint long --gap-hint multimodal` -> 0

## Gap Hints

`quantization`, `adaptive`, `cache`, `content`, `context`, `inference`, `long`, `multimodal`

## Policy

- Review-only: candidate evidence may update the graph and research receipts.
- Do not auto-promote skill/app/CLI guidance without source provenance, validation gate, rollback condition, and tests.
