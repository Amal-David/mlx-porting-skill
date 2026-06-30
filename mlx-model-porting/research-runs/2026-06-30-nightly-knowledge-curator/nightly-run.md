# Nightly MLX Knowledge Curator

- Run id: `2026-06-30-nightly-knowledge-curator`
- Started: `2026-06-30T22:06:52+00:00`
- Finished: `2026-06-30T22:07:18+00:00`
- Graph: `/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/knowledge_graph.json`
- Delta: `/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/research-runs/2026-06-30-nightly-knowledge-curator/knowledge-delta.json`
- Research loop: `/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/research-runs/2026-06-30-nightly-knowledge-curator/research-loop`

## Commands

- `/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/scripts/collect_contributors.py --repo ml-explore/mlx --requested-count 1000 --output /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/contributor-refresh.json` -> 0
- `/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/scripts/update_sources.py --output /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/update-candidates.json --fail-on-network-error` -> 0
- `/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/scripts/knowledge_curator.py --run-id 2026-06-30-nightly-knowledge-curator --update-candidates /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/update-candidates.json --previous-graph /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/knowledge_graph.json --graph-output /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/knowledge_graph.json --delta-output /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/research-runs/2026-06-30-nightly-knowledge-curator/knowledge-delta.json --markdown-output /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/research-runs/2026-06-30-nightly-knowledge-curator/knowledge-delta.md` -> 0
- `/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/scripts/research_loop.py --run-id 2026-06-30-nightly-knowledge-curator-research-loop --objective Nightly MLX knowledge curator: top contributors, papers, blogs, package releases, model outcomes, speedup ranges, and app/CLI skill deltas --assignment-mode dynamic --agent-count 6 --min-sampled-targets 6 --min-non-github-lanes 4 --require-source-lane papers --require-source-lane repositories --require-source-lane repo_local_audit --output-dir /Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/research-runs/2026-06-30-nightly-knowledge-curator/research-loop` -> 0

## Gap Hints

None.

## Policy

- Review-only: candidate evidence may update the graph and research receipts.
- Do not auto-promote skill/app/CLI guidance without source provenance, validation gate, rollback condition, and tests.
