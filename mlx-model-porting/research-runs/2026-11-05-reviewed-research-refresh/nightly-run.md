# Nightly MLX Knowledge Curator

- Run id: `2026-11-05-reviewed-research-refresh`
- Started: `2026-11-05T06:18:31+00:00`
- Finished: `2026-11-05T06:18:31+00:00`
- Graph: `mlx-model-porting/assets/knowledge_graph.json`
- Delta: `mlx-model-porting/research-runs/2026-11-05-reviewed-research-refresh/knowledge-delta.json`
- Research loop: ``

## Commands

- `python3 mlx-model-porting/scripts/knowledge_curator.py --run-id 2026-11-05-reviewed-research-refresh --update-candidates mlx-model-porting/assets/update-candidates.json --contributor-refresh mlx-model-porting/assets/contributor-refresh.json --research-backlog mlx-model-porting/assets/research_backlog.json --previous-graph mlx-model-porting/assets/knowledge_graph.json --graph-output mlx-model-porting/assets/knowledge_graph.json --delta-output mlx-model-porting/research-runs/2026-11-05-reviewed-research-refresh/knowledge-delta.json --markdown-output mlx-model-porting/research-runs/2026-11-05-reviewed-research-refresh/knowledge-delta.md` -> 0
- `python3 mlx-model-porting/scripts/knowledge_curator.py --reconcile-backlog --previous-graph mlx-model-porting/assets/knowledge_graph.json --update-candidates mlx-model-porting/assets/update-candidates.json --research-backlog mlx-model-porting/assets/research_backlog.json` -> 0

## Gap Hints

`cache`, `audio`, `native`, `quantization`, `streaming`, `spectral`, `transforms`, `weight`, `adaptive`, `low`, `route`, `tts`

## Policy

- Review-only: candidate evidence may update the graph and research receipts.
- Do not auto-promote skill/app/CLI guidance without source provenance, validation gate, rollback condition, and tests.
