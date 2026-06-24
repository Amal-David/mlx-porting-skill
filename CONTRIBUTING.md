# Contributing

## Evidence classes

Every recommendation must be assigned one of these evidence classes:

- `native-mlx`: implemented in the MLX framework itself.
- `official-mlx-project`: implemented in an official `ml-explore` project.
- `proven-mlx-port`: implemented in an active MLX ecosystem project and covered by tests or benchmarks.
- `research-candidate`: supported by a paper or non-MLX implementation but not yet demonstrated on MLX/Metal.
- `rejected-or-superseded`: measured as ineffective, unsafe, or replaced for the stated workload.

For `proven-mlx-port` entries outside Apple-maintained MLX projects, add `support_scope: third-party-pinned` and cite a pinned repository, source file, or release. That label means "validated in a pinned non-Apple MLX implementation; reproduce locally before recommending for a target port."

## Pull request requirements

A change to `techniques.yaml` must include:

1. source links and pinned revisions where possible;
2. applicable architecture/workload;
3. expected bottleneck addressed;
4. correctness gate;
5. benchmark protocol;
6. rollback rule;
7. review date.

Run `python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting` before opening a PR that changes `sources.yaml` or `techniques.yaml`. A supported technique must cite implementation evidence; a paper-only source can justify only `research-candidate` until an MLX path is reproduced.

A new architecture runbook must include detection signals, state/cache semantics, weight transforms, parity checkpoints, optimization ordering, quantization exclusions, and a minimal test matrix.
