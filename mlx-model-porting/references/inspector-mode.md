# Inspector mode for existing MLX projects

Use inspector mode when the user already has local MLX code, a served MLX app, or a converted checkpoint that appears to run. The goal is not to re-port from scratch; it is to inspect what is present, name the proof gaps, and identify whether any local learning is worth contributing back to the runbook.

## Engineering review applied

This mode is shaped by five review lenses:

| Lens | Decision |
|---|---|
| CTO | Keep inspection static-first. Do not import the target project, execute model code, or trust a working sample as correctness. |
| CPO | Support both user states: "I used this porter" and "I already have an MLX port." Both should end in the same evidence vocabulary. |
| Principal engineer | Tie every recommendation to a validation gate: parity fixture, benchmark metadata, quality metric, and rollback condition. |
| Senior engineer | Reuse the existing inspect -> plan -> recommend spine instead of creating a separate magic advisor. |
| Developer/agent | Emit JSON and Markdown reports with concrete next actions that smaller models can follow line by line. |

## Canonical command

```bash
python3 mlx-model-porting/scripts/inspect_mlx_project.py /path/to/mlx-project \
  --model /path/to/local-model \
  --markdown MLX_INSPECTION.md \
  --output inspection.json
```

Use `--inspection existing-inspection.json` when `inspect_model.py` has already been run. Use `--hash-small-files` only when a provenance receipt needs file hashes.

## Report interpretation

The report has four main sections:

- `health`: says whether the project looks good, has proof gaps, needs risk review, or has no detected MLX surface.
- `code_surface`: lists detected MLX packages, modules, feature flags, evidence paths, and high-signal risks.
- `improvement_opportunities`: names missing proofs or likely optimization paths, each with a validation gate.
- `contribution_candidates`: flags local patterns that might teach the corpus something new.

`looks-good` means static evidence found MLX code plus parity and benchmark signals. It does not mean the project is numerically correct. Runtime correctness still requires the parity ladder and task-quality checks.

## Flow

1. Inventory project files without importing the project.
2. Detect MLX packages and code features: raw `mlx.core`, `mlx_lm`, `mlx_vlm`, `mlx_audio`, `mx.eval`, `mx.compile`, fast attention, quantization, cache paths, custom kernels, benchmark evidence, and parity evidence.
3. Fold in model inspections from `inspect_model.py` when model paths or prior inspection reports exist.
4. Classify health and proof gaps.
5. Return improvement opportunities and contribution candidates.

## Contribution gate

Do not promote a local finding directly into a supported runbook. A contribution candidate needs:

- a minimal reproducible fixture;
- a readable MLX fallback when a custom kernel is involved;
- parity or task-quality evidence;
- benchmark metadata with hardware, OS, MLX/package versions, workload, baseline, and precision;
- a rollback condition;
- the exact runbook, asset, or rule that would change.

If those pieces are missing, keep the finding as a review lead or backlog item.
