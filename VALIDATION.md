# Validation status

This file states plainly what the skill's tooling **demonstrates offline** versus what
still **requires Apple Silicon or network access**. It exists so readiness is never
overclaimed: passing the offline suite does **not** prove a real model ports correctly
on a Mac — it proves the method, routing, and guards behave as documented.

Corpus / validation review date: **2026-06-27**.

## Demonstrated offline (no Apple Silicon, no model download, no network)

Run with `python3 -m unittest discover -s tests -v` plus the audit, provenance, and
manifest gates. Each item below is enforced by a runnable pass/fail test.

| Claim | Guard |
|---|---|
| An agent routes a model to the correct family **and** runbook | `tests/test_scenarios.py` (14/14 declared families with fixtures) |
| Expected source weight keys are present after inspection | `tests/test_scenarios.py` |
| A **seeded parity bug is caught** by the parity stage | `tests/test_scenarios.py` (compare_tensors must fail) |
| The right optimizations are recommended and wrong ones are **not** | `tests/test_scenarios.py`, `test_tooling.py` |
| KV-cache quantization actually reaches dense/MoE decoders | `test_optimization_guidance_has_no_unreachable_methods`, `test_recommender_rejects_cuda_only_method` |
| A CUDA-only optimization is **rejected**, never recommended | `test_recommender_rejects_cuda_only_method`, `test_port_plan_excludes_rejected_methods` |
| Blocked intake **holds** recommendations until resolved | `test_recommender_holds_candidates_when_intake_blocked` |
| Parity check fails loud on NaN/Inf, cosine drift, and reports report-only honestly | `test_compare_tensors_*` |
| Weight-map shape transforms (reshape/squeeze/unsqueeze/slice/permute) are correct | `test_weight_map_transform_ops_*` |
| The CLI pipeline chains (`inspect → plan → recommend → validate_weight_map`) | `test_pipeline_chains_inspect_to_plan_recommend_and_validate` |
| Fixtures are reproducible and auditable from a generator | `tests/test_fixtures.py` |
| The deep research loop emits review-only assignments, blogs, synthesis, and rejects malformed findings | `test_research_loop_*` |
| The distributed artifact has no silent file drift | `scripts/manifest.py check` |
| Source provenance is well-formed; supported techniques cite implementation evidence | `scripts/validate_sources.py` |

## Requires Apple Silicon + MLX (not proven here)

- Real MLX-vs-PyTorch numeric parity on an actual ported block or model.
- Latency / throughput / peak-memory benchmarks and any speedup number.
- Quantization, KV-cache, and speculative-decoding quality on real weights.
- End-to-end real-model conversion runs for the 14 architecture families; the
  offline scenarios prove routing and guard behavior, not real checkpoint support.

## Requires network access (not proven here)

- Source link health: `scripts/validate_sources.py --check-urls`.
- Upstream revision pin-drift detection and the live daily-research pipeline.
- Live execution of external research subagents or network collectors; offline tests cover the harness contract and fixture ingestion only.

## Offline gate commands

```bash
python3 -m unittest discover -s tests -v
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/manifest.py check
```
