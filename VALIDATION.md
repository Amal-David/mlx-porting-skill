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
| An agent routes a model to the correct family **and** runbook | `tests/test_scenarios.py` (17/17 declared families with fixtures) |
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
| Static source-format manifests hold recommendations and report conservative operator/layer coverage for ONNX, GGUF, Flax/Orbax, TensorFlow SavedModel, Keras archive, Core ML package, and safetensors-only checkpoint artifacts | `test_static_inspection_reports_*_source_format_manifest`, `test_static_inspection_reports_*_manifest`, `test_static_inspection_holds_safetensors_only_checkpoint` |
| Source-format reports with static tensor metadata can feed deterministic weight-map shape coverage, while reports without shapes fail loud | `test_weight_mapping_accepts_onnx_source_format_initializers`, `test_weight_mapping_accepts_gguf_source_format_tensors`, `test_weight_mapping_rejects_source_format_without_static_tensors` |
| The deep research loop emits review-only assignments, dynamic assignment-planner receipts, subagent handoff manifests, campaign-wave receipts, campaign-runner receipts, externally ingested result receipts, planned-versus-sampled coverage, review-readiness gates, fixed and adaptive iterative loop receipts, planned multi-source sampling, worker-authored or generated blogs with provenance, synthesis, bounded parallel executor receipts, and rejects malformed findings | `test_research_loop_*`, `test_research_campaign_runner_*` |
| Contributor-scale collection follows `Link` pagination, caps retained counts, stores receipts, and redacts raw `anon=true` identities | `test_contributor_collector_*` |
| The distributed artifact has no silent file drift | `scripts/manifest.py check` |
| Source provenance is well-formed; supported techniques cite implementation evidence | `scripts/validate_sources.py` |

## Requires Apple Silicon + MLX (not proven here)

- Real MLX-vs-PyTorch numeric parity on an actual ported block or model.
- Latency / throughput / peak-memory benchmarks and any speedup number.
- Quantization, KV-cache, and speculative-decoding quality on real weights.
- End-to-end real-model conversion runs for the 17 architecture families; the
  offline scenarios prove routing and guard behavior, not real checkpoint support.
- Tabular, ranking, and recommender support; the offline forecasting scenario
  proves only time-series forecasting routing and guards.
- Dense vision, promptable masks, OCR, depth, and pose support; the offline CV
  backbone scenario proves only image-backbone/classifier routing and guards.
- Point-cloud, equivariant, protein, chemistry, and energy/force scientific
  support; the offline graph scenario proves only GCN-style message-passing
  routing and guards.
- Source-format conversion from ONNX, GGUF, Flax/Orbax, TensorFlow/Keras, or
  Core ML artifacts into executable MLX graphs; offline tests prove static
  metadata triage and recommendation holds only.

## Requires network access (not proven here)

- Source link health: `scripts/validate_sources.py --check-urls`.
- Upstream revision pin-drift detection and the live daily-research pipeline.
- Live execution of external research subagents or network collectors; offline tests cover the harness contract, campaign receipts, campaign runner behavior with deterministic fake researchers, bounded local executor concurrency, iterative receipts, fixture ingestion, and contributor collector receipt behavior only.

## Offline gate commands

```bash
python3 -m unittest discover -s tests -v
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/manifest.py check
```
