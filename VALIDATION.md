# Validation status

This file states plainly what the skill's tooling **demonstrates offline** versus what
still **requires Apple Silicon or network access**. It exists so readiness is never
overclaimed: passing the offline suite does **not** prove a real model ports correctly
on a Mac — it proves the method, routing, and guards behave as documented.

Corpus review date: **2026-06-27**. Receipt / validation review date: **2026-07-06**.

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
| The deep research loop emits review-only assignments, dynamic assignment-planner receipts, subagent handoff manifests, campaign-wave receipts, adaptive next-wave scaffold handoffs, campaign-runner receipts, opt-in adaptive campaign-runner followups, explicit sampling-target receipts, externally ingested result receipts, planned-versus-sampled coverage, cross-agent evidence matrices, review-readiness gates, per-iteration and loop-level promotion-review ledgers, loop-level learning dossiers, fixed and adaptive iterative loop receipts, planned multi-source sampling, worker-authored or generated blogs with provenance and section-contract receipts, synthesis, bounded parallel executor receipts, and rejects malformed findings | `test_research_loop_*`, `test_research_campaign_runner_*` |
| Contributor-scale collection follows `Link` pagination, caps retained counts, stores receipts, and redacts raw `anon=true` identities | `test_contributor_collector_*` |
| The distributed artifact has no silent file drift | `scripts/manifest.py check` |
| Source provenance is well-formed; supported techniques cite implementation evidence | `scripts/validate_sources.py` |

## Demonstrated on Apple Silicon (this repository's receipts)

The local receipt catalogue is `mlx-model-porting/assets/benchmarks/receipts_index.json`.
All receipt-backed rows below were measured on Apple M4 Pro (Mac16,8, 48 GB
unified memory), macOS 26.3, Python 3.14.4, MLX 0.30.4, and MLX-LM 0.31.1.
They are workload-specific receipts, not portable guarantees.

### Locally reproduced receipts

| Claim | Receipt labels | Chip | What was measured | Honest band / result |
|---|---|---|---|---|
| Native low-bit weight quantization | `quant-baseline-bf16`, `quant-4bit` | Apple M4 Pro | `mlx-community/Qwen3-1.7B-bf16` versus `mlx-community/Qwen3-1.7B-4bit`; 548 median prompt tokens, 256 generated tokens; median decode 54.318 -> 130.179 tok/s; median peak memory 3.825 -> 1.773 GB. | `1.0x-2.4x` decode-tokens/sec for this workload. Prefill ratio was 0.858x, and the greedy 3-prompt note diverged 3/3 versus bf16, so quality remains eval-required. |
| Prompt prefix cache | `pcache-cold`, `pcache-warm` | Apple M4 Pro | Cold 4064-token prompt versus warm 21-token suffix after reusing a 4040-token prefix cache; median TTFT proxy 2.8588s -> 0.1204s; median decode 99.749 -> 91.547 tok/s. | `1.0x-23.8x` TTFT-proxy only. This is `prompt_tokens / prompt_tps`, not instrumented first-token latency; decode ratio was 0.918x. |
| Uniform KV quantization | `kv-baseline-8k`, `kv-4bit-8k` | Apple M4 Pro | Same Qwen2.5-Coder 7B 4-bit target on an 8058-token long prompt, 256 generated tokens; median decode 29.248 -> 32.594 tok/s; median prefill 311.125 -> 282.858 tok/s; median peak memory 5.453 -> 6.570 GB. | `1.0x-1.1x` decode-tokens/sec. This receipt does not demonstrate a memory saving; peak memory increased by 1.117 GB. |
| Draft-model speculative decoding | `spec-baseline`, `spec-draft-k2`, `spec-draft-k3`, `spec-draft-k4` | Apple M4 Pro | Qwen2.5-Coder 7B 4-bit target with 1.5B 4-bit draft on a 552-token code prompt, 256 generated tokens; median decode baseline 45.122 tok/s, k=2 56.583 tok/s, k=3 30.359 tok/s, k=4 24.809 tok/s. | `1.0x-1.3x` decode-tokens/sec for the k=2-compatible slice. k=3 and k=4 regressed, so the selected draft-token count must be profiled locally. |
| Dense decoder stack measured together | `spec-baseline`, `stack-measured-together` | Apple M4 Pro | Plain 4-bit single-request baseline versus target 4-bit weights plus 4-bit KV, prompt cache, and k=2 draft model on a cached 8053-token prefix plus 34-token suffix; median decode 45.122 -> 9.516 tok/s; median TTFT proxy 1.3890s -> 0.5134s. | Measured together as `0.21x` decode throughput, so it is a negative primary-metric result, not a speedup. TTFT proxy improved 2.705x. The advisor's derived compound ceiling remains an unmeasured multiplicative hypothesis, not a measured claim. |

The local 3D `grid_sample` kernel note in `optimization_guidance.yaml` is a local
reproduction without an `assets/benchmarks/` receipt and is therefore not part of
this receipt catalogue.

### Source-reported bands, not locally reproduced here

| Claim | Provenance | Band | Boundary |
|---|---|---:|---|
| Continuous batching serving | Source-reported `vllm-mlx` benchmark | `1.0x-4.3x` batch-throughput | Applies only to comparable concurrent serving workloads; local baseline reproduction is still required. |
| VLM repeated media / multimodal content prefix cache | Source-reported `vllm-mlx` benchmark | `1.0x-28.0x` repeated-media TTFT | High end is for repeated identical media cache hits, not generic VLM prompting. |
| Qwen3-TTS batch generation | Source-reported MLX-Audio docs | `1.0x-5.45x` batch-throughput | Applies only to comparable batch size, prompt length, quantization, voice/reference, and audio-quality gates. |

### Still profile-required or parity-required

- Real MLX-vs-PyTorch numeric parity on an actual ported block or model.
- Formal task-quality gates for the local quantization, KV-cache, prompt-cache,
  speculation, and stack receipts; the current receipts use fixed greedy prompts
  and explicit no-formal-eval caveats.
- A KV-cache memory-saving claim for the measured flag set; the local 8k receipt
  increased peak memory.
- Any positive dense-decoder compound decode-speed claim from the derived
  multiplicative ceiling; the only measured-together stack receipt regressed on
  decode throughput.
- Local reproduction of source-reported continuous batching, repeated-media VLM
  cache, and Qwen3-TTS batch bands.
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
- Live execution of external research subagents or network collectors; offline tests cover the harness contract, campaign receipts, adaptive next-wave scaffold receipts, campaign runner behavior with deterministic fake researchers, bounded local executor concurrency, iterative receipts, loop-level learning dossiers, fixture ingestion, and contributor collector receipt behavior only.

## Offline gate commands

```bash
python3 -m unittest discover -s tests -v
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/manifest.py check
```
