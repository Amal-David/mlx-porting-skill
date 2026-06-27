# Research Loop 2026-06-27-top1000-contributor-loop

Objective: Extend MLX contributor implementation learning to the top-1000 contributor request and preserve promotion/backlog decisions

Review only: True
Findings: 10
Non-GitHub lanes covered: official_docs, repo_local_audit

## Decision Counts
- adopted: 3
- held: 2
- rejected: 0
- needs-validation: 5

## Adopted
- top1000-contributor-set-covered-available-api: Top-1000 request covered API-returned linked contributors and recorded the anonymous delta - The expanded sweep requested ml-explore/mlx contributors pages 1-10 at 100 per page. GitHub returned 256 linked contributors across pages 1-3, or 262 author buckets with anon=true, so the top-1000 objective covered every linked contributor available through that API at retrieval time.
- top1000-promoted-block-streaming: Repeated-block weight streaming has pinned implementation and rollback gates - dgrauet/ltx-2-mlx provided inspectable safetensors block streaming code, tests, and a regression report, so the skill now treats repeated-block weight streaming as a proven MLX-port technique for memory-bound diffusion or flow-style models.
- top1000-promoted-grid-sample: PyTorch-compatible grid_sample custom kernels have enough evidence for scoped guidance - katlun-lgtm/mlx-grid-sample provided custom Metal code, correctness tests, and a benchmark harness for 2D/3D grid_sample. The skill now scopes this as a proven MLX-port technique for spatial or volume warping after parity and end-to-end benchmark gates.

## Held
- top1000-source-format-onnx-adjacent: MLX ONNX export tooling is adjacent evidence, not intake support - A top-1000 contributor-owned mlx-onnx repository shows MLX-to-ONNX export ecosystem work, but it does not prove ONNX-to-MLX model intake. Keep ONNX/JAX/TF/GGUF/Core ML intake as a P0 validation backlog.
- top1000-paged-kv-moe-reinforces-existing: Contributor serving repos reinforce paged-KV and MoE gates - tiny-llm and mlx-moe show useful paged-KV, continuous batching, quantized expert, and synthetic MoE assembly patterns, but they reinforce existing validation gates rather than introducing a new supported optimization.

## Rejected
- None

## Needs Validation
- top1000-link-header-collector-receipts: Contributor-scale sweeps need Link-header and rate-limit receipts - GitHub pagination and API best-practice docs require following Link headers, using authenticated serial requests, and preserving rate-limit/conditional-request receipts. Hard-coded page caps are safety caps, not proof that all pages exist.
- top1000-rwkv-ssm-conformer-held: RWKV, SSM, and Conformer prototypes reinforce validation gaps - Contributor-owned RWKV, SSM/Mamba, and Conformer repositories are useful architecture leads, but current snapshots show WIP or incomplete kernels and no packaged parity suite strong enough for supported guidance.
- source-format-static-intake-expanded: Static intake must cover source formats beyond PyTorch and Hugging Face - Primary docs for ONNX, Flax/Orbax, TensorFlow/Keras, GGUF, Core ML, and safetensors show inspectable metadata surfaces. These can support intake triage and unsupported-op reporting, but not automatic conversion claims.
- top1000-long-tail-rescreening-needed: Top-1000 contributor research needs repeatable long-tail rescreening - The sweep found 71 repository-search matches and retained earlier code-search results, but GitHub code search rate limits interrupted several code queries. Search is lead generation, not proof that the long tail was exhausted, so preserve this as an explicit backlog item.
- coverage-gaps-need-family-specific-fixtures: Underserved model families need family-specific fixtures before support language - Non-generative CV, graph/geometric/scientific ML, time-series, structured, and recsys ports have source-framework semantics that generic encoder or VLM parity checks do not cover.

## Limitations
- The top-1000 request used the GitHub contributors API and returned 256 linked contributors and 262 anon=true author buckets at retrieval time.
- Repository search matched 71 candidate repositories; several GitHub code-search queries hit rate limits and remain backlog.
- Only two implementation learnings were promoted, both with pinned evidence and explicit validation gates.
