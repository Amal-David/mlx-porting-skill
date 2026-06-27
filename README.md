# mlx-porting-skill

An MLX model porting and optimization Agent Skill.

A portable Agent Skill for taking an unfamiliar PyTorch or Hugging Face model—language, vision, audio, speech, codec, diffusion, or hybrid—and producing a correct, measured MLX implementation.

This repository is deliberately **not** a single giant prompt. It has four layers:

1. `mlx-model-porting/SKILL.md`: the compact orchestration contract loaded by an agent.
2. `references/`: architecture-specific runbooks and optimization decision guides.
3. `assets/`: machine-readable architecture, technique, and evidence registries.
4. `scripts/`: safe inspection, planning, parity, benchmarking, auditing, and update tools.

## What it covers

- model intake, architecture fingerprinting, licensing and remote-code risk review;
- source-oracle construction and layer-by-layer numerical parity;
- deterministic weight conversion and shape-transform manifests;
- MLX lazy evaluation, compilation, streams, memory, fused operations, and custom Metal kernels;
- dense decoder, MoE, encoder, encoder-decoder, SSM/recurrent, diffusion/flow, VLM/omni;
- neural audio codecs, autoregressive audio LMs, flow/diffusion TTS, vocoders, ASR, streaming speech, and separation;
- weight/KV/mixed-bit quantization, prompt caching, continuous batching, speculative decoding, and serving;
- benchmark gates that prevent unsupported speedup claims;
- review-only daily and deep-research loops that never auto-merge research changes.

## Install

The skill itself is the `mlx-model-porting/` directory. Copy or symlink that directory into the Agent Skills root used by your client. For clients that support the shared Agent Skills format, no rewrite is necessary.

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest ~/.agents/skills
```

For a repository-scoped Codex installation:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest .agents/skills --mode symlink
```

See `adapters/README.md` for client notes. Exact discovery roots can vary by product and version, so the installer accepts an explicit destination rather than guessing silently.

## First use

```bash
python3 mlx-model-porting/scripts/inspect_model.py /path/to/model --output inspection.json
python3 mlx-model-porting/scripts/make_port_plan.py inspection.json --output PORT_PLAN.md
python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json --markdown OPTIMIZATIONS.md
```

Then ask the agent:

> Use the MLX model porting skill. Port this model to MLX, establish source parity first, and optimize only through benchmarked changes.

## Try it offline (no model download, no Apple Silicon, no network)

The repository ships tiny synthetic fixtures, so you can exercise the whole static pipeline immediately:

```bash
python3 mlx-model-porting/scripts/inspect_model.py tests/fixtures/models/decoder --output /tmp/inspection.json
python3 mlx-model-porting/scripts/make_port_plan.py /tmp/inspection.json --output /tmp/PORT_PLAN.md
python3 mlx-model-porting/scripts/recommend_optimizations.py /tmp/inspection.json --markdown /tmp/OPTIMIZATIONS.md
```

Expected results for this fixture:

- `inspection.json` → `recommended_family: dense-decoder-transformer`, `recommended_runbook: references/runbook-decoder-transformer.md`;
- `OPTIMIZATIONS.md` lists `fast-sdpa` and `uniform-kv-quantization` as ready candidates and `cuda-graphs-decode-capture` under **Rejected for MLX (do not port)**.

This exact flow — routing, weight-key coverage, seeded-parity-bug detection, and optimization inclusion/exclusion — is guarded end to end by `tests/test_scenarios.py`.

## Validation

```bash
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/manifest.py check
python3 -m unittest discover -s tests -v
```

`VALIDATION.md` states exactly what these offline gates prove versus what still requires an Apple Silicon Mac or network access — readiness is intentionally not overclaimed.

When `skills-ref` is installed, also run:

```bash
skills-ref validate ./mlx-model-porting
```

## Research status

`RESEARCH_REPORT.md` explains the landscape and design. `EVIDENCE_INDEX.md` renders the complete source catalogue, while `mlx-model-porting/assets/sources.yaml` is the machine-readable evidence index with review depth. A source marked `indexed` has been catalogued but must not be represented as fully reviewed. A source marked `synthesized` directly informed a runbook or decision rule.

For broad ecosystem research, use `mlx-model-porting/scripts/research_loop.py`. It generates bounded researcher assignments, can dynamically choose agents from objective terms and `--gap-hint` inputs, writes subagent handoff packets and campaign receipts, can run bounded local executor workers or ingest externally written subagent results, ingests returned findings and worker-authored blogs, validates blog-section contracts while preserving generated fallbacks, validates optional explicit sampling-target receipts, and emits review-only `synthesis`, sampling coverage, promotion-review ledger, plus fixed or review-gate-adaptive multi-iteration `loop` receipts with aggregate promotion-review rollups under `mlx-model-porting/research-runs/` without modifying recommendation assets automatically. Use `mlx-model-porting/scripts/run_research_campaign.py` when an explicit local researcher command should execute the campaign receipt wave by wave and preserve `campaign-run` receipts.

Current corpus review date: **2026-06-27**.

## Versioning

See `CHANGELOG.md`; the current package version is recorded in `VERSION` and the skill frontmatter.
