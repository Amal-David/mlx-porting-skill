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
- a review-only daily update pipeline that never auto-merges research changes.

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

## Validation

```bash
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 -m unittest discover -s tests -v
```

When `skills-ref` is installed, also run:

```bash
skills-ref validate ./mlx-model-porting
```

## Research status

`RESEARCH_REPORT.md` explains the landscape and design. `EVIDENCE_INDEX.md` renders the complete source catalogue, while `mlx-model-porting/assets/sources.yaml` is the machine-readable evidence index with review depth. A source marked `indexed` has been catalogued but must not be represented as fully reviewed. A source marked `synthesized` directly informed a runbook or decision rule.

Current corpus review date: **2026-06-24**.

## Versioning

See `CHANGELOG.md`; the current package version is recorded in `VERSION` and the skill frontmatter.
