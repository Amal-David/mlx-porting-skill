# Port and inspect PyTorch, Hugging Face, and MLX projects faster

An MLX model porting, inspection, and optimization Agent Skill (`mlx-porting-skill`).

A portable Agent Skill for taking an unfamiliar PyTorch or Hugging Face model -- language, vision, audio, speech, codec, diffusion, or hybrid -- and producing a correct, measured MLX implementation. It can also inspect an existing local MLX project or already-running port and turn it into a proof-gap, improvement, and contribution report.

Paste a PyTorch module, point at a model directory, bring a checkpoint, or point at a local MLX codebase. The skill gives you the route: architecture classification, weight-conversion guidance, parity tests, common fix patterns, project inspection, contribution candidates, and only then benchmarked MLX optimizations.

The primary goal is to distill the relevant MLX knowledge into a step-by-step runbook that frontier cloud agents, open-source models, and smaller local models can all follow instead of rediscovering the same optimization path at runtime.

This repository is deliberately **not** a single giant prompt. It has four layers:

1. `mlx-model-porting/SKILL.md`: the compact orchestration contract loaded by an agent.
2. `references/`: architecture-specific runbooks and optimization decision guides.
3. `assets/`: machine-readable architecture, technique, and evidence registries.
4. `scripts/`: safe inspection, planning, parity, benchmarking, auditing, and update tools.

## For agents

Agents should treat this repo as an executable runbook, not background reading.
Load `mlx-model-porting/SKILL.md`; then either inspect an existing MLX project
or classify the source model, choose the closest runbook, build a source oracle,
port the smallest eager MLX path, validate the weight map, pass parity, profile,
and apply one optimization dimension at a time. This lets frontier cloud agents
and smaller open-source/local models follow the same disciplined path without
doing fresh runtime discovery for every CUDA/PyTorch-to-MLX request.

A web version of the runbook lives in [`site/index.html`](site/index.html) and
is published at [mlx-porter.pages.dev](https://mlx-porter.pages.dev/).

## What it covers

- model intake, local MLX project inspection, architecture fingerprinting, licensing and remote-code risk review;
- source-oracle construction and layer-by-layer numerical parity;
- deterministic weight conversion and shape-transform manifests;
- MLX lazy evaluation, compilation, streams, memory, fused operations, and custom Metal kernels;
- dense decoder, MoE, encoder, encoder-decoder, SSM/recurrent, diffusion/flow, VLM/omni;
- neural audio codecs, autoregressive audio LMs, flow/diffusion TTS, vocoders, ASR, streaming speech, and separation;
- weight/KV/mixed-bit quantization, prompt caching, continuous batching, speculative decoding, and serving;
- benchmark gates that prevent unsupported speedup claims;
- review-only daily and deep-research loops that never auto-merge research changes.

## Porting examples people can start from

For the recognizable "before/after" paths, see
[`mlx-model-porting/examples/porting-patterns.md`](mlx-model-porting/examples/porting-patterns.md).

The examples are deliberately compact pattern cards, not claims that every named
model family is fully ported. They show how to turn common PyTorch surfaces into
an MLX port plan with the validation gates this repo already enforces:

- LoRA fine-tuning to MLX;
- Whisper-style audio model port;
- small diffusion block port;
- Transformer attention block port;
- PyTorch checkpoint loading into MLX.

## Install

The skill itself is the `mlx-model-porting/` directory. Copy or symlink that directory into the Agent Skills root used by your client. For clients that support the shared Agent Skills format, no rewrite is necessary.

Fresh checkouts include repo-scoped discovery links for Claude Code and Codex-style `.agents` roots. Use the installer presets below when you need to refresh those links or install into another client root.

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex
```

For an explicit user-scoped or version-specific installation:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest ~/.agents/skills --mode symlink
```

See `adapters/README.md` for client notes. Exact discovery roots can vary by product and version, so the installer accepts an explicit destination rather than guessing silently.

## First use

```bash
python3 mlx-model-porting/scripts/inspect_model.py /path/to/model --output inspection.json
python3 mlx-model-porting/scripts/make_port_plan.py inspection.json --output PORT_PLAN.md
python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json --markdown OPTIMIZATIONS.md
```

When a family stack applies, `OPTIMIZATIONS.md` includes a recommended stack with
per-step bands plus a measured-together receipt or a derived ceiling clearly
flagged as an unmeasured multiplicative hypothesis.

Then ask the agent:

> Use the MLX model porting skill. Port this model to MLX, establish source parity first, and optimize only through benchmarked changes.

If you are starting from a pasted module instead of a full model directory, pick
the closest pattern in
[`mlx-model-porting/examples/porting-patterns.md`](mlx-model-porting/examples/porting-patterns.md)
and ask for a `PORT_PLAN.md`, source-oracle checkpoints, weight-map rules, and
the smallest eager MLX block before any optimization work.

## Inspector mode for existing MLX projects

If you already have local MLX code, a served MLX app, or an already-converted
checkpoint, start with the project inspector instead of pretending the work is
greenfield:

```bash
python3 mlx-model-porting/scripts/inspect_mlx_project.py /path/to/mlx-project \
  --model /path/to/local-model \
  --markdown MLX_INSPECTION.md \
  --output inspection.json
```

The report says what is running, what proof is visible, what likely improvement
paths exist, and whether any local pattern is a contribution candidate for this
repo. Contribution candidates still need a small fixture, parity evidence,
benchmark metadata, rollback condition, and the exact runbook or asset that
would change.

## Try it offline (no model download, no Apple Silicon, no network)

The repository ships tiny synthetic fixtures, so you can exercise the whole static pipeline immediately:

```bash
python3 mlx-model-porting/scripts/inspect_model.py tests/fixtures/models/decoder --output /tmp/inspection.json
python3 mlx-model-porting/scripts/make_port_plan.py /tmp/inspection.json --output /tmp/PORT_PLAN.md
python3 mlx-model-porting/scripts/recommend_optimizations.py /tmp/inspection.json --markdown /tmp/OPTIMIZATIONS.md
```

Expected results for this fixture:

- `inspection.json` → `recommended_family: dense-decoder-transformer`, `recommended_runbook: references/runbook-decoder-transformer.md`;
- `OPTIMIZATIONS.md` lists a dense-decoder recommended stack, `fast-sdpa` and `uniform-kv-quantization` as ready candidates, and `cuda-graphs-decode-capture` under **Rejected for MLX (do not port)**.

This exact flow — routing, weight-key coverage, seeded-parity-bug detection, and optimization inclusion/exclusion — is guarded end to end by `tests/test_scenarios.py`.

## Measured on Apple Silicon (representative receipts)

These are single-workload receipts on ONE machine (Apple M4 Pro, Mac16,8, 48 GB, macOS 26.3, MLX 0.30.4, MLX-LM 0.31.1), reproducible via `mlx-model-porting/scripts/benchmark_generation.py`, and are planning bands, not portable guarantees.

| Technique | Model / workload | Measured result | Notes |
|---|---|---|---|
| 4-bit weight quantization | Qwen3-1.7B, 548-tok prompt | **2.40x** decode (54.3 → 130.2 tok/s); peak mem 3.83 → 1.77 GB | Quality caveat: greedy outputs diverged on 3/3 short prompts, no formal eval |
| Prompt cache (cold → warm) | Qwen3-1.7B, ~4k-tok prompt | **23.75x** TTFT-proxy (2.859 → 0.120 s) | Prefill/TTFT win; decode unchanged |
| Speculative decoding | Qwen2.5-Coder-7B-4bit + 1.5B draft, 552-tok code prompt | k=2 **1.25x** (45.1 → 56.6 tok/s); k=3 0.67x; k=4 0.55x | Best at k=2; k≥3 regresses |
| 4-bit KV quantization @ 8k ctx | Qwen2.5-Coder-7B-4bit, 8058-tok prompt | **1.11x** decode (29.2 → 32.6 tok/s); peak mem 5.45 → 6.57 GB | This flag set cost memory, not saved it |
| Full stack @ 8k ctx (vs plain 4-bit @ 8k) | 4-bit + KV-4bit + prompt cache + k=2 draft | **0.33x** decode (29.2 → 9.5 tok/s) — regression | Driven by the draft model at long context; mlx-lm 0.31.1 also needs a combined target+draft cache |

The honest headline is that 4-bit quant and prompt caching are the reliable Apple-Silicon wins; speculative decoding pays off only at small draft depth on short context; and naively stacking a draft model onto a long-context run regresses throughput — which is exactly why the skill measures stacks instead of multiplying per-technique bands.

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

For broad ecosystem research, use `mlx-model-porting/scripts/research_loop.py`. It generates bounded researcher assignments, can dynamically choose agents from objective terms and `--gap-hint` inputs, writes subagent handoff packets and campaign receipts, can run bounded local executor workers or ingest externally written subagent results, ingests returned findings and worker-authored blogs, validates blog-section contracts while preserving generated fallbacks, validates optional explicit sampling-target receipts, and emits review-only `synthesis`, sampling coverage, cross-agent evidence matrix, promotion-review ledger, plus fixed or review-gate-adaptive multi-iteration `loop` receipts with aggregate promotion-review rollups, loop-level learning dossiers, and post-ingest `next_wave_scaffold` commands for adaptive external dispatch under `mlx-model-porting/research-runs/` without modifying recommendation assets automatically. Use `mlx-model-porting/scripts/run_research_campaign.py` when an explicit local researcher command should execute the campaign receipt wave by wave, preserve `campaign-run` receipts, and optionally follow next-wave scaffold receipts for bounded adaptive local loops.

Current README positioning review date: **2026-07-08**. See `VALIDATION.md` for the deeper corpus and receipt validation dates.

## Versioning

See `CHANGELOG.md`; the current package version is recorded in `VERSION` and the skill frontmatter.
