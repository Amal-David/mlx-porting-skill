# mlx-model-porting

[![skills.sh installs](https://skills.sh/b/Amal-David/mlx-porting-skill)](https://skills.sh/Amal-David/mlx-porting-skill)

An Agent Skill and offline CLI toolchain for porting PyTorch / Hugging Face
models to Apple MLX:

> static inspection → architecture routing → source-oracle capture →
> explicit weight conversion → staged parity → benchmarking →
> evidence-gated optimization advice

It can also inspect an existing MLX project before you change it.

Docs: [mlx-porter.pages.dev](https://mlx-porter.pages.dev/) (offline source in
[`site/`](site/)).

## What works today

This is an engineering workflow, **not an arbitrary-checkpoint converter**.
17 architecture families are routed and planned; the executable depth varies:

| Route | Executable today |
|---|---|
| Dense decoder | Full chain, proven end to end on Qwen2.5-0.5B-Instruct |
| BERT encoder | Full chain, proven on BGE-base |
| Encoder-decoder (non-gated ReLU T5) | Full chain, proven on t5-small |
| ASR encoder (HuBERT/Wav2Vec2) | Full chain from extracted features, proven on HuBERT-base |
| Sparse MoE, selective SSM | Scaffold and parity on synthetic profiles only |
| Other 11 families | Inspection, routing, planning, and generic validation; MLX modules remain runbook-guided |

Exact output parity is the only built-in quality metric, and no performance
claim is promoted. [`VALIDATION.md`](VALIDATION.md) states precisely what the
checked-in gates do and do not prove.

The 0.6.1 corpus behind the skill:

- 363 evidence sources with explicit review depth; 35 currently carry classified
  support scope and claim types, while 328 remain intentionally unclassified;
- 33 inspectable Python scripts and 537 offline tests;
- 13 benchmark receipts (12 observations, 1 rejected) and 10 effective claims,
  all withheld.

## Install

The distributable skill is the `mlx-model-porting/` directory:

```bash
npx skills add Amal-David/mlx-porting-skill --skill mlx-model-porting
```

For the stricter manifest-attested install (verifies every file hash against
`MANIFEST.json`), run from a repository checkout:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex
# or an explicit destination:
python3 mlx-model-porting/scripts/install_skill.py --dest ~/.agents/skills --mode symlink
```

Native Windows should use the `npx` path (attested copy mode requires WSL).
Client discovery notes: [`adapters/README.md`](adapters/README.md). The
installer never installs Python packages or executes model code.

### Dependencies

Inspection, routing, planning, and validation need only **Python 3.10+
stdlib**. The optional tools need more:

| Tool | Requires |
|---|---|
| `compare_tensors.py` | NumPy |
| `capture_oracle.py` | PyTorch, Transformers, NumPy |
| `--allow-network` Hugging Face intake | `huggingface-hub` |
| Running the finished port | Apple Silicon + the port's MLX packages |

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes \
  -r mlx-model-porting/requirements-tools.lock
```

## Port a model

Everything below is offline by default and refuses Hugging Face remote code.

**1. Inspect and plan** (any family, static only):

```bash
python3 mlx-model-porting/scripts/inspect_model.py MODEL \
  --output inspection.json --markdown inspection.md

python3 mlx-model-porting/scripts/make_port_plan.py inspection.json \
  --artifact-root MODEL --output PORT_PLAN.md
```

If inspection reports a blocker, the plan contains remediation steps only.
The planner re-inspects the `--artifact-root` bytes itself, so a hand-edited
`inspection.json` is not trusted.

**2. Capture the source oracle** — frozen Torch ground truth (inputs,
per-layer tensors, logits, greedy token IDs):

```bash
python3 mlx-model-porting/scripts/capture_oracle.py MODEL \
  --token-ids 1 42 17 9 --generate-steps 4 \
  --output source-oracle.npz
```

Capture/parity modes: `dense-decoder` (default), `encoder`, `encoder-decoder`,
`ssm`, and `asr` (add `--waveform-samples 16000`).

**3. Scaffold the MLX package and draft the weight map:**

```bash
python3 mlx-model-porting/scripts/scaffold_port.py inspection.json \
  --artifact-root MODEL --output mlx_port

python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source inspection.json \
  --scaffold-manifest mlx_port/scaffold-manifest.json \
  --emit-draft-map WEIGHT_MAP.draft.json
```

Unsupported or ambiguous configs fail closed before code generation. Review
the draft map, resolve every entry, set `"draft": false`, then validate and
convert:

```bash
python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source inspection.json \
  --target mlx_port/scaffold-manifest.json \
  --mapping WEIGHT_MAP.json --output weight-map-report.json

python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source MODEL --mapping WEIGHT_MAP.json --output converted
```

**4. Run staged parity** — captures both sides and stops at the first
divergence (input → embedding → layer → norm → logits → generated IDs):

```bash
python3 mlx-model-porting/scripts/run_parity.py \
  --source-model MODEL --package mlx_port --weights converted \
  --token-ids 1 42 17 9 --generate-steps 4 \
  --output parity-report.json
```

Choose tolerances for your model and task; never relax a default just to turn
a first-divergence report green.

### Try it without downloading a model

The synthetic fixture exercises the router and guard path only:

```bash
python3 mlx-model-porting/scripts/inspect_model.py tests/fixtures/models/decoder \
  --output /tmp/inspection.json

python3 mlx-model-porting/scripts/make_port_plan.py /tmp/inspection.json \
  --artifact-root tests/fixtures/models/decoder --output /tmp/PORT_PLAN.md
```

### Worked examples

Complete offline runs with manifests, weight maps, and parity reports (weights
excluded) in [`mlx-model-porting/examples/`](mlx-model-porting/examples/):
[Qwen2.5-0.5B-Instruct](mlx-model-porting/examples/worked-port-qwen2.5-0.5b-instruct/README.md)
(all 29 parity rungs and 8 greedy tokens matched),
[BGE-base](mlx-model-porting/examples/worked-port-bge-base-en),
[t5-small](mlx-model-porting/examples/worked-port-t5-small), and
[HuBERT-base](mlx-model-porting/examples/worked-port-hubert-base-ls960/README.md).

## Other workflows

### Inspect an existing MLX project

Inventories files and proof surfaces without running the project:

```bash
python3 mlx-model-porting/scripts/inspect_mlx_project.py PROJECT \
  --model LOCAL_MODEL --output inspection.json --markdown MLX_INSPECTION.md
```

### Validate a port you already have

```bash
python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source source.json --target target.json \
  --mapping WEIGHT_MAP.json --output weight-map-report.json

python3 mlx-model-porting/scripts/compare_tensors.py source.npz target.npz \
  --mapping mapping.json --atol 1e-5 --rtol 1e-4 --cosine-min 0.99 \
  --output parity.json
```

A generic tensor threshold alone does not prove task correctness for language,
vision, audio, speech, diffusion, or streaming workloads.

### Get optimization advice

```bash
python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json \
  --target-profile target-profile.json --objective peak-memory \
  --output recommendations.json --markdown OPTIMIZATIONS.md
```

Run it only after a clean inspection. Advice lands in five buckets:
`validated-locally`, `validated-source-theory`, `benchmark-required`,
`experimental-approach` (explicit opt-in), and `rejected-do-not-use`. Numbers
come only from the generated
[`effective_claims.json`](mlx-model-porting/assets/effective_claims.json) —
all 10 current claims are withheld until a local benchmark passes the
promotion contract. Benchmark state:
[`BENCHMARK_REPORT.md`](mlx-model-porting/assets/BENCHMARK_REPORT.md);
promotion rules and evidence semantics: [`VALIDATION.md`](VALIDATION.md) and
[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md).

## Repository layout

| Path | Purpose |
|---|---|
| [`mlx-model-porting/SKILL.md`](mlx-model-porting/SKILL.md) | Compact agent contract and trigger map |
| [`mlx-model-porting/references/`](mlx-model-porting/references/) | Porting method, failure atlas, optimization guides, 17 family runbooks |
| [`mlx-model-porting/assets/`](mlx-model-porting/assets/) | Canonical architecture, technique, source, benchmark, and claim registries |
| [`mlx-model-porting/scripts/`](mlx-model-porting/scripts/) | Non-destructive inspection, planning, parity, benchmarking, and packaging tools |
| [`mlx-model-porting/examples/`](mlx-model-porting/examples/) | Porting patterns and worked example ports |
| [`tests/`](tests/) | Offline contract, security, determinism, and portability tests |
| [`site/`](site/) | Offline source of the public runbook site |

## Development

```bash
python3 -m unittest discover -s tests
```

Never hand-edit generated reports or indexes — change the canonical registry,
runbook, receipt, or version input and regenerate in dependency order. The
ownership table, regeneration commands, and full release-gate list are in
[`CONTRIBUTING.md`](CONTRIBUTING.md); architecture and extension flows are in
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md).

## Versioning and license

Current release: **0.6.1** (2026-07-14). Version lives in
[`VERSION`](VERSION) and the skill frontmatter; changes in
[`CHANGELOG.md`](CHANGELOG.md). Licensed [Apache-2.0](LICENSE).
