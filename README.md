# MLX model porting, inspection, and evidence-gated optimization

`mlx-model-porting` is a portable Agent Skill for taking an unfamiliar PyTorch
or Hugging Face model through static intake, architecture routing, source
capture, explicit weight conversion, staged MLX parity, profiling, and
evidence-gated optimization. It can also inspect an existing MLX project before
improving it.

This repository is an engineering workflow, **not an arbitrary-checkpoint
conversion system**. Six families have executable scaffolds. Four real packets
prove Qwen2.5 dense decoder, BGE BERT encoder, t5-small encoder-decoder, and
HuBERT-base acoustic encoder. Sparse MoE and selective SSM have synthetic
correctness gates only. The other 11 architecture families remain
runbook-guided. All 17 routes have golden routing scenarios; those fixtures are
not 17 completed real-model ports. Exact output is the only built-in task
metric, and no performance claim is promoted.

The public runbook is available at
[mlx-porter.pages.dev](https://mlx-porter.pages.dev/), and its offline source is
in [`site/`](site/).

## What is in the repository

| Layer | Purpose |
|---|---|
| [`mlx-model-porting/SKILL.md`](mlx-model-porting/SKILL.md) | Compact agent contract and trigger map. |
| [`mlx-model-porting/references/`](mlx-model-porting/references/) | Porting method, failure atlas, optimization guides, and 17 family runbooks. |
| [`mlx-model-porting/assets/`](mlx-model-porting/assets/) | Canonical architecture, technique, guidance, stack, source, benchmark, and claim data. |
| [`mlx-model-porting/scripts/`](mlx-model-porting/scripts/) | Non-destructive inspection, planning, parity, benchmarking, evidence, and packaging tools. |
| [`mlx-model-porting/examples/`](mlx-model-porting/examples/) | Porting patterns and no-weights worked examples. |
| [`tests/`](tests/) | Synthetic scenarios and contract, security, determinism, and portability tests. |

Current 0.6.1 corpus snapshot:

- 17 architecture-family routes with synthetic golden coverage;
- 361 evidence sources with explicit review depth; 35 currently carry classified
  support scope and claim types, while 326 remain intentionally unclassified;
- 70 technique records, 28 optimization-guidance methods, and 4 stack
  definitions;
- 33 inspectable Python scripts and 537 offline tests;
- a 708-node, 501-edge research graph plus a deterministically reconciled
  backlog;
- 13 checked-in benchmark receipts: 12 `performance_observation`, 0
  `promotion_ready`, and 1 `rejected`;
- 10 effective claims, all withheld.

## Capability boundary

The tooled path is deliberately narrower than the routing catalogue:

| Scope | What is executable now | What remains runbook-guided or future work |
|---|---|---|
| `dense-decoder-transformer` | Static inspection, Torch oracle capture, eager MLX scaffold generation, schema-2 weight-map conversion, MLX capture, first-divergence parity, benchmarking, and claim gating. | Additional configs still fail closed until their semantics are implemented; only one real checkpoint has completed the full path. |
| `encoder-transformer` | Absolute-position BERT scaffold, conversion, padded encoder capture, pooler handling, and per-layer parity; proven on BGE base. | RoBERTa and other position schemes, non-BERT encoders, and task-quality evaluation. |
| `encoder-decoder-transformer` | Non-gated ReLU T5 scaffold with encoder/decoder/cross-attention capture and cache checks; proven on t5-small. | BART, NLLB, Whisper, gated T5, beam search, and task-quality evaluation. |
| `automatic-speech-recognition` | HuBERT/Wav2Vec2 feature projection, convolutional positional embedding, transformer-encoder scaffolding, shared extracted-feature capture, conversion, and per-layer parity. | Raw-waveform convolutional frontend, CTC/transducer/seq2seq decoding, Whisper, transcripts, WER, and timestamps. |
| `moe-decoder-transformer` | Synthetic sparse router/expert scaffold and parity for supported profiles. | A checked-in real checkpoint, unsupported router profiles, shared experts, grouped routing, and expert parallelism. |
| `ssm-recurrent-hybrid` | Synthetic opt-in `minimal_selective` recurrence with carried-state and NumPy checks. | Real Mamba/Mamba2 checkpoints and attention-mixed hybrids. |
| Other 11 families | Static inspection, routing, planning, generic tensor comparison, benchmarking, and evidence gates. | Architecture-specific MLX modules and capture wiring remain runbook-guided. |
| Quality evaluation | Exact tensor/ID comparison and controlled exact-output benchmark parity. | Domain evaluators for language quality, vision, audio, speech, diffusion, streaming, scientific tasks, and lossy changes. |

## How the workflow fits together

1. **Inspect without executing model code.** Inventory configuration,
   safetensors metadata, source formats, licenses, and remote-code risks. Local
   intake binds every file into a portable SHA-256 tree identity; incomplete
   identity or missing artifact-bound license evidence blocks advice.
2. **Route to a family runbook.** Confidence and ambiguity gates can stop a
   weak route; hybrid models can require more than one runbook.
3. **Build a source oracle and the smallest eager MLX graph.** Freeze fixtures
   and intermediate tensors before changing performance behavior. The six
   executable families use the checked-in scaffold and mode-dispatched capture
   tools; other families follow their runbooks.
4. **Convert weights deterministically.** Every rename, transpose, reshape,
   split, merge, and tie belongs in an inspectable weight map.
5. **Pass staged parity.** The parity runner captures both sides and
   stops at the first input, embedding, layer, norm, logit, or exact-token
   divergence; ASR encoder mode compares extracted features, encoder input,
   every layer, and final hidden state. Other families use generic primitives until a
   family-specific runner exists.
6. **Profile, then advise.** Recommendations match controlled family,
   capability, workload, objective, and software identifiers exactly. A local
   numeric promotion additionally requires the exact receipt-derived experiment
   fingerprint, not a look-alike generic profile.
7. **Measure one change at a time.** Benchmark receipts are observations until
   lineage, workload, runner, stability, quality, rollback, and compatibility
   gates all pass.
8. **Package with provenance.** Preserve source revision, license, conversion
   recipe, validation results, limits, and rollback conditions.

See [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) for the architecture and extension
flows, and [`VALIDATION.md`](VALIDATION.md) for exactly what the checked-in gates
do and do not prove.

## Install

The distributable skill is the `mlx-model-porting/` directory. For a supported
client preset:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex
```

For an explicit destination:

```bash
python3 mlx-model-porting/scripts/install_skill.py \
  --dest ~/.agents/skills \
  --mode symlink
```

Copy mode is a release-artifact operation: run it from the repository checkout
that contains `MANIFEST.json`. It installs only the `mlx-model-porting/`
allowlist from that manifest, verifies every file hash and executable bit,
rejects unlisted source content and unsafe links, and then verifies the staged
tree is an exact identity match. Known local cache/build noise is never copied.
The installed skill includes its own complete Apache-2.0 `LICENSE`.

Symlink mode points the client at the checkout instead of materializing an
attested copy. Both modes refuse unsafe source/destination nesting and are
idempotent for an unchanged target. See
[`adapters/README.md`](adapters/README.md) for client-specific discovery notes.

### Runtime dependencies

| Surface | Requirement |
|---|---|
| Static intake, routing, planning, registries, and validation | Python 3.10+ standard library |
| Tensor parity with `compare_tensors.py` | NumPy |
| Local source-oracle capture with `capture_oracle.py` | PyTorch, Transformers, and NumPy |
| Explicit `--allow-network` Hugging Face intake | `huggingface-hub` |
| Model execution | Apple Silicon plus the exact MLX/framework packages required by the chosen port |

Create a dedicated environment for the optional tools rather than relying on
machine-global packages:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --require-hashes \
  -r mlx-model-porting/requirements-tools.lock
```

`requirements-tools.txt` is the small authored input; the universal lock pins
transitive versions and artifact hashes. Refresh it with `uv pip compile
mlx-model-porting/requirements-tools.txt --universal --python-version 3.10
--generate-hashes --output-file mlx-model-porting/requirements-tools.lock`.
CI additionally pins its minimal platform wheels in
`.github/requirements-ci.txt`. The skill installer deliberately does not install
Python packages or execute model code.

## Start from a source model

The default inspection path is offline and static. Network metadata access must
be enabled explicitly.

```bash
python3 mlx-model-porting/scripts/inspect_model.py MODEL \
  --output inspection.json \
  --markdown inspection.md

python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json \
  --output recommendations.json \
  --markdown OPTIMIZATIONS.md

python3 mlx-model-porting/scripts/make_port_plan.py inspection.json \
  --artifact-root MODEL \
  --recommendations recommendations.json \
  --output PORT_PLAN.md
```

Run the recommender only after inspection is clean. If inspection reports any
blocker, generate the port plan without recommendations; it will contain
remediation steps only, and `--family` cannot turn it into an actionable plan.
Every actionable plan requires `--artifact-root`: the planner reruns the static
inspector against those local bytes and rejects any artifact, routing, license,
or safety drift. It also recomputes the complete recommendation report before
embedding advice, so an edited intermediate report is not trusted.

Inspection reports use portable basename-only local references by default.
Pass `--include-local-paths` only when an absolute path is deliberately needed
for local debugging; do not publish that form.

### Generated execution toolchain

For any of the six executable families, the chain runs in this order:

1. `capture_oracle.py` loads only a pinned local Hugging Face model with remote
   code disabled and records the deterministic ladder selected by `--mode` in
   a bounded NPZ plus a content-addressed manifest. Text modes capture their
   IDs/masks and relevant encoder, decoder, or SSM boundaries; ASR captures
   frozen input features and acoustic-encoder states. Only generative decoder
   modes include logits and greedy token IDs.
2. `scaffold_port.py` re-inspects the same artifact and generates the minimal
   eager MLX package. Family-specific config identity and tensor topology fail
   closed before code generation.
3. `convert_checkpoint.py` drafts or applies a complete `WEIGHT_MAP` schema 2,
   rejects unresolved coverage and shape drift, and writes deterministic
   safetensors for the generated package.
4. `capture_mlx.py` validates the scaffold identity, executes the user-owned MLX
   package, and emits source-compatible tensors and a bounded manifest.
5. `run_parity.py` is the one-command parity flow after scaffolding and
   conversion. It invokes source and MLX capture, compares the ordered ladder,
   and stops at the first input, embedding, layer, final-norm, logit, or exact
   generated-ID divergence.
6. `_capture_common.py` is the shared non-CLI contract for bounded inputs,
   manifests, tensor inventories, and strict artifact writing used by all three
   capture/parity commands.

Capture/parity modes are `dense-decoder` (default), `encoder`,
`encoder-decoder`, `ssm`, and `asr`. ASR additionally accepts
`--waveform-samples 16000`; its source capture freezes real Torch-extracted
frontend features and the MLX capture consumes that same NPZ tensor. See
[`worked-port-hubert-base-ls960`](mlx-model-porting/examples/worked-port-hubert-base-ls960/README.md).

After inspection pins the local source artifacts, capture the executable source
oracle before implementing or optimizing the MLX graph. Use token IDs when the
fixture should not depend on tokenizer behavior:

```bash
python3 mlx-model-porting/scripts/capture_oracle.py MODEL \
  --token-ids 1 42 17 9 \
  --generate-steps 4 \
  --output source-oracle.npz
```

The command stays offline, refuses Hugging Face remote code, and writes
`source-oracle.manifest.json` beside the NPZ. The manifest binds config and
weight digests, capture inputs, library versions, and every tensor's shape,
dtype, and raw-byte SHA-256.

Generate the dense-decoder package, resolve and validate the draft map, then
convert the weights:

```bash
python3 mlx-model-porting/scripts/scaffold_port.py inspection.json \
  --artifact-root MODEL \
  --output mlx_port

python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source inspection.json \
  --scaffold-manifest mlx_port/scaffold-manifest.json \
  --emit-draft-map WEIGHT_MAP.draft.json

python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source inspection.json \
  --target mlx_port/scaffold-manifest.json \
  --mapping WEIGHT_MAP.json \
  --output weight-map-report.json

python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source MODEL \
  --mapping WEIGHT_MAP.json \
  --output converted
```

The draft is not executable authority: review it, set `draft` to `false`, and
resolve every entry before conversion. Then run the source-to-MLX parity ladder
in one command:

```bash
python3 mlx-model-porting/scripts/run_parity.py \
  --source-model MODEL \
  --package mlx_port \
  --weights converted \
  --token-ids 1 42 17 9 \
  --generate-steps 4 \
  --output parity-report.json
```

Choose model-specific tolerances and quality gates; never relax a default just
to turn a first-divergence report green.

For a local synthetic smoke path that does not download a model:

```bash
python3 mlx-model-porting/scripts/inspect_model.py \
  tests/fixtures/models/decoder \
  --output /tmp/inspection.json

python3 mlx-model-porting/scripts/make_port_plan.py \
  /tmp/inspection.json \
  --artifact-root tests/fixtures/models/decoder \
  --output /tmp/PORT_PLAN.md
```

The fixture is expected to route to `dense-decoder-transformer`. That result
demonstrates the router and guard path only; it does not demonstrate an
end-to-end checkpoint conversion.

### Real worked port

[`worked-port-qwen2.5-0.5b-instruct`](mlx-model-porting/examples/worked-port-qwen2.5-0.5b-instruct/README.md)
records a complete offline run over the pinned local Qwen2.5-0.5B-Instruct
checkpoint. It includes the portable inspection and capture manifests, complete
weight map, per-rung parity report, exact Torch/standalone-MLX/MLX-LM transcript,
and schema-2 benchmark receipt pointers. All 29 ordered parity rungs and the
eight greedy token IDs matched. Weights and NPZ tensors are excluded.

## Start from an existing MLX project

Use inspector mode for local MLX code, an already-converted checkpoint, or a
served MLX application. It inventories files and proof surfaces without running
the project:

```bash
python3 mlx-model-porting/scripts/inspect_mlx_project.py PROJECT \
  --model LOCAL_MODEL \
  --output inspection.json \
  --markdown MLX_INSPECTION.md
```

This inspector also emits basename-only local references unless
`--include-local-paths` is explicitly requested for private debugging output.

If an inventory is truncated, the inspector blocks a clean or recommendation-
ready conclusion instead of treating unseen files as safe.

## Validate a port

Weight coverage and tensor parity are explicit steps, not side effects of
loading a checkpoint:

```bash
python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source source.json \
  --target target.json \
  --mapping WEIGHT_MAP.json \
  --output weight-map-report.json

python3 mlx-model-porting/scripts/compare_tensors.py \
  source.npz target.npz \
  --mapping mapping.json \
  --atol 1e-5 \
  --rtol 1e-4 \
  --cosine-min 0.99 \
  --output parity.json
```

Tolerance and quality gates must be chosen for the model and task. Passing a
generic tensor threshold alone is not sufficient for language, vision, audio,
speech, diffusion, or streaming correctness.

## Request optimization advice

Supply a `TargetProfile` when asking for version- or workload-sensitive advice:

```bash
python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json \
  --target-profile target-profile.json \
  --objective peak-memory \
  --output recommendations.json \
  --markdown OPTIMIZATIONS.md
```

No current claim has an effective range. Inspect the generated catalogue's
withholding reasons before designing a new measurement:

```python
import json

catalog = json.load(open("mlx-model-porting/assets/effective_claims.json"))
held = {
    row["method_id"]: row["withheld_reasons"]
    for row in catalog["claims"]
    if row["effective_range"] is None
}
print(json.dumps(held, indent=2))
```

Any future promotion must bind the candidate receipt SHA-256, model and source
revisions, target and workload descriptors, enabled methods, metric, experiment
contract, baseline receipt, aggregates, every measured result and raw-output
digest, and the quality result digest. It must also carry an external signature
verified against an out-of-repository trust anchor. A copied digest, empty
profile, or merely similar Mac cannot unlock a number.

The advisor separates results into five controlled buckets:

1. `validated-locally`;
2. `validated-source-theory`;
3. `benchmark-required`;
4. `experimental-approach` — requires explicit user opt-in;
5. `rejected-do-not-use`.

Numeric authority comes from the generated
[`effective_claims.json`](mlx-model-porting/assets/effective_claims.json), not
from prose or an unverified receipt sidecar. All current source-reported ranges
remain withheld observations: the existing `TargetProfile` cannot encode their
complete benchmark scope closely enough to make them profile-eligible.
Source-reported numbers are never advisor-visible until an equivalent local
target-workload benchmark passes the promotion contract.
Twelve local receipts remain performance observations. The generic external
command and legacy MLX-LM lanes do not independently attest the executed
dependency bytes and model/workload semantics. The narrow repository-owned
`attested-mlx-port-wall-time` adapter retains runner, dependency, challenge,
and output evidence for its exact Qwen workload, establishing internal
consistency and reproducibility-on-request. It does not close the authenticity
boundary: SHA-256 is a digest, not a signature, and no external signer or trust
root exists today. The raw `1.8122003933x` inverse-wall-time ratio is one reproducible local-run observation, not a claim of reliable speedup. It covers load plus
six captured greedy tokens and is not an effective range, portable guarantee,
or pure-decode claim.
Compound numbers are withheld unless compatible steps were validated together
with unique evidence lineage; multiplying per-step ceilings is never a measured
claim.

## Evidence and benchmark boundaries

[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) is generated from the canonical
[`sources.yaml`](mlx-model-porting/assets/sources.yaml). Review depth means:

- `synthesized`: directly informed a rule, runbook, or registry decision;
- `screened`: relevant material and limits were reviewed;
- `indexed`: catalogued for later review, with no implied support.

Support scope and review depth are separate from benchmark promotion. A pinned
third-party implementation can support an implementation path while still
requiring local parity and workload measurement.

Supported guidance status is also evidence-gated: native methods require
classified official MLX evidence, Apple-project methods require classified
official-project source, and third-party ports require a synthesized pinned or
locally reproduced implementation reference.

The benchmark truth is generated in
[`BENCHMARK_REPORT.md`](mlx-model-porting/assets/BENCHMARK_REPORT.md). The current
13-receipt set contains 12 observations, 0 promotion-ready receipts, and 1
rejected receipt. All ten effective claims are withheld.

Measurement validation has two generic runner lanes. `mlx-lm-generate` retains
its token-throughput observations. `external-command-wall-time` covers other
architecture families with direct, no-shell argv: a digest-pinned Python runner
at argv position 1, `models.target.id`, `models.target.revision`, checked-in
workload inputs, and semantic `variant_config` must all be bound through an
exact argv template. The resolved interpreter and sanitized ambient environment
are hashed into the target identity, and the runner starts with `-I -B` to
exclude current-directory, user-site, and environment-injected imports. Static
symlink components are rejected and the quality contract is frozen before
execution. Its only performance metric is
`wall_seconds`, measured by the parent `benchmark_command.py` process and
checked against the complete bounded schema-1 raw report. Every measured run
must recreate the exact label-owned quality output; values printed by the model
command are never accepted as metrics. This generic lane is observation-only:
a digest does not prove that arbitrary runner code used its bound arguments.
The MLX-LM lane is also observation-only until its imported package bytes and
per-run model output are independently bound. The separate
`attested-mlx-port-wall-time` adapter is intentionally Qwen/workload-specific:
it binds a fresh parent challenge, reviewed runner bytes, retained loaded
dependencies, model/workload identity, and every output. Those bindings support
reproduction but cannot set `execution_attested=true` without a signature from
a protected Apple-Silicon signer verified against an out-of-repository trust
anchor. That external trust root is future work.

## Release gates

Run these checks from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting
python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting
python3 mlx-model-porting/scripts/knowledge_curator.py --check-backlog
python3 mlx-model-porting/scripts/validate_benchmarks.py check
python3 mlx-model-porting/scripts/generate_claim_catalog.py --check
python3 mlx-model-porting/scripts/generate_evidence_index.py --check
python3 mlx-model-porting/scripts/generate_site_data.py --check
node --check site/data.js
node --check site/app.js
python3 mlx-model-porting/scripts/manifest.py check
git diff --check
```

These are offline method and artifact gates. They do not replace a real
Apple-Silicon port, task-quality evaluation, or target-workload benchmark.

## Extending the skill

Do not hand-edit generated reports or indexes. Add or change the canonical
registry, runbook, receipt, or version input, regenerate in dependency order,
and review the resulting diff. The ownership table and exact regeneration
commands are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Research automation is review-only: it may collect and rank candidates and
prepare a change, but it may not promote guidance or auto-merge research-derived
recommendations.

## Versioning

The 0.6.1 release snapshot is dated 2026-07-14. The package version is recorded
in [`VERSION`](VERSION) and the skill frontmatter; release changes are recorded
in [`CHANGELOG.md`](CHANGELOG.md).
