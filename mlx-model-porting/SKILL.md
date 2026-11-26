---
name: mlx-model-porting
description: Guides and validates architecture-aware ports of PyTorch/Hugging Face models to Apple MLX, inspects existing local MLX projects, and plans evidence-gated optimizations for Apple Silicon. Use when the user asks to run, port, convert, inspect, quantize, benchmark, or fix a model (LLM, VLM, audio/TTS/ASR, diffusion, SSM, MoE) for MLX, MLX-LM, MLX-VLM, MLX-Audio, or a Mac - e.g. "port this HF model to my Mac", "inspect this MLX app", "run Qwen on Apple Silicon", "convert these safetensors to MLX", "make this faster on my M3", "fix NaN in my MLX port", "speed up prefill / KV cache / speculative decoding", "publish an MLX checkpoint". Also use mid-task when a config.json, safetensors index, weight-shape or tokenizer mismatch, or Metal kernel question appears. Do not use for CUDA-only optimization, non-Apple hardware targets, or general PyTorch/ML questions with no MLX or Apple Silicon connection.
license: Apache-2.0
compatibility: Execution and performance validation require an Apple Silicon Mac with a supported MLX installation. Planning and static inspection can run elsewhere. Python 3.10+ and git are recommended; NumPy is required only for tensor parity and huggingface-hub only for explicitly enabled network intake (see requirements-tools.txt). Network access is optional and must be explicitly enabled.
metadata:
  author: mlx-porting-skill
  version: "0.7.0"
  last-reviewed: "2026-07-23"
---

# MLX model porting and optimization

Research refresh: **2026-11-05** (unreleased; the metadata date tracks the
published release).

## Mission

Produce or inspect a **correct, reproducible, architecture-aware MLX implementation**. Correctness before speed. Every speed or memory claim must name hardware, software versions, workload, baseline, and quality gate.

Six families have scaffolds (MoE/SSM synthetic, others runbook-guided); four
have worked packets under [examples/](examples/porting-patterns.md): Qwen2.5,
BGE, t5-small, HuBERT. Exact output is the built-in metric.

## Trigger map

| Signal | Load |
| --- | --- |
| Port/convert/run, `config.json`, safetensors, model directory, or Hub id. | [intake](references/intake-and-routing.md), Workflow 1-6; [worked chain](examples/worked-port-qwen2.5-0.5b-instruct/README.md) for dense decoders. |
| Existing MLX project, app, or completed port. | [inspector mode](references/inspector-mode.md), [inspect_mlx_project.py](scripts/inspect_mlx_project.py) |
| Model capability fit or model-specific advice. | [model advisor playbook](references/model-advisor-playbook.md) |
| Architecture support or runbook selection. | [model support map](references/model-support-map.md), Workflow 4 |
| Scaffold or capture/parity mode selection. | Workflow 4 and 6; check the exact supported config. |
| NaN/Inf, tensor drift, parity failure, or garbage output. | [failure atlas](references/failure-atlas.md) |
| Weight mapping, conversion, or shape transforms. | [core porting method](references/porting-core.md) |
| Performance request without a profile or baseline. | [benchmarking](references/benchmarking.md) |
| Combining optimizations or expected compound gains. | [compound stacks](references/compound-stacks.md) |
| Cache, long context, recurrent state, or attention memory. | [attention and KV cache](references/attention-and-kv.md) |
| Quantization or "4-bit". | [guide](references/quantization.md), [quality gate](references/quantization-quality-gate.md) |
| Structured local optimization sweep. | [loop](references/optimization-loop.md) |
| Decoding, serving, speculation, batching, or streaming. | [decoding and serving](references/decoding-and-serving.md) |
| Compilation, custom kernels, graph capture, Metal, or fusion. | [compile and kernels](references/compile-and-kernels.md) |
| Recent MLX APIs, M5/NAX, FP4/FP8, FFT, interop, or distributed backends. | [runtime compatibility](references/runtime-compatibility.md); verify target profile and API/device path. |
| Recent papers, compressed attention, or kernel search. | [November research update](references/research-updates-2026-11.md); apply its workload and state gates. |
| Publish, release, model card, provenance, or licensing. | [packaging and publication](references/packaging-and-publication.md) |
| Deep model-specific research or optimization ideas. | [hypothesis-led learning](references/hypothesis-led-learning.md) |
| Vision-language, multimodal, or image+text (VLM) input appears. | [multimodal/omni runbook](references/runbook-multimodal-omni.md) |
| Diffusion, flow-matching, or image/video generation appears. | [diffusion/flow runbook](references/runbook-diffusion-flow.md) |
| Text-to-speech, vocoder, or audio generation appears. | [flow-TTS](references/runbook-flow-tts.md), [autoregressive audio](references/runbook-autoregressive-audio.md) |
| ASR, transcription, or streaming speech appears. | [ASR](references/runbook-asr.md), [streaming speech](references/runbook-streaming-speech.md) |
| Sparse mixture-of-experts or top-k expert routing appears. | [MoE runbook](references/runbook-moe-transformer.md) |
| Selective state-space, Mamba, or linear-attention hybrid appears. | [SSM/hybrid runbook](references/runbook-ssm-hybrid.md) |
| Graph, GNN, message passing, node/edge features, or sparse graph workload appears. | [graph message passing runbook](references/runbook-graph-message-passing.md) |
| Classic CV detection, segmentation, keypoints, depth, OCR, or non-generative vision appears. | [non-generative CV runbook](references/runbook-non-generative-cv.md) |
| Time-series, forecasting, tabular sequence, anomaly detection, or temporal model appears. | [time-series forecasting runbook](references/runbook-time-series-forecasting.md) |

Re-consult the map on a new config, parity failure, performance complaint, or
publish request.

## Non-negotiable rules

1. **Do not execute untrusted model code during intake.** Inspect JSON, safetensors headers, source files, and licenses statically. Treat `auto_map`, custom modules, install hooks, and `trust_remote_code` as review gates.
2. **Pin the source.** Record repository, revision, model files, tokenizer/processor revision, license, and checksum or artifact manifest.
3. **Build a source oracle before porting.** Freeze deterministic fixtures and capture intermediate tensors at meaningful boundaries.
4. **Port the smallest eager path first.** No quantization, compilation, custom kernels, batching, or speculative decoding until basic parity passes.
5. **Change one optimization dimension at a time.** Keep a measurement and rollback record.
6. **Prefer native MLX operations.** Try built-in fused operations, layout changes, cache design, and `mx.compile` before a custom Metal kernel.
7. **Do not translate CUDA folklore mechanically.** A CUDA technique is only a research candidate until its Metal/MLX bottleneck and implementation are demonstrated.
8. **Never hide quality regressions behind throughput.** For audio, language, vision, and generative models, use task-specific quality checks in addition to tensor tolerances.
9. **Do not publish converted weights without license and provenance checks.** Preserve the original model card, attribution, generation config, tokenizer/processor files, and conversion recipe.
10. **Daily research automation is review-only.** It may collect and rank candidates, but must not silently rewrite runbooks or merge recommendations. Use the promotion-review ledger to separate review-ready findings from validation backlog and rejected leads.
11. **Experimental approaches require explicit opt-in.** Label unvalidated contributor, blog, paper, or repository learnings as experimental, state the missing validation gate, and ask before executing them: “This is an experimental approach. Do you want to try it?” Continue only if the user says to try it.

## Workflow

### 1. Inspect and classify

For source models, run `scripts/inspect_model.py`, then `scripts/recommend_optimizations.py`, then `scripts/make_port_plan.py --artifact-root MODEL --recommendations ...`. An actionable plan re-inspects those bytes and recomputes the full recommendation report before advice; a blocked inspection yields only a remediation plan, and no override or registry read bypasses its blockers. An override may reorder a hybrid route but must preserve every routed family, runbook, and trait. For existing MLX projects or running ports, run `scripts/inspect_mlx_project.py` and read [inspector mode](references/inspector-mode.md) first.

Read [intake and routing](references/intake-and-routing.md). Confirm source, risk, Mac, memory, performance, quality; record ambiguity in `PORT_PLAN.md`. For advice, read [model advisor playbook](references/model-advisor-playbook.md) and separate results into its five controlled advisor buckets. Numeric output may come only from `assets/effective_claims.json`; missing gates withhold the number.

Use `--target-profile target-profile.json` for runtime-dependent methods.
Missing, old, or ambiguous versions/chips hold execution even with
`--allow-blocked`. These prerequisites do not authorize numeric claims or
replace API, device, parity, and workload gates.

### 2. Select the closest proven MLX reference

Consult [model support map](references/model-support-map.md) and `assets/architectures.yaml`. Prefer official MLX, Apple projects, pinned third-party MLX evidence, paper-only candidates, then new code; keep support scope explicit and verify config/layout.

### 3. Establish the source oracle

Run `scripts/capture_oracle.py` against the pinned local Hugging Face model before the MLX graph, then follow [parity and testing](references/parity-and-testing.md). It records inputs, embeddings, blocks, final norm, logits, hookable branches, and greedy IDs in a bounded NPZ plus manifest.

### 4. Implement the minimal eager MLX graph

Read [core porting method](references/porting-core.md); choose via the trigger map and the family records in `assets/architectures.yaml`. Load every runbook a hybrid route returns; the registry is the full inventory—never substitute an abbreviated list.

Scaffold an unblocked dense decoder, sparse-MoE decoder, BERT encoder, T5 encoder-decoder, HuBERT/Wav2Vec2 acoustic encoder, or opt-in `minimal_selective` SSM:

```bash
python3 scripts/scaffold_port.py inspection.json --artifact-root MODEL --output mlx_port
```

It re-inspects and fails closed; review unsupported config, never patch around a generator blocker.

Start eager: FP, batch one unless intrinsic, no compile/kernels, state/cache, assertions, reversible map.

### 5. Convert weights deterministically

Draft with `scripts/convert_checkpoint.py --emit-draft-map`, resolve the
schema-2 `WEIGHT_MAP`, validate with `scripts/validate_weight_map.py`, then
convert. Reject shard, coverage, shape, draft, or unresolved gaps; never mask
exceptions.

### 6. Pass the parity ladder

After conversion, `scripts/run_parity.py --mode MODE` runs source/MLX capture
and stops at the first failed rung. Modes: `dense-decoder` (default), `encoder`,
`encoder-decoder`, `ssm`, `asr`. `capture_mlx.py` retains captures,
`compare_tensors.py` extras, `_capture_common.py` bounded rules.

When parity fails, use [failure atlas](references/failure-atlas.md); do not optimize a failing graph.

### 7. Profile before choosing optimizations

Read [benchmarking](references/benchmarking.md). Separate prefill, decode, postprocess, compile, memory, movement, sync, Python overhead; use `scripts/benchmark_command.py`.

### 8. Apply the optimization ladder

Use guides: [C](references/compile-and-kernels.md),[KV](references/attention-and-kv.md),[S](references/decoding-and-serving.md),[Q](references/quantization.md),[T](references/training-and-finetuning.md),[CS](references/compound-stacks.md).

Tiers: `assets/recommendation-taxonomy.yaml`; for an experimental approach, state the gate and use the rule-11 opt-in prompt.

Post-parity: one dimension; Metal only after proven bottleneck. Record hypothesis, gates, metrics, decision.

Compare native baselines at the same precision; measure prefill, committed-token
decode, peak transient memory, and cold/warm behavior separately.
`optimize_port.py` covers only its declared affine recipes, not the new methods.

### 9. Package and publish

Follow [packaging and publication](references/packaging-and-publication.md). Include source, conversion, versions, artifacts, quantization, smoke, benchmarks, limits, license/attribution; no unsupported “faster” wording.

### 10. Return an engineering report

Summarize architecture/runbook, source/evidence, implementation/weights, parity, metrics, optimizations, risks, commands, and artifacts. Use `assets/`.

## When to stop

Stop when license, remote code, unresolved parity, regressions, missing kernel fallback/tests, or missing performance metadata blocks the result.

## Maintenance

[maintenance](references/maintenance-and-provenance.md), [hypothesis-led learning](references/hypothesis-led-learning.md), [research loop](references/deep-research-loop.md), [evidence graph](references/evidence-graph.md). Run `scripts/audit_skill.py --strict` and `scripts/validate_sources.py` before release.
Keep `tests/test_scenarios.py` at full family coverage when `assets/architectures.yaml` changes.
