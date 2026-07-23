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

## Mission

Produce or inspect a **correct, reproducible, architecture-aware MLX implementation**. Correctness before speed. Every speed or memory claim must name hardware, software versions, workload, baseline, and quality gate.

Six families have scaffolds (MoE/SSM synthetic, others runbook-guided); four
have worked packets under [examples/](examples/porting-patterns.md): Qwen2.5,
BGE, t5-small, HuBERT. Exact output is the built-in metric.

## When to use this skill

Port, convert, run, inspect, quantize, package, or publish a PyTorch/Hugging Face model or MLX project on Apple Silicon, or fix any parity, NaN/Inf, shape, tokenizer, preprocessing, output, performance, memory, cache, serving, benchmark, or provenance issue in a port. Not for CUDA/non-Apple targets, ML theory without an MLX target, or training from scratch.

## Trigger map

| Signal | Load |
| --- | --- |
| Port/convert/run request, `config.json`, safetensors index, model directory, or Hub id. | [intake](references/intake-and-routing.md), then Workflow 1-6; [worked chain](examples/worked-port-qwen2.5-0.5b-instruct/README.md) for dense decoders, routed runbooks otherwise. |
| User points at an existing local MLX project, running MLX app, or completed MLX port. | [inspector mode](references/inspector-mode.md) plus [inspect_mlx_project.py](scripts/inspect_mlx_project.py) |
| User asks "what can I do with this model?", asks for capability fit, or wants model-specific advice. | [model advisor playbook](references/model-advisor-playbook.md) |
| A known architecture family needs the right runbook. | [model support map](references/model-support-map.md), then the architecture table in Workflow step 4 below |
| Dense decoder, BERT encoder, T5 encoder-decoder, HuBERT/Wav2Vec2 acoustic encoder, sparse MoE, or selective SSM. | Use `scaffold_port.py`; select capture/parity mode `dense-decoder` (default), `encoder`, `encoder-decoder`, `asr`, or `ssm`. |
| NaN, Inf, cosine-similarity drift, parity failure, or garbage output appears versus the source. | [failure atlas](references/failure-atlas.md) |
| Weight conversion, key mapping, tensor rename, transpose, reshape, split, merge, or shape transform is in scope. | [core porting method](references/porting-core.md) |
| The user says "make it faster" but no profile, workload, or baseline exists yet. | [benchmarking](references/benchmarking.md) |
| Speedup plan, how techniques combine, or expected compound gains. | [compound stacks](references/compound-stacks.md) |
| KV cache, long context, recurrent state, attention memory, or prefill/decode memory is the bottleneck. | [attention and KV cache](references/attention-and-kv.md) |
| Quantization or "4-bit". | [guide](references/quantization.md), [quality gate](references/quantization-quality-gate.md) |
| Structured local optimization sweep. | [loop](references/optimization-loop.md) |
| Decoding, serving, speculative decoding, batching, streaming, or API runtime behavior is requested. | [decoding and serving](references/decoding-and-serving.md) |
| Compile behavior, `mx.compile`, custom kernel, graph capture, Metal, or operation fusion comes up. | [compile and kernels](references/compile-and-kernels.md) |
| Publish, release, checkpoint conversion, model card, provenance, or license packaging is requested. | [packaging and publication](references/packaging-and-publication.md) |
| The user asks for "50-100 optimization ideas", a deep model-specific hunt, or research-backed candidates. | [hypothesis-led learning](references/hypothesis-led-learning.md) |
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

### 9. Package and publish

Follow [packaging and publication](references/packaging-and-publication.md). Include source, conversion, versions, artifacts, quantization, smoke, benchmarks, limits, license/attribution; no unsupported “faster” wording.

### 10. Return an engineering report

Summarize architecture/runbook, source/evidence, implementation/weights, parity, metrics, optimizations, risks, commands, and artifacts. Use `assets/`.

## When to stop

Stop when license, remote code, unresolved parity, regressions, missing kernel fallback/tests, or missing performance metadata blocks the result.

## Maintenance

Maintenance: [maintenance](references/maintenance-and-provenance.md), [hypothesis-led learning](references/hypothesis-led-learning.md), [research loop](references/deep-research-loop.md). Run `scripts/audit_skill.py --strict` and `scripts/validate_sources.py` before distribution.
When `assets/architectures.yaml` changes, keep the golden scenario gate in `tests/test_scenarios.py` at full family coverage.
