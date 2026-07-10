---
name: mlx-model-porting
description: Guides and validates architecture-aware ports of PyTorch/Hugging Face models to Apple MLX, inspects existing local MLX projects, and plans evidence-gated optimizations for Apple Silicon. Use when the user asks to run, port, convert, inspect, quantize, benchmark, or fix a model (LLM, VLM, audio/TTS/ASR, diffusion, SSM, MoE) for MLX, MLX-LM, MLX-VLM, MLX-Audio, or a Mac - e.g. "port this HF model to my Mac", "inspect this MLX app", "run Qwen on Apple Silicon", "convert these safetensors to MLX", "make this faster on my M3", "fix NaN in my MLX port", "speed up prefill / KV cache / speculative decoding", "publish an MLX checkpoint". Also use mid-task when a config.json, safetensors index, weight-shape or tokenizer mismatch, or Metal kernel question appears. Do not use for CUDA-only optimization, non-Apple hardware targets, or general PyTorch/ML questions with no MLX or Apple Silicon connection.
license: Apache-2.0
compatibility: Execution and performance validation require an Apple Silicon Mac with a supported MLX installation. Planning and static inspection can run elsewhere. Python 3.10+ and git are recommended; NumPy is required only for tensor parity and huggingface-hub only for explicitly enabled network intake (see requirements-tools.txt). Network access is optional and must be explicitly enabled.
metadata:
  author: mlx-porting-skill
  version: "0.4.0"
  last-reviewed: "2026-07-10"
---

# MLX model porting and optimization

## Mission

Produce or inspect a **correct, reproducible, architecture-aware MLX implementation**. Correctness comes before speed. Every speed or memory claim must name the hardware, software versions, workload, baseline, and quality gate.

## When to use this skill

- Use when porting, converting, or running a PyTorch or Hugging Face model on MLX, MLX-LM, MLX-VLM, MLX-Audio, or Apple Silicon.
- Use when making an MLX model faster, smaller, more memory-efficient, or more suitable for a specific Mac.
- Use when debugging parity, NaN/Inf, shape, tokenizer, preprocessing, or garbage-output issues in an MLX port.
- Use when inspecting an existing local MLX project, served MLX app, or already-converted checkpoint to find proof gaps, improvement paths, or contribution candidates.
- Use when quantizing, packaging, publishing, or validating provenance for MLX checkpoints.
- Use when choosing optimization paths for a model family, architecture runbook, cache design, serving mode, or benchmark plan.
- Do not use when the target is CUDA-only, non-Apple hardware, or a deployment path with no MLX/Apple Silicon connection.
- Do not use when the question is general PyTorch, machine learning, or model theory without a concrete MLX or Apple Silicon target.
- Do not use when the task is training from scratch and is unrelated to porting, conversion, inference, checkpoint packaging, or MLX validation.

## Trigger map

| Signal | Load |
| --- | --- |
| User pasted a `config.json`, safetensors index, model directory, or Hugging Face repo id. | [intake and routing](references/intake-and-routing.md) plus [inspect_model.py](scripts/inspect_model.py) |
| User points at an existing local MLX project, running MLX app, or completed MLX port. | [inspector mode](references/inspector-mode.md) plus [inspect_mlx_project.py](scripts/inspect_mlx_project.py) |
| User asks "what can I do with this model?", asks for capability fit, or wants model-specific advice. | [model advisor playbook](references/model-advisor-playbook.md) |
| A known architecture family needs the right runbook. | [model support map](references/model-support-map.md), then the architecture table in Workflow step 4 below |
| NaN, Inf, cosine-similarity drift, parity failure, or garbage output appears versus the source. | [failure atlas](references/failure-atlas.md) |
| Weight conversion, key mapping, tensor rename, transpose, reshape, split, merge, or shape transform is in scope. | [core porting method](references/porting-core.md) |
| The user says "make it faster" but no profile, workload, or baseline exists yet. | [benchmarking](references/benchmarking.md) |
| Speedup plan, how techniques combine, or expected compound gains. | [compound stacks](references/compound-stacks.md) |
| KV cache, long context, recurrent state, attention memory, or prefill/decode memory is the bottleneck. | [attention and KV cache](references/attention-and-kv.md) |
| Quantization, "fit in 16GB", "4-bit", mixed precision, or memory reduction is requested. | [quantization](references/quantization.md) |
| Decoding, serving, speculative decoding, batching, streaming, or API runtime behavior is requested. | [decoding and serving](references/decoding-and-serving.md) |
| Compile behavior, `mx.compile`, custom kernel, graph capture, Metal, or operation fusion comes up. | [compile and kernels](references/compile-and-kernels.md) |
| Publish, release, checkpoint conversion, model card, provenance, or license packaging is requested. | [packaging and publication](references/packaging-and-publication.md) |
| The user asks for "50-100 optimization ideas", a deep model-specific hunt, or research-backed candidates. | [hypothesis-led learning](references/hypothesis-led-learning.md) |
| Graph, GNN, message passing, node/edge features, or sparse graph workload appears. | [graph message passing runbook](references/runbook-graph-message-passing.md) |
| Classic CV detection, segmentation, keypoints, depth, OCR, or non-generative vision appears. | [non-generative CV runbook](references/runbook-non-generative-cv.md) |
| Time-series, forecasting, tabular sequence, anomaly detection, or temporal model appears. | [time-series forecasting runbook](references/runbook-time-series-forecasting.md) |

Re-consult this map whenever a new signal appears mid-session: a pasted config, a failed parity stage, a performance complaint, a publish request.

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
10. **Daily research automation is review-only.** It may collect and rank candidates, but must not silently rewrite runbooks or merge recommendations. Use the promotion-review ledger to separate findings ready for skill-update review from validation backlog and rejected leads.
11. **Experimental approaches require explicit opt-in.** Label unvalidated contributor, blog, paper, or repository learnings as experimental approaches, state the missing validation gate, and ask before helping execute them: “This is an experimental approach. Do you want to try it?” Continue only if the user explicitly says to try it.

## Workflow

### 1. Inspect and classify

For source models, run `scripts/inspect_model.py`, then `scripts/recommend_optimizations.py`, then `scripts/make_port_plan.py --artifact-root MODEL --recommendations ...`. An actionable plan re-runs static inspection against those local bytes and recomputes the complete recommendation report before embedding advice. A blocked inspection may produce only a remediation plan; neither a family override nor direct registry reads may bypass its blockers. A family override may reorder an inspected hybrid route but must preserve every routed family, runbook, and trait. For existing MLX projects or already-running ports, run `scripts/inspect_mlx_project.py` and read [inspector mode](references/inspector-mode.md) before changing code.

Read [intake and routing](references/intake-and-routing.md). Confirm source, risk, Mac, memory, performance, quality; record ambiguity in `PORT_PLAN.md`. For advice, read [model advisor playbook](references/model-advisor-playbook.md) and separate results into all five controlled advisor buckets: validated locally, validated by source or theory, benchmark required, experimental approach, and rejected / do not use. Numeric output may come only from `assets/effective_claims.json`; missing gates withhold the number.

### 2. Select the closest proven MLX reference

Consult [model support map](references/model-support-map.md) and `assets/architectures.yaml`. Prefer official MLX, Apple-maintained projects, pinned third-party MLX implementation evidence, paper-only research candidates, then new code; keep the support scope explicit and verify config/layout.

### 3. Establish the source oracle

Run `scripts/capture_oracle.py` against the pinned local Hugging Face model before implementing the MLX graph, then follow [parity and testing](references/parity-and-testing.md) for fixtures, tolerances, core cases, and additional state/cache or modality-specific captures. The tool records input IDs, embeddings, every decoder block, final norm, logits, hookable attention/MLP branches, and deterministic greedy continuation IDs in a bounded NPZ plus a content-addressed manifest.

### 4. Implement the minimal eager MLX graph

Read [core porting method](references/porting-core.md); choose via the trigger map and the controlled family records in `assets/architectures.yaml`. Load every runbook returned by a hybrid route. The registry is the complete runbook inventory; do not substitute an abbreviated parallel list.

Start eager: FP, batch one unless intrinsic, no compile/kernels, state/cache, assertions, reversible map.

### 5. Convert weights deterministically

Map keys/shapes, transform, dtype, ties; validate with `scripts/validate_weight_map.py`. `inspection.json` can be `--source`.

No `strict=False` masking; classify every exception.

### 6. Pass the parity ladder

Use `scripts/compare_tensors.py`: config/preprocess, weights/shapes, primitive/block, intermediates, end-to-end, quality, cache/state, boundary/long/streaming/batch.

When parity fails, use [failure atlas](references/failure-atlas.md); do not optimize a failing graph.

### 7. Profile before choosing optimizations

Read [benchmarking](references/benchmarking.md). Separate prefill, decode, postprocess, compile, memory, movement, sync, Python overhead; use `scripts/benchmark_command.py`.

### 8. Apply the optimization ladder

Use guides: [C](references/compile-and-kernels.md),[KV](references/attention-and-kv.md),[S](references/decoding-and-serving.md),[Q](references/quantization.md),[T](references/training-and-finetuning.md),[CS](references/compound-stacks.md).

Tiers: `assets/recommendation-taxonomy.yaml`; experimental approach: state gate and ask: “This is an experimental approach. Do you want to try it?” Continue after opt-in.

Post-parity: one dimension; Metal only after proven bottleneck. Record hypothesis, gates, metrics, decision.

### 9. Package and publish

Follow [packaging and publication](references/packaging-and-publication.md). Include source, conversion, versions, artifacts, quantization, smoke, benchmarks, limits, license/attribution, and no unsupported “faster” wording.

### 10. Return an engineering report

Summarize architecture/runbook, source/evidence, implementation/weights, parity, metrics, optimizations, risks, commands, and artifacts. Use `assets/`.

## When to stop

Stop when license, remote code, unresolved parity, regressions, missing kernel fallback/tests, or missing performance metadata blocks the result.

## Maintenance

For maintenance, read [mnt](references/maintenance-and-provenance.md), [hyp](references/hypothesis-led-learning.md), [deep](references/deep-research-loop.md). Use `scripts/nightly_knowledge_curator.py` for receipts, delta, packets; use hypothesis-led for 50-100/learn-flow/public-port/kernel/2x/5x targets. Keep flags/receipts/gates/matrices/dossiers/ledger in [deep](references/deep-research-loop.md); run `scripts/audit_skill.py --strict` and `scripts/validate_sources.py` before distribution.
When `assets/architectures.yaml` changes, keep the golden scenario gate in `tests/test_scenarios.py` at full family coverage.
