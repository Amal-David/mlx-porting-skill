---
name: mlx-model-porting
description: Port, validate, quantize, benchmark, and optimize PyTorch or Hugging Face language, vision, audio, speech, codec, diffusion, and hybrid models for Apple MLX. Use when adding a new architecture to MLX, MLX-LM, MLX-VLM, or MLX-Audio; converting weights; fixing an MLX port; improving Apple Silicon latency, throughput, memory, streaming, attention, KV cache, speculative decoding, or quantization; or publishing a reproducible MLX checkpoint.
license: Apache-2.0
compatibility: Execution and performance validation require an Apple Silicon Mac with a supported MLX installation. Planning and static inspection can run elsewhere. Python 3.10+ and git are recommended; network access is optional and must be explicitly enabled.
metadata:
  author: mlx-porting-skill
  version: "0.1.0"
  last-reviewed: "2026-06-27"
---

# MLX model porting and optimization

## Mission

Produce a **correct, reproducible, architecture-aware MLX implementation**. Correctness comes before speed. Every speed or memory claim must name the hardware, software versions, workload, baseline, and quality gate.

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

Run:

```bash
python3 scripts/inspect_model.py MODEL_OR_DIRECTORY --output inspection.json
python3 scripts/make_port_plan.py inspection.json --output PORT_PLAN.md
python3 scripts/recommend_optimizations.py inspection.json --markdown OPTIMIZATIONS.md
```

Read [intake and routing](references/intake-and-routing.md). Confirm:

- task and model domain;
- architecture family and recurrent/cache state;
- source framework and custom operations;
- parameter count, dtypes, tied/shared weights, shards, and adapters;
- preprocessing, tokenization, sampling, and postprocessing;
- license and remote-code risk;
- target Mac, memory budget, latency/throughput objective, and quality objective.

Do not begin implementation if the architecture, source revision, or evaluation target remains ambiguous. Record uncertainties in `PORT_PLAN.md` rather than guessing.

### 2. Select the closest proven MLX reference

Consult [model support map](references/model-support-map.md) and `assets/architectures.yaml`.

Use this order:

1. official MLX / MLX-LM implementation;
2. active MLX-VLM or MLX-Audio implementation with tests;
3. active third-party MLX implementation;
4. upstream source implementation plus architecture paper;
5. research prototype requiring a new MLX implementation.

Reuse architectural patterns, not copied assumptions. Verify config semantics and tensor layouts against the pinned source.

### 3. Establish the source oracle

Follow [parity and testing](references/parity-and-testing.md):

- set deterministic seeds and inference mode;
- save exact inputs after preprocessing;
- capture shape/dtype/statistics and selected tensors after embeddings/frontends, every block group, bottleneck, cache update, logits/latents, and decoder/vocoder output;
- save source outputs in portable `.npz`, `.json`, or `.wav` fixtures;
- record tolerances by dtype and operation;
- include at least one minimal, one ordinary, and one boundary case.

### 4. Implement the minimal eager MLX graph

Read [core porting method](references/porting-core.md) and the matching architecture runbook:

- [dense decoder Transformer](references/runbook-decoder-transformer.md)
- [Mixture-of-Experts Transformer](references/runbook-moe-transformer.md)
- [encoder Transformer](references/runbook-encoder-transformer.md)
- [encoder-decoder Transformer](references/runbook-encoder-decoder.md)
- [SSM, recurrent, and hybrid](references/runbook-ssm-hybrid.md)
- [diffusion and flow](references/runbook-diffusion-flow.md)
- [vision-language and omni](references/runbook-multimodal-omni.md)
- [neural audio codec](references/runbook-audio-codec.md)
- [autoregressive audio LM / TTS](references/runbook-autoregressive-audio.md)
- [flow or diffusion TTS](references/runbook-flow-tts.md)
- [vocoder](references/runbook-vocoder.md)
- [ASR](references/runbook-asr.md)
- [streaming speech](references/runbook-streaming-speech.md)
- [separation and enhancement](references/runbook-separation-enhancement.md)

Initial implementation constraints:

- floating-point weights only;
- batch size one unless batching is intrinsic;
- no compile decorator;
- no custom kernels;
- explicit state and cache objects;
- shape assertions around every nontrivial transform;
- reversible weight-map manifest.

### 5. Convert weights deterministically

Create a weight map with source key, target key, source shape, target shape, transform, dtype, and tie/share rule. Validate it:

```bash
python3 scripts/validate_weight_map.py   --source source-manifest.json   --target target-manifest.json   --mapping WEIGHT_MAP.json
```

The `inspection.json` emitted by `inspect_model.py` in step 1 is itself a valid `--source` manifest (its `tensors` list carries the source keys and shapes), so no separate source-manifest export is required for the source side.

Never rely on load-time `strict=False` to conceal missing or extra weights. Categorize every exception as intentionally ignored, generated, shared, or unsupported.

### 6. Pass the parity ladder

Use `scripts/compare_tensors.py` and pass, in order:

1. config and preprocessing parity;
2. weight coverage and transformed-shape parity;
3. single primitive/block parity;
4. staged intermediate parity;
5. end-to-end deterministic parity;
6. task-quality parity;
7. cache/state and incremental-generation parity;
8. boundary, long-input, streaming, and batch parity.

When parity fails, use [failure atlas](references/failure-atlas.md). Do not optimize a failing graph.

### 7. Profile before choosing optimizations

Read [benchmarking](references/benchmarking.md). Separate:

- prefill/encoder/frontend time;
- per-token or per-frame decode time;
- codec/vocoder/postprocess time;
- compilation and first-run cost;
- peak and steady-state memory;
- data movement, synchronization, and Python overhead.

Use:

```bash
python3 scripts/benchmark_command.py --warmup 2 --runs 8 --output benchmark.json -- COMMAND ...
```

### 8. Apply the optimization ladder

Consult `assets/optimization_guidance.yaml`, `assets/recommendation-taxonomy.yaml`, `assets/techniques.yaml`, and load only relevant guides:

1. [MLX runtime, compilation, and kernels](references/compile-and-kernels.md)
2. [attention and KV cache](references/attention-and-kv.md)
3. [decoding and serving](references/decoding-and-serving.md)
4. [quantization](references/quantization.md)
5. [training and fine-tuning](references/training-and-finetuning.md)

Before recommending or trying an optimization for a specific model, classify each option:

- **validated locally**: reproduced by a local test, benchmark, or fixture for this skill or the current port;
- **validated by source or theory**: backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model;
- **benchmark-required**: safe to try after parity, but no speedup or memory number may be claimed until measured on the target Mac and workload;
- **experimental approach**: promising from contributor, blog, repository, or research-loop evidence, but not promotion-ready for supported guidance;
- **rejected/do not use**: incompatible, unsafe, contradicted, license-blocked, CUDA-only, or superseded.

For an experimental approach, run a short plan session first. Say why it is promising, why it is experimental, the required validation and rollback gate, then ask: “This is an experimental approach. Do you want to try it?” If the user does not explicitly agree to try it, keep the approach in the report/backlog and continue only with validated or benchmark-required guidance.

Default order:

1. remove unintended evaluations and host transfers;
2. correct dtype and tensor layout;
3. use native fused operations and fast SDPA;
4. reduce allocations and make state/cache updates explicit;
5. compile stable regions and control recompilation;
6. chunk prefill/frontends or stream activations where the architecture permits;
7. stream repeated-block weights only for memory-bound diffusion/flow-style ports after eager parity;
8. optimize KV/cache policy and batching;
9. quantize weights, then KV/state if justified;
10. add speculative or multi-token decoding only for compatible autoregressive paths;
11. write a custom Metal kernel only after profiling proves a remaining kernel bottleneck.

For every change, record hypothesis, diff, correctness result, benchmark result, memory result, quality result, and keep/revert decision.

### 9. Package and publish

Follow [packaging and publication](references/packaging-and-publication.md). Include:

- pinned source and conversion command;
- compatible MLX and library versions;
- model/config/tokenizer/processor files;
- exact quantization recipe and exclusions;
- deterministic smoke test;
- benchmark protocol and raw results;
- known limitations;
- original license and attribution;
- no unsupported “faster” or “lossless” wording.

### 10. Return an engineering report

The final response must summarize:

- architecture and selected runbook;
- source revision and evidence references;
- implementation and weight-map status;
- parity matrix;
- baseline and optimized metrics;
- accepted and rejected optimizations;
- remaining risks and unsupported paths;
- reproducible commands and artifact locations.

Use the templates in `assets/`.

## When to stop

Stop and report rather than improvising when:

- the source license prohibits the requested distribution;
- required source behavior only exists in unreviewed remote code;
- parity cannot be localized after the staged checks;
- an optimization improves a microbenchmark but worsens end-to-end latency, memory, or quality;
- a custom kernel lacks a portable fallback or adequate tests;
- hardware, MLX version, workload, or quality target is missing from a performance claim.

## Maintenance

For source review, security, daily candidates, and broad multi-agent research loops, read [maintenance and provenance](references/maintenance-and-provenance.md) and [deep research loop](references/deep-research-loop.md). Use research-loop gap hints when the worker roster should be selected dynamically from the objective rather than fixed config order, subagent handoff packets and campaign receipts when delegating real researchers, `scripts/run_research_campaign.py` when an explicit local researcher command should execute a campaign wave by wave, `--follow-next-wave-scaffold` when that local campaign should run bounded adaptive follow-up waves after ingest, `--ingest-subagent-results` when separately spawned researchers have written their result files, worker-authored blog ingestion when researchers write to `MLX_RESEARCH_BLOG_PATH`, `--require-worker-blog-contract` when incomplete worker blogs should fail a campaign after receipts are written, `--require-explicit-sampling-receipts` when matched planned targets must be backed by worker-declared target receipts, `--iterations` when held or needs-validation findings should drive follow-up sampling, post-ingest `next_wave_scaffold` receipts when an external dispatcher needs to create the next adaptive wave from current gap hints, `--until-review-gate` when that loop should stop after enough sampled evidence is returned, `--executor-workers` only for explicit local review workers that preserve receipts, sampling coverage, evidence-matrix, and learning-dossier receipts to distinguish matched targets, substitutions, repeated sources, thin lanes, cross-agent citation links, learned findings, validation backlog, and next research actions, review gates when a deep run must prove minimum sampled breadth before skill updates, and the promotion-review ledger when deciding which findings can move toward asset/runbook edits. Run `scripts/audit_skill.py` and `scripts/validate_sources.py` before distributing this skill.
When `assets/architectures.yaml` changes, keep the golden scenario gate in `tests/test_scenarios.py` at full family coverage.
