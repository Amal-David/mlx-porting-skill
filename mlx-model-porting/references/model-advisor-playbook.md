# Model Advisor Playbook

Use this when a user wants model-specific direction, a CLI-first advisor flow,
or a UI-style answer for “what can I do with this model?” This playbook turns
the research assets into step-by-step directions without hiding unvalidated
branches.

## Table of Contents

1. Advisor contract
2. Step-by-step flow
3. Evidence sweep
4. Family branches
5. Recent research branches
6. Output shape
7. Copyable instructions

## Advisor Contract

For a specific model, answer as an engineering advisor:

- identify the model source, source revision, license risk, target hardware,
  memory budget, latency/throughput/quality objective, and user constraints;
- identify the detected architecture family and primary runbook;
- give the validated conversion route first;
- show benchmark-required optimizations separately from already validated
  source/theory guidance;
- preserve every relevant research lead as either validated, experimental,
  context-only, or rejected;
- ask before helping execute experimental work:
  “This is an experimental approach. Do you want to try it?”

Do not collapse all research into one shortlist. A useful advisor tells the
user what to do now, what can be tried with caution, what still needs a gate,
and what not to use.

## Step-By-Step Flow

1. **Collect model facts.** Record model id/path, source framework, artifact
   formats, license, `trust_remote_code` or custom module risk, tokenizer or
   processor, target Mac, memory budget, objective, and quality target.
2. **Run static intake.**
   ```bash
   python3 scripts/inspect_model.py MODEL_OR_DIRECTORY --output inspection.json
   python3 scripts/make_port_plan.py inspection.json --output PORT_PLAN.md
   python3 scripts/recommend_optimizations.py inspection.json --markdown OPTIMIZATIONS.md
   ```
3. **Classify the route.** Load `assets/architectures.yaml`,
   `references/model-support-map.md`, and the selected runbook. If source
   format manifests are present, treat them as static triage until parity and
   conversion gates exist.
4. **Sweep research assets.** Read the evidence sources listed in the next
   section before finalizing advice. Do not skip a relevant lead just because
   it is not ready for supported guidance.
5. **Build the validated path.** Give the source-oracle, minimal eager MLX,
   weight-map, parity, quality, profile, and packaging steps before any
   optimization.
6. **Classify all branches.** Use the advisor buckets:
   - `validated-locally`: local test, benchmark, fixture, or skill gate exists;
   - `validated-source-theory`: official docs, primary paper, or pinned source
     backs the path, but local confirmation is still needed;
   - `benchmark-required`: safe after parity, but no speed/memory number may be
     claimed before target-hardware measurement;
   - `experimental-approach`: promising but not promotion-ready;
   - `rejected-do-not-use`: incompatible, unsafe, contradicted, license-blocked,
     CUDA-only, or superseded.
7. **Run a short plan session for experiments.** For each experimental branch,
   say why it is promising, why it is experimental, the required validation
   gate, and the rollback condition. Ask the opt-in question exactly enough to
   require explicit consent. If the user does not say to try it, keep it in the
   report/backlog and proceed only with validated or benchmark-required steps.
8. **Try only the selected branch.** When the user opts in, execute it after
   eager parity. Change one dimension at a time, record correctness, quality,
   benchmark, memory, and rollback evidence.
9. **Return the user-facing card.** Include supported path, experimental
   options, rejected paths, evidence basis, commands, and a copyable agent
   instruction.

## Evidence Sweep

Always check these assets when producing model-specific advice:

- `assets/recommendation-taxonomy.yaml`: maps optimization statuses to advisor
  buckets and defines the experimental opt-in prompt.
- `assets/optimization_guidance.yaml`: source of optimization methods,
  statuses, applies-to families, gates, rollback conditions, and evidence refs.
- `assets/contributor_learnings.json`: top-1000 contributor and repository
  learnings. Read `advisor_buckets` and preserve validated, reinforcing,
  experimental, and context-only branches.
- `assets/research_backlog.json`: comprehensive-coverage backlog. Map
  `validated` to validated guidance and `needs-validation` to experimental or
  validation-backlog guidance.
- `references/deep-research-loop.md`: promotion-review rules and research
  receipt semantics.
- The selected runbook and relevant optimization references such as
  `attention-and-kv.md`, `compile-and-kernels.md`,
  `decoding-and-serving.md`, `quantization.md`, or
  `training-and-finetuning.md`.

If a lead matches the model family or objective and is not in a validated
bucket, include it under experimental approaches with its gate. Do not omit it.

## Family Branches

### Dense Decoder LM

Validated route:

1. Use `runbook-decoder-transformer.md`.
2. Prefer official MLX-LM architecture, tokenizer, generation, and cache
   patterns when available.
3. Establish logits and cache parity before optimizations.
4. Consider native/source-backed options: fast SDPA, `mx.compile` for stable
   decode regions, prompt cache, weight quantization, KV/state policy, and
   serving scheduler only when the objective needs them.

Experimental branches:

- prompt-lookup or multi-token speculation when compatibility, quality, and
  acceptance-rate gates are not yet reproduced;
- layer eval/watchdog cadence changes when command-buffer risk is suspected but
  no local reproducer exists;
- layer-range network sharding unless local distributed parity, failure, and
  throughput gates exist.

### MoE Decoder LM

Validated route:

1. Use `runbook-moe-transformer.md`.
2. Preserve router logits, top-k expert selection, expert weights, and dispatch
   layout before optimization.
3. Treat native MLX expert dispatch and quantization guidance as
   source/theory-validated until model-local parity and benchmarks pass.

Experimental branches:

- synthetic MoE assembly and model creation workflows; these are not
  source-preserving ports until a source oracle and quality target exist;
- fused gate/up projection patches unless parity, fallback, and benchmark gates
  exist for the specific model.

### Encoder, Encoder-Decoder, ASR, And Streaming Speech

Validated route:

1. Use `runbook-encoder-transformer.md`, `runbook-encoder-decoder.md`,
   `runbook-asr.md`, or `runbook-streaming-speech.md`.
2. Preserve feature extraction, masks, positional encodings, cache/state, beam
   or transducer state, and decoded-output quality gates.
3. Treat Conformer, RNNT, and streaming state as architecture-critical, not
   interchangeable implementation detail.

Experimental branches:

- RWKV, SSM, and Conformer prototypes from contributor research when they lack
  packaged tests, streaming-state parity, or task-quality gates.

### Diffusion, Flow, Vision-Language, And Spatial Ports

Validated route:

1. Use `runbook-diffusion-flow.md` or `runbook-multimodal-omni.md`.
2. Preserve scheduler, latent scaling, conditioning, text/image/audio
   frontends, sampling, and postprocessing before speed work.
3. For repeated-block models that exceed unified memory, show
   `block-weight-streaming` as benchmark-required: it has pinned contributor
   evidence and tests, but still needs model-local parity, memory, wall-time,
   and fallback gates.
4. For PyTorch `grid_sample`-style spatial or volume warping, show
   `spatial-grid-sample-kernel` as benchmark-required only when the source
   actually depends on supported grid-sample modes and fallback parity exists.

Experimental branches:

- untested grid-sample modes, training gradients, reflection padding, or nearest
  sampling beyond the validated kernel modes;
- layer eval/watchdog cadence changes without a local reproducer.

### Audio, TTS, Vocoder, Codec, Separation

Validated route:

1. Use the audio runbook matching the model:
   `runbook-audio-codec.md`, `runbook-autoregressive-audio.md`,
   `runbook-flow-tts.md`, `runbook-vocoder.md`, or
   `runbook-separation-enhancement.md`.
2. Preserve sample rate, chunk length/stride, bandwidth, tokenizer/codebook
   count, reference audio format, duration prediction, mel path, vocoder decode,
   streaming state, and perceptual/task quality gates.

Reinforcing research:

- `audio-flow-tts-duration-vocoder-gates` supports existing duration, vocoder,
  EnCodec, and reference-audio gates. Mention it as evidence, not a new
  standalone optimization.

Experimental branches:

- audio or speech prototypes without local quality, streaming, and state
  fixtures.

### Non-Generative CV

Validated route:

1. Use `runbook-non-generative-cv.md` for backbone/classifier style ports.
2. Keep the current synthetic backbone/classifier fixture green and verify
   preprocessing, layout, logits/features, and task metric.

Experimental branches:

- promptable masks, detection, segmentation, depth, OCR, pose, dense vision, and
  feature extraction beyond the validated backbone/classifier path until their
  fixtures and task metrics exist.

### Time-Series, Structured, Ranking, And Recommenders

Validated route:

1. Use `runbook-time-series-forecasting.md` for the existing forecasting path.
2. Preserve scaler/normalizer, context/prediction split, observed-mask handling,
   lags, known-future covariates, quantiles, and forecast metrics.

Experimental branches:

- tabular/skops, deep tabular, text ranking, and recommender subfamilies until
  loader, preprocessing, pair-score/top-k, and task-metric gates exist.

### Graph, Geometric, Scientific

Validated route:

1. Use `runbook-graph-message-passing.md` for GCN-style message passing.
2. Preserve graph structure, node/edge features, scatter/segment/reduce parity,
   permutation behavior, and task metric.

Experimental branches:

- point-cloud, neighbor/radius graph, equivariant, protein, chemistry,
  energy/force, and scientific ports until ragged batching, symmetry, units,
  and domain metric gates exist.

### Training And Fine-Tuning As The Target

Treat training as experimental until a local training fixture exists. The
required gate includes loss parity, selected gradient parity, trainable/frozen
membership, tiny overfit, optimizer-state round trip, checkpoint resume,
train/eval mode checks, adapter merge/fuse parity, compile/checkpointing
parity, and memory graph-retention checks.

### Source Formats Beyond PyTorch Or Hugging Face

Static source-format intake is useful triage, not executable support by itself.
ONNX, GGUF, Flax/Orbax, TensorFlow SavedModel, Keras archive, Core ML package,
and safetensors-only checkpoint manifests can feed planning and weight-map
shape coverage only when the manifest exposes safe static tensor metadata.
Treat broader checkpoint-only conversion as experimental until source oracle,
operator/layer mapping, and executable parity gates exist.

## Recent Research Branches

Preserve these recent top-1000 contributor and multi-source findings:

| Branch | Advisor bucket | When to show |
|---|---|---|
| `block-weight-streaming` | benchmark-required | repeated-block diffusion/flow, VLM, or flow-TTS memory pressure after eager parity |
| `spatial-grid-sample-kernel` | benchmark-required | PyTorch-compatible image/volume warping with fallback parity |
| `audio-flow-tts-duration-vocoder-gates` | reinforces existing guidance | flow/diffusion TTS, vocoder, codec, or streaming audio ports |
| `operation-benchmark-hygiene` | reinforces existing guidance | every speed or memory claim |
| `paged-kv-and-gather-qmm-serving` | reinforces existing guidance | concurrent LLM/VLM serving and MoE cache policies |
| `layer-eval-watchdog-guard` | experimental approach | suspected command-buffer/watchdog risk without local reproducer |
| `layer-range-network-sharding` | experimental approach | attempted multi-device/network sharding |
| `synthetic-moe-assembly` | experimental approach | architecture creation rather than source-preserving port |
| `rwkv-ssm-conformer-prototypes` | experimental approach | SSM/RWKV/Conformer prototypes without packaged gates |
| `mlx-onnx-export-adjacent` | context-only | export ecosystem context, not model-to-MLX execution guidance |
| `source-formats-beyond-pytorch-hf` | experimental/backlog | static manifests exist but broader executable conversion is not proven |
| `top1000-contributor-long-tail-rescreening` | experimental/backlog | GitHub long-tail search was rate-limited or incomplete |
| `non-generative-cv` | experimental/backlog beyond backbone | dense/promptable/detection/segmentation/depth/OCR/pose paths |
| `structured-timeseries-recsys` | experimental/backlog beyond forecasting | tabular, ranking, and recommender subfamilies |
| `graph-geometric-scientific` | experimental/backlog beyond GCN | point-cloud, equivariant, protein, chemistry, energy/force |
| `training-as-port-target` | experimental/backlog | training and fine-tuning as the primary port target |

## Output Shape

Use this structure for a CLI or UI answer:

```text
Model: <id or path>
Detected family: <family>
Primary route: <runbook and source reference>
Target: <hardware/objective/quality gate>

What we know:
- source/revision/license:
- artifact format:
- blockers:

Validated path:
1. ...
2. ...

Validated by source/theory:
- ...

Benchmark-required:
- ...

Experimental approaches:
- <id>: promising because ...; experimental because ...; gate ...; rollback ...
  Ask: This is an experimental approach. Do you want to try it?

Rejected / do not use:
- ...

Copyable instruction:
...
```

## Copyable Instructions

For a supported path:

```text
Use the MLX model porting skill. For <model>, inspect statically, classify the
architecture, build the source oracle, port the smallest eager MLX path first,
validate the weight map and parity ladder, then apply only validated or
benchmark-required optimizations. Record hardware, workload, quality gates,
benchmark results, and rollback conditions.
```

For an experimental branch after opt-in:

```text
Use the MLX model porting skill. Try experimental approach <id> for <model>
only after eager parity passes. State why it is experimental, implement the
smallest isolated branch, run the required validation gate, measure quality,
memory, and latency, and revert if the rollback condition is hit.
```
