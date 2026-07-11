# Parity and testing

## Capture the source oracle

Use `scripts/capture_oracle.py` instead of assembling decoder captures by hand.
It executes only a local Hugging Face model directory, keeps Transformers in
offline mode, refuses remote model/tokenizer code, and writes a bounded NPZ plus
a deterministic JSON manifest. A tokenizer-free fixture is the most portable
starting point:

```bash
python3 mlx-model-porting/scripts/capture_oracle.py MODEL \
  --token-ids 1 42 17 9 \
  --generate-steps 4 \
  --output source-oracle.npz
```

Use one or more `--prompt` values, or line-oriented `--prompts-file`, when the
local tokenizer behavior is itself part of the oracle. The NPZ keys are a
stable cross-framework contract:

| Key | Meaning |
|---|---|
| `input_ids` | Rank-2 source token IDs. |
| `attention_mask` | Integer mask paired with `input_ids`. |
| `embed` | Decoder embedding-stage hidden state before layer 0. |
| `layer.{i}.hidden` | Post-block hidden state for every zero-based decoder layer. |
| `layer.{i}.attention` | Optional attention branch output when a standard attention submodule is hookable. |
| `layer.{i}.mlp` | Optional MLP branch output before the residual when a standard MLP submodule is hookable. |
| `final_norm` | Final decoder normalization output. |
| `logits` | Full prompt logits. |
| `generated_token_ids` | Exactly N greedy continuation IDs; prompt IDs are not repeated. |

Encoder-decoder captures use architecture-specific stable keys:
`encoder.embed`, every `encoder.layer.{i}.hidden`, `encoder.final_norm`,
`decoder_input_ids`, `decoder.embed`, every
`decoder.layer.{i}.cross_attention` and `decoder.layer.{i}.hidden`,
`decoder.final_norm`, first-step `logits`, and `generated_token_ids`. Source
input IDs and the attention mask always describe the encoder input.

Floating captures are saved as float32 unless `--keep-dtype` is explicit.
Integer IDs and masks retain their integer dtype. Future target-side capture
tools should mirror these names exactly; model-specific extra checkpoints may
use additional keys without renaming the stable set.

## Run the cross-framework ladder

After scaffolding the MLX package and converting its weights, use the bounded
one-command runner instead of manually pairing archives:

```bash
python3 mlx-model-porting/scripts/run_parity.py \
  --source-model MODEL \
  --package mlx_port \
  --weights converted \
  --token-ids 1 42 17 9 \
  --generate-steps 4 \
  --output parity-report.json
```

Prompt mode accepts the same repeatable `--prompt` and `--prompts-file`
fixtures as `capture_oracle.py`; the runner passes the pinned local source
tokenizer to `capture_mlx.py`. Token-ID mode never loads a tokenizer. The
strict-JSON report compares same-name keys in runbook order and stops
immediately after the first failure. Decoder-only order remains `input_ids`,
`embed`, every `layer.{i}.hidden`, `final_norm`, `logits`, and exact generated
IDs. Encoder-decoder order follows the encoder stack, then decoder start,
cross-attention and block boundaries, decoder norm, logits, and exact generated
IDs.

`capture_mlx.py` validates the scaffold generator header, config digest,
execution-file digests, and converted target parameter contract before running
the package. The package remains user-owned Python code executed with the
current user's authority; this validation detects drift but is not a sandbox.
Use `--allow-modified` only after reviewing intentional package edits.

## The parity ladder

### 1. Artifact parity

Check config values, tokenizer/processor files, special IDs, sample rate, FFT/mel settings, context windows, codebooks, delay pattern, generation defaults, and normalization constants.

### 2. Weight parity

Check key coverage, shape, dtype, transformed statistics, tied parameters, and shard completeness. Randomly sample elements after each nontrivial transform.

### 3. Primitive parity

Use tiny deterministic tensors for normalization, RoPE/position encoding, attention masks, convolution padding, FFT/STFT, quantizer lookup, recurrence, cache update, and sampler logic.

### 4. Block parity

Compare one block with fixed inputs and fixed state. Capture pre-normalization, projections, attention/SSM output, residual branches, MLP/MoE output, and updated state.

### 5. Staged model parity

Compare after frontend/embedding, every N blocks, bottleneck, output projection, and postprocessor. Binary-search the first divergent boundary.

### 6. End-to-end parity

Use deterministic decode or reconstruction. Compare logits/latents before comparing decoded text or audio, because postprocessing can obscure the first error.

### 7. Stateful parity

Compare:

- full sequence versus token/chunk incremental execution;
- empty, growing, rotating, and truncated cache;
- reset/reuse behavior;
- batch merge/split if supported;
- streaming chunk boundaries and final flush.

### 8. Task-quality parity

Tensor allclose is not enough when approximations are introduced.

- LLM: perplexity or log-prob delta, task accuracy, generation agreement under deterministic decoding.
- ASR: WER/CER and timestamps where relevant.
- TTS: intelligibility via ASR, speaker similarity, duration/prosody checks, clipping, and listening tests.
- Codec/vocoder: SI-SDR/SNR where appropriate, spectral distance, PESQ/STOI where licensed and applicable, and perceptual review.
- VLM: task/benchmark sample agreement and visual-token path tests.
- Diffusion/flow: fixed-seed latent and output metrics plus perceptual/task review.
- Non-generative CV: top-k accuracy for classification; IoU/AP/Dice for
  detection and segmentation; prompt-to-mask IoU for SAM-like models;
  AbsRel/RMSE/delta metrics for depth; COCO OKS/AP for pose; normalized edit
  distance or text-line accuracy for OCR.
- Time-series forecasting: scaler and lag-construction parity, fixed
  context/prediction splits, quantile or forecast tensor parity, leakage checks
  for known-future covariates, and forecast-error metrics appropriate to the
  source benchmark.
- Structured and tabular loaders: scaler/normalizer parity, categorical vocab
  and missing-value policy, preprocessing leakage checks, and task metrics
  appropriate to the source estimator or deep tabular model.
- Ranking and recommender subfamilies: pair-score parity, top-k ordering
  stability, NDCG, AUC/AP, retrieval recall, and candidate-id versus score-based
  retrieval checks.
- Graph message passing: scatter/segment/reduce parity, permutation
  invariance, batched-graph boundary checks, and task metrics such as OGB
  accuracy/ROC-AUC.
- Point-cloud, equivariant, and scientific ML: neighbor-list determinism,
  rotation/reflection equivariance, unit constraints, and task metrics such as
  ModelNet/ShapeNet accuracy or mIoU, molecular MAE/RMSE, and energy/force
  errors.
- Training/fine-tuning: scalar loss parity, selected gradient parity,
  trainable-parameter membership, tiny-overfit behavior, checkpoint-resume next
  loss and parameter delta, adapter merge/fuse parity, and memory
  graph-retention checks.

## Tolerance policy

Set tolerances per stage and dtype. Do not use one permissive global tolerance.

Typical policy structure:

| Stage | Precision | Metric | Starting threshold |
|---|---|---|---|
| FP32 primitive | FP32 | max abs / max rel | tight, operation-specific |
| FP16/BF16 block | reduced | allclose + cosine | moderate, inspect accumulation |
| Logits/latents | mixed | max abs + rank/top-k agreement | task-specific |
| Quantized model | low-bit | quality metric and distribution drift | baseline-relative |
| Audio waveform | floating | aligned spectral/perceptual metrics | architecture-specific |

Thresholds are hypotheses, not universal constants. Record why each is acceptable.

## Failure localization

When a final output fails:

1. reproduce with the smallest fixture;
2. compare preprocessing output;
3. find the first divergent checkpoint;
4. disable cache/streaming/quantization/compile;
5. compare in FP32 where possible;
6. inspect shape, axis order, masks, positions, and state update;
7. test the isolated primitive;
8. fix the first divergence only, then rerun the ladder.

The `run_parity.py` summary makes step 3 executable: a failure at
`layer.7.hidden` reports `layer 7` as the debug target and does not compare
final norm, logits, or generation downstream of that divergence.

## Required regression matrix

At minimum include:

- minimal valid input;
- common input;
- boundary length;
- odd/nonmultiple dimensions when legal;
- empty or reset state;
- long cache/stream;
- batch one and batch greater than one if advertised;
- every supported dtype/quantization mode;
- save/reload;
- deterministic repeatability.
