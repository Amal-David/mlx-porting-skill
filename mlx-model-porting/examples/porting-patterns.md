# Before/after porting patterns

These examples are search-friendly starting points for common PyTorch-to-MLX
requests. They are not finished model ports and they do not weaken the core
rule: establish a source oracle, port the smallest eager path, pass parity, then
optimize with measured receipts.

Use a card when someone arrives with a pasted PyTorch module, a Hugging Face
checkpoint, or a fine-tuning script and wants to know what the MLX conversion
should look like.

## How to use a card

1. Pick the closest source surface below.
2. Paste the source module, config, checkpoint index, or training script.
3. Ask the agent to produce `PORT_PLAN.md`, source-oracle checkpoints, a weight
   map, and the smallest eager MLX block.
4. Do not quantize, compile, fuse, or add custom kernels until the card's gate
   passes.

## Transformer attention block port

Search terms: Llama attention to MLX, Qwen attention to MLX,
`torch.nn.functional.scaled_dot_product_attention`, KV cache, RoPE.

Before:

- PyTorch block with separate or fused `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Tensor layout is often `[batch, seq, hidden]` until it is reshaped into heads.
- RoPE, causal mask, padding mask, and cache position may be spread across
  helpers.
- Weight loading often hides QKV split, transpose, or tied-output assumptions.

After:

- MLX `nn.Module` with explicit q/k/v projection rules and visible layout
  transitions between token, head, and cache shapes.
- RoPE base, scaling, offset, and cache write position are first-class config.
- A readable attention path is implemented first; native fast SDPA is allowed
  only after mask and cache parity pass.
- Weight map records split/fuse, transpose, head count, KV head count, and tied
  parameters.

Primary runbook: `../references/runbook-decoder-transformer.md`.

Gate:

- Q/K/V before and after RoPE match the source.
- One-layer attention output matches with causal and padding masks.
- Full-prefill logits and incremental-cache logits agree with full recompute.

Prompt:

```text
Use the MLX model porting skill. Treat this as a Transformer attention block
port. Start from the pasted PyTorch block, write the source-oracle checkpoints,
then produce the smallest eager MLX block and weight map before any optimization.
```

## PyTorch checkpoint loading into MLX

Search terms: load PyTorch checkpoint in MLX, safetensors to MLX, state_dict to
MLX, convert Hugging Face weights to MLX.

Before:

- PyTorch or Hugging Face `state_dict`, `.bin`, `.pt`, `.pth`, or
  `.safetensors` files.
- Source code may use permissive `strict=False`, regex renames, or silent
  missing-key ignores.
- Linear, convolution, QKV, gate/up/down, and embedding/head ties can require
  different transforms.

After:

- Static intake first: inspect config, safetensors headers, source-format
  metadata, remote-code risk, license, and provenance before executing anything.
- Deterministic weight map records source key, target key, source shape, target
  shape, and transforms such as rename, transpose, split, concat, reshape, cast,
  squeeze, or tie.
- Missing source keys, generated target keys, or ambiguous transforms fail loud.
- Save/reload is part of the minimal port, not a packaging afterthought.

Primary references: `../references/porting-core.md` and
`../references/packaging-and-publication.md`.

Gate:

- `inspect_model.py` emits the expected source tensors.
- `validate_weight_map.py` proves deterministic shape coverage.
- Primitive or block parity proves the mapping is semantically correct; shape
  coverage alone is not a parity claim.

Prompt:

```text
Use the MLX model porting skill. Convert this PyTorch/Hugging Face checkpoint
to an MLX weight map. Do static intake first, reject silent missing keys, and
stop at deterministic shape coverage until block parity proves the mapping.
```

## Whisper-style audio model port

Search terms: Whisper to MLX, ASR to MLX, log-mel frontend, CTC to MLX,
streaming speech MLX.

Before:

- PyTorch ASR stack with waveform preprocessing, resampling, log-mel or learned
  frontend, encoder, decoder or CTC head, tokenizer, timestamps, and text
  normalization.
- Optional VAD, beam search, language/task prompts, suppression processors, or
  timestamp postprocessing may live outside the neural graph.
- Quality can appear wrong because preprocessing or text normalization drifted,
  not because the encoder port failed.

After:

- MLX path is split into waveform/features, encoder hidden states, logits or
  decoder tokens, transcript, timestamps, and streaming state where applicable.
- Frontend dependencies and language/task tokens are recorded as provenance.
- The first decoder path is deterministic and simple: greedy CTC, greedy
  seq2seq, or greedy RNNT before beam/search work.
- Streaming state and endpointing are separate product gates.

Primary runbook: `../references/runbook-asr.md`.

Gate:

- Features, selected encoder layers, logits, token IDs, and transcript match the
  source fixture.
- WER/CER, timestamp boundaries, silence, noise, long audio, and code-switch
  cases are reported before quality or speed claims.
- End-to-end metrics include preprocessing and postprocessing time.

Prompt:

```text
Use the MLX model porting skill. Treat this as a Whisper-style ASR port. Freeze
waveform and log-mel fixtures first, compare encoder/logit checkpoints, then add
the simplest deterministic decoder before timestamp, beam, or streaming work.
```

## Small diffusion block port

Search terms: diffusion U-Net block to MLX, DiT block to MLX, scheduler step,
latent diffusion MLX.

Before:

- PyTorch U-Net, DiT, or flow block with timestep or sigma embeddings,
  conditioning, attention, normalization/modulation, and scheduler code.
- Latent scaling, CFG ordering, random state, scheduler off-by-one behavior,
  and patch/video layout can be more important than the block code itself.
- "Fewer steps" or a distilled sampler is often a model/quality mode, not a
  lossless runtime optimization.

After:

- MLX denoiser block runs for one fixed timestep before full sampling.
- Scheduler config and update math are explicit artifacts.
- Conditioning, latent shape, patch/video axes, and VAE scaling are recorded at
  the boundary.
- Compile, quantization, tiling, and custom kernels wait until the fixed-seed
  eager path is correct.

Primary runbook: `../references/runbook-diffusion-flow.md`.

Gate:

- One-step denoiser output and scheduler update match the source.
- Fixed-seed latent trajectory stays within declared tolerance at early, middle,
  and final steps.
- Decoded output and postprocessing are checked before visual quality or speed
  claims.

Prompt:

```text
Use the MLX model porting skill. Treat this as a small diffusion block port.
Port one denoiser timestep and one scheduler update first, then compare a tiny
fixed-seed latent trajectory before any compile, quantization, or step reduction.
```

## LoRA fine-tuning to MLX

Search terms: PyTorch LoRA to MLX, PEFT LoRA to MLX, QLoRA MLX, adapter
checkpoint to MLX.

Before:

- PyTorch or PEFT fine-tuning script with target modules, rank, scaling,
  dropout, adapter checkpoint, base revision, tokenizer, dataset, and optimizer.
- The script may rely on hidden trainable/frozen membership, non-strict adapter
  loading, or training resume state outside weights.
- A working LLM LoRA path does not prove arbitrary audio, vision, diffusion, or
  graph training support.

After:

- MLX adapter plan records target modules, rank, scaling, dropout, base model
  revision, tokenizer, dataset, seed, merge/fuse status, and saved adapter keys.
- Frozen versus trainable parameter membership is tested explicitly.
- Eager scalar-loss and selected-gradient parity come before compiled training.
- Checkpoint resume includes model state, optimizer state, scheduler/counters,
  random state, data cursor, train/eval mode, and accumulation remainder when
  exact resume is claimed.

Primary reference: `../references/training-and-finetuning.md`.

Gate:

- Scalar loss parity passes before the first optimizer step.
- Selected gradients and trainable parameter count match intent.
- Tiny overfit, adapter save/load, and reload output or loss checks pass.
- Exact resume is claimed only after next-loss and next-update parity.

Prompt:

```text
Use the MLX model porting skill. Treat this as a LoRA fine-tuning port. Start by
listing target modules and trainable parameters, then build eager scalar-loss and
gradient parity before adapter save/load, resume, or compiled training.
```
