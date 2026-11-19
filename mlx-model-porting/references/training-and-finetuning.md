# Training and fine-tuning

## Scope

Use training only when required for model support or an optimization artifact: adapter fine-tuning, quantization-aware training, speculative drafter/head training, distillation, architecture adaptation, or task validation.

## MLX training loop requirements

- source-oracle scalar loss parity before the first optimizer update;
- gradient tree key, shape, dtype, and selected-value parity;
- frozen versus trainable parameter membership checked explicitly;
- explicit lazy evaluation boundaries;
- optimizer/model/random state captured correctly for compilation;
- gradient accumulation semantics verified;
- train/eval mode and dropout behavior tested;
- checkpoint save/load and resume parity;
- deterministic small-batch overfit test;
- memory and graph-retention monitoring.

Use `value_and_grad` only with a scalar loss or with a tuple whose first element
is the scalar loss. For `Module` training, remember that gradients are taken
with respect to trainable parameters, so adapter/base freeze boundaries are part
of the parity contract.

## Checkpoint and resume contract

Full-model checkpoints should load strictly and preserve deterministic outputs.
Partial or adapter checkpoints may use non-strict loading only with an explicit
manifest of saved keys, expected shapes, trainable-parameter count, and a
before/after output or loss check.

Exact training resume must persist more than weights:

- model weights and any module state;
- optimizer state plus optimizer class and hyperparameters;
- scheduler state and counters;
- random state;
- data order, cursor, and gradient-accumulation remainder;
- train/eval mode and adapter merge/fuse state.

After reload, compare the next loss, optimizer state, and parameter delta
against an uninterrupted run within the declared tolerance.

## Compiled training

Compile only after eager training parity is established. A compiled train step
must pass model state and optimizer state as explicit inputs/outputs, and must
include random state when stochastic modules such as dropout are active. Compare
eager versus compiled loss and parameter updates, and run at least two legal
input shapes or batch configurations to expose retracing or stale captured
state.

Treat hidden mutation, Python side effects, closure-captured arrays, and
shapeless compilation as separate risks. Do not use a compiled path as the only
correctness oracle.

The [upstream state pattern](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/docs/src/usage/compile.rst)
is `state = [model.state, optimizer.state, mx.random.state]` with
`@partial(mx.compile, inputs=state, outputs=state)` when randomness is active.
Verify that consecutive stochastic calls advance RNG state and that restoring
it reproduces the same next update. Evaluate the returned loss and updated
state outside the compiled step.

## Native gradients and low-bit training

The reviewed 0.32.4 source contains native SDPA and gated-delta derivative
paths; use the [versioned runtime contract](runtime-compatibility.md) before
replacing an older fallback. Test the requested shape, dtype, mask, grouped
heads and recurrent-state gradients against eager/source references. A native
forward or VJP symbol does not establish all transforms or an entire model's
training support.

Distinguish frozen packed weights from differentiable training weights.
`QuantizedLinear` / `QuantizedEmbedding` freeze their quantized parameters;
`qqmm` requires an unquantized weight input if that weight must receive
gradients. Do not assume that converting a module with `nn.quantize` enables
QAT. Check trainable keys, adapter gradients and export-to-real-kernel quality
explicitly, using the [quantized-layer source](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/mlx/nn/layers/quantized.py)
and [QQMM contract](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/src/ops.cpp).

## Memory and graph retention

A training fit claim should log active, cache, and peak memory across repeated
steps, reset peak memory at phase boundaries, and include a budget-pressure run
when the target Mac matters. If gradient checkpointing is used, compare loss and
gradient behavior with and without checkpointing and record the extra compute
cost. Active memory alone is not a fit claim.

## Adapter path

Prefer LoRA or another parameter-efficient adapter for compatibility experiments. Record target modules, rank, scaling, dropout, base revision, tokenizer, dataset, seed, and merge status.

Official MLX-LM LoRA and QLoRA paths are reference patterns for LLM adapters,
not generic proof for audio, diffusion, vision, tabular, graph, or arbitrary
trainable ports. Before advertising a family as training-capable, reproduce a
tiny adapter train/eval path, verify adapter key coverage and trainable
parameter count, test adapter resume, and separately test exact optimizer-state
resume if claimed.

## Speculative artifacts

Medusa/MTP/EAGLE/diffusion drafters are trained components, not runtime switches. Define:

- target model and frozen/trainable parts;
- feature/logit interface;
- training data and on/off-policy behavior;
- loss and acceptance objective;
- tokenizer/vocabulary compatibility;
- evaluation across tasks and sampling regimes;
- artifact license and target revision lock.

## Quantization-aware training

Use when post-training quantization fails a quality target and a compatible low-bit kernel exists. Do not train for a format that the intended MLX runtime cannot execute efficiently.

## Audio fine-tuning

Preserve sample rate, loudness policy, segmentation, speaker splits, text normalization, phonemization, and codec version. Evaluate content leakage and voice-cloning consent/licensing where relevant.
