# Runbook: dense decoder Transformer

## Applies to

Llama/Mistral/Qwen/Gemma/Phi/GPT/Falcon/OLMo-style causal decoders and close variants with dense attention/MLP blocks. Route MoE, recurrent/SSM, multimodal, or codec-token architectures to their specialized runbooks.

## Architecture fingerprint

Confirm:

- decoder-only causal stack;
- token embedding and optional tied LM head;
- pre-norm, post-norm, or sandwich norm;
- MHA/GQA/MQA/MLA attention;
- RoPE/ALiBi/relative positions and scaling variant;
- SwiGLU/GeGLU/GELU or other MLP;
- attention bias, QK norm, logit soft cap, sliding window, sinks;
- cache shape and layer sharing;
- tokenizer/chat template and generation processors.

## Source oracle checkpoints

Capture:

1. token IDs and attention mask;
2. embeddings after scale/dropout;
3. Q/K/V before and after RoPE;
4. attention scores/output for one layer;
5. MLP gate/up/down intermediates;
6. residual output every N layers;
7. final norm and logits;
8. cache after prefill and several decode steps.

Use a short prompt that includes padding, a position beyond zero, and enough tokens to expose causal masking.

## Weight conversion

Common transforms:

- split or fuse Q/K/V projections;
- transpose framework linear weights only when target convention requires it;
- map gate/up/down projections and preserve activation order;
- preserve tied embedding/head identity;
- map per-head Q/K norms;
- account for shared KV or cross-layer projections;
- store RoPE constants/config rather than converting nonexistent weights;
- preserve quantization metadata only after unquantized parity.

Never infer QKV split sizes from equal thirds when GQA/MQA makes K/V smaller. Derive sizes from hidden size, head count, KV head count, and head dimension.

## Minimal MLX path

1. Capture the pinned source oracle with `scripts/capture_oracle.py`.
2. For an unblocked trusted inspection, run `scripts/scaffold_port.py inspection.json --artifact-root MODEL --output mlx_port`. It re-inspects the artifact and fails closed on unsupported config semantics.
3. Review the generated config assumptions and stable parameter-name scheme against the source implementation before converting any weight.
4. Validate normalization, configured gated MLP activation, and RoPE scaling independently.
5. Validate attention without cache, including exact causal/padding mask behavior, and test one complete block.
6. Validate the generated full/growing KV cache against full-context logits.
7. Declare every rename, transpose, QKV split, merge, target shape, and dtype policy in `WEIGHT_MAP.json`; run `scripts/convert_checkpoint.py --source MODEL --mapping WEIGHT_MAP.json --output converted`, then validate `converted/target-manifest.json` with `scripts/validate_weight_map.py` before checking the assembled stack, final norm, tied or untied head, and logits. The converter refuses draft or unresolved maps and unexplained source tensors.
8. Run `scripts/run_parity.py --source-model MODEL --package mlx_port --weights converted` with the shared prompt or token-ID fixture. It drives source and MLX capture, maps the stable keys, and stops at the first failing embedding/layer/final-norm/logit/generation rung. Debug only that first divergence before rerunning.
9. Use `scripts/compare_tensors.py` directly for optional attention/MLP branch captures, then validate deterministic greedy generation, save/reload, boundary context, and cache reset/reuse.

`scaffold_port.py` is a starting implementation, not a guaranteed port. It currently covers only the explicit dense-decoder feature allowlist and rejects sliding-window attention, QK normalization, MoE, quantization metadata, unknown RoPE scaling, soft caps, and unrecognized computation-bearing config keys. A generated package must still pass source parity and task-quality gates; edit it when inspection-backed source semantics differ.

## Parity traps

- different RoPE interleaving, base, scaling, or offset;
- mask type/broadcast differences and finite versus `-inf` masking;
- QK normalization placement;
- attention scale based on head dimension versus config override;
- final logit scaling or soft cap;
- residual in FP32 versus reduced precision;
- tied weights duplicated during serialization;
- cache position advanced before versus after write;
- sliding-window layers mixed with full-attention layers;
- special BOS/EOS and chat-template behavior mistaken for model logic.

## Optimization ladder

1. Remove host sync from decode loop and batch token decoding outside the model step.
2. Use native fast SDPA for compatible attention.
3. Preallocate or use a proven cache object; avoid per-token concatenation.
4. Compile the single-step decoder with explicit cache state when shapes are stable.
5. Chunk long prefill if peak memory is the bottleneck.
6. Apply weight-only quantization with sensitive-module exclusions.
7. Apply KV quantization or rotating cache only after long-context quality checks.
8. Add prompt/prefix caching for repeated prefixes.
9. Add continuous batching for concurrent serving.
10. Evaluate compatible speculative decoding: MLX-LM draft-model speculation where tokenizer and acceptance match; prompt lookup only if the target runtime implements it; EAGLE/DFlash/MTP only with pinned compatible artifacts.

## Quantization policy

Start by leaving token embeddings, LM head, norms, small projections, and Q/K norms unquantized. Quantize large linear layers first. Compare logits and perplexity before generation quality. For mixed-bit recipes, test early/late layers and attention/MLP sensitivity separately.

## Completion gates

- full and incremental logits agree within declared tolerance;
- cache reset/reuse and long positions pass;
- greedy generation matches the source on fixtures;
- tokenizer/template behavior is separately verified;
- performance report separates prefill, decode, and first-run compile;
- every advertised context, batch, dtype, and quantization mode has a test.
