# Qwen3.5-text (linear-attention hybrid) → MLX: verified worked port (arch stress-test, 2026-07-15)

A standalone eager MLX port of the **text tower of `Qwen/Qwen3.5-2B`** (`qwen3_5_text`; the full
model is a vision-language model). This is the toolkit's **hardest** architecture family: a
linear-attention / Mamba hybrid, not a dense transformer. Proof artifact, outside the skill
payload.

## Architecture (materially different from a dense decoder)
- 24 layers: **18 `linear_attention` (Gated-DeltaNet / Mamba-style) + 6 `full_attention`** (every 4th).
- Linear-attention mixer: causal depthwise conv1d (kernel 4) + a token-by-token **Gated-DeltaNet
  recurrence** (16 key/value heads, head dims 128), f32 SSM state.
- Full-attention: GQA (8 q / 2 kv heads, head_dim 256), **gated attention output**, **partial RoPE**
  (`partial_rotary_factor 0.25` — only 25% of head_dim rotated), **text M-RoPE** (`mrope_section
  [11,11,10]`; text positions share all sections), `rope_theta 1e7`.
- Gated RMSNorm (SiLU gate), SiLU MLP (intermediate 6144), tied embeddings, MTP head (excluded from the trunk parity).

## Result: full 24-layer parity on real Metal
Fixed 9-token full-sequence FP32 fixture; HF eager oracle vs eager MLX on `mx.gpu`.

| Rung | result |
|---|---|
| all 24 layers (linear + full) + embeddings + final norm | **cos ≥ 0.9999999999994, max_abs ~1e-6 — FP32 parity within thresholds** |
| first divergence | **none** — `max_consecutive_pass = 24` |

**Primitive-level cross-check** (independent, in `primitive_parity.json`) — rules out compensating errors:
- `gated_delta_recurrence`: cos 0.99999999999, **max_abs 3.7e-8**
- `causal_depthwise_conv1d`: max_abs 9.5e-7 · `rms_norm_additive`: 2.4e-7 · `text_mrope_cos/sin`: ~2.4e-7

So both the Gated-DeltaNet recurrence and the gated partial-M-RoPE attention are numerically correct.

## Honest scope / caveats
- Prefill only, on ONE fixed fixture. **Incremental decode / cache is not claimed.** Broader fixtures,
  the MTP head, and the vision tower are out of scope.
- Independent host re-verification of the oracle capture was not repeated; the evidence here is the
  per-rung FP32 threshold parity + the independent primitive checks + a real (non-stub) `model.py`.
- This is a correctness reference, not an auto-scaffolded family — landing linear-attention hybrids in
  the executable generator is the follow-up this artifact de-risks.

## Files
`model.py` (eager MLX qwen3_5_text), `parity.json` (24-rung evidence), `primitive_parity.json`
(primitive cross-checks), `oracle_manifest.json`. No model weights included.
