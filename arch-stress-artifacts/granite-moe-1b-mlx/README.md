# Granite-3.0-1B-A400M (MoE) → MLX: verified worked port (arch stress-test, 2026-07-16)

A standalone eager MLX port of **`ibm-granite/granite-3.0-1b-a400m-instruct`** (`granitemoe`),
proving the **Mixture-of-Experts (MoE) architecture class** — the last of the seven — ports to MLX
with full parity. Proof artifact, outside the skill payload.

## Architecture
- 24 layers, hidden 1024, GQA (16 q / 8 kv heads), RMSNorm, RoPE.
- **MoE MLP**: router `Linear(1024→32)` → softmax → **top-8 of 32 experts** (renormalized weights);
  each expert a gated-SiLU MLP (intermediate 512); weighted combine of the 8 selected experts. No shared expert.
- **muP scalars**: embedding_multiplier ×12.0, attention_multiplier 0.015625 as the softmax scale,
  residual_multiplier ×0.22 on each block's branch, logits_scaling ÷6.0.

## Result: full 24-layer parity on real Metal
Fixed token fixture; HF `GraniteMoe` eager f32 oracle vs eager MLX on `mx.gpu`.

| Check | result |
|---|---|
| all 24 layers + embeddings + final norm + logits | ✅ pass — **min cosine 0.99999999987**, `first_divergence: none` |
| **router top-8 expert selection** | ✅ **exact expert IDs, in order, for every token in all 24 layers** |
| renormalized top-8 weights | ✅ cos ≥ 0.9999 |
| selected-expert outputs + weighted combine | ✅ match (combine cos 0.9999999999) |
| muP scaling (embed/attn/residual/logits) | ✅ applied correctly (parity holds end-to-end) |

The MoE-specific machinery — sparse top-k routing and the expert combine — is what makes this class
hard; here it matches HF *exactly* (expert IDs bit-identical), which is the meaningful proof.

## Honest scope
- Fixed fixture, prefill only. No incremental decode/cache claim. Independent oracle re-capture not repeated;
  evidence is the per-layer FP32 parity within thresholds + the exact router-ID match + a real (non-stub) `model.py`.
- Correctness reference, not an auto-scaffolded family — landing MoE in the executable generator is the follow-up.

## Files
`model.py` (eager MLX granitemoe: attention + top-8 MoE + muP), `parity.json`. No model weights included.
