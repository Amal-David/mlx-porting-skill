# ModernBERT-base → MLX: verified worked port (arch stress-test, 2026-07-15)

A standalone eager MLX port of [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
(Apache-2.0), built to test whether this toolkit's porting **methodology** extends to a genuinely
novel, leaderboard-trending encoder architecture that the automated scaffold does **not** yet cover.

This is a proof artifact, not a scaffold-generated port. It lives outside the `mlx-model-porting/`
skill payload on purpose (it is evidence + reusable reference code, not a shipped skill file).

## Result: correct port, per-layer parity on real Metal

Captured a Hugging Face eager FP32 oracle (`ModernBertModel`, `output_hidden_states`) and compared
every rung against the eager MLX encoder on an Apple M4 Pro (`mx.gpu`), 160-token fixture
(147 real tokens + 13 pads, so local sliding attention is exercised beyond one window).

| Rung | cosine vs HF oracle |
|---|---|
| embeddings | 0.9999999999999971 |
| layers 0–14 | ≥ 0.99999999999 |
| layers 15–21 | ≥ 0.99999999977 |
| **final_norm** | **0.9999999999918 — max_abs_diff 1.16e-4 — PASS** |

Every rung's cosine is ≈ 1.0 and the final encoder output matches to max-abs **1.16e-4**.

### On the "first-divergence at layer 11" flag
A strict per-rung max-abs threshold (2e-2) trips at `layer.11.hidden` (max-abs ≈ 0.04). This is **not**
an architecture bug: fed the exact HF layer-10 input, layer 11's attention output max-abs is 5.5e-6 and
its isolated block max-abs is 0.0034 — i.e. the layer is exact. The 0.04 is FP32 Metal-vs-CPU reduction
drift **amplified by a trained residual outlier activation** (HF magnitude 10673.1), and the final
LayerNorm normalizes it back to 1.16e-4. Cosine ≈ 1.0 at every rung confirms directional identity.
No tolerances were relaxed.

## Architecture implemented (differs materially from vanilla BERT)
- 22 layers, hidden 768, 12 heads (head_dim 64), GeGLU MLP (intermediate 1152), **no biases**, LayerNorm (not RMSNorm).
- **RoPE** (config's `position_embedding_type: absolute` is misleading — ModernBERT is rotary):
  `theta=160000` on global layers (`i % 3 == 0`), `theta=10000` on local layers.
- **Alternating attention**: full (global) every 3rd layer; otherwise local sliding window of ±64
  (inclusive `abs(q-k) <= 64`, key-position masking — verified against Transformers' eager mask by exact array comparison).
- Embedding LayerNorm; tied MLM head; RoPE duplicates the 32 half-dim frequencies before rotate-half.

## Reproduce
```bash
python3 capture_oracle.py --model <local ModernBERT-base dir> --output oracle.npz
python3 run_parity.py --model <local ModernBERT-base dir> --oracle oracle.npz \
  --result result.json --status status.txt --details details.json
```
Pass `--delete-oracle` to remove the large NPZ after the result is written; by
default a caller-supplied oracle is kept for reuse.
`parity-details.json` here is the checked-in per-rung evidence from the verified run. Model weights are
not included; point `--model` at a local dereferenced ModernBERT-base checkout (the repo also ships an
`onnx/` subdir — ignore it; use only `model.safetensors`).

## Why it isn't auto-scaffolded yet
`modernbert` routes ambiguously (score 7.4 < 8.0 floor — unregistered) and the encoder scaffold generator
is BERT-restricted. Landing this as an automated family would need: an `architectures.yaml` registry entry,
an encoder generator branch for dual-θ RoPE + local/global sliding attention + GeGLU, and golden-scenario
coverage. This artifact is the correctness reference for that work.
