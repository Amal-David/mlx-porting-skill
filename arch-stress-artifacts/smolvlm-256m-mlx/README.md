# SmolVLM-256M (vision-language) → MLX: verified worked port (arch stress-test, 2026-07-15)

A standalone eager MLX port of **`HuggingFaceTB/SmolVLM-256M-Instruct`** (Idefics3), proving the
**VLM (vision-language) architecture class** ports to MLX with full-stack parity. Proof artifact,
outside the skill payload.

A VLM decomposes into pieces this toolkit already proves the method on:
- **Vision tower**: 12-layer Idefics3/SigLIP-style ViT (patch-conv embed + learned position embed,
  GELU, LayerNorm, full attention) — encoder-class (same method as the verified ModernBERT port).
- **Connector**: pixel-shuffle (scale_factor 4) + MLP projector to the text hidden size.
- **Text tower**: Llama 30-layer dense decoder — the proven family.
- **Fusion**: projected image embeddings spliced into the text token embeddings at placeholder positions.

## Result: full-stack parity on real Metal
Frozen single-image + text fixture; HF Idefics3 eager oracle vs eager MLX on `mx.gpu`.

| Stage | result |
|---|---|
| vision patch-embed → all 12 ViT layers → vision final norm | ✅ pass (min cos 0.9999999999 at vision_final) |
| pixel-shuffle + projector connector | ✅ pass |
| fused image+text input embeddings | ✅ pass |
| text decoder (all layers) | ✅ pass, cos ~1.0 |
| logits | ✅ pass |
| **first divergence** | **none** |

The `max_abs ~0.0068` from text_layer_11 onward is a benign image-token outlier activation (cosine
stays ~0.99999999999), the same pattern seen in the ModernBERT port; no tolerances were relaxed.

## Honest scope / caveats
- One frozen single-image fixture, prefill only. Multi-image, video, generation/decoding loop, and
  broader fixtures are not claimed.
- Independent oracle re-capture was not repeated; evidence is the per-stage cos ~1.0 parity + a real
  (non-stub) `model.py` covering vision + connector + fusion + text.
- Correctness reference, not an auto-scaffolded family — landing VLMs in the executable generator
  (vision tower families + connector + image pipeline) is the follow-up this artifact de-risks.

## Files
`model.py` (eager MLX Idefics3 VLM: vision + connector + fusion + text), `parity.json` (per-stage
evidence). No model weights included.
