# Worked port: `t5-small`

This packet records a real offline FP32 source-to-MLX port of the locally cached
Hugging Face `t5-small` snapshot
`df1b051c49625cf57a3d0d8d3863ed4d13564fe4` on an Apple M4 Pro. No model
weights or captured tensors are checked in.

## Scope and architecture decisions

- `SOURCE_PROVENANCE.json` binds the Hub revision, the original config digest,
  and the deterministic artifact overlay. The overlay adds only
  `tie_word_embeddings=true` to `config.json` (the Transformers T5 default) and
  an artifact-bound copy of the Apache-2.0 terms; it does not inject a license
  declaration into model config. The original weight bytes remain unchanged.
- License applicability is pinned to the immutable upstream model-card URL in
  `SOURCE_PROVENANCE.json`. Those model-card bytes were not present in the
  offline cache, so this is recorded as upstream evidence, not as a locally
  verified model-card digest, and redistribution compatibility remains
  unassessed.
- `shared.weight` is the sole embedding/tied-head owner. Decoder final hidden
  states are scaled by `d_model ** -0.5` before the tied projection.
- Encoder and decoder self-attention share their first-layer bucketed relative
  position bias across all stack layers. Cross-attention has no relative bias.
- The checkpoint contains one unused legacy
  `decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight`;
  `WEIGHT_MAP.json` ignores it explicitly instead of inventing a target.
- T5 LayerNorm uses mean-square normalization only, with no mean subtraction
  and no bias. The feed-forward path is the non-gated ReLU variant.
- Decoder self-attention grows a KV cache. Encoder cross-attention K/V is
  projected once per layer and reused during greedy decoding.

## Reproduce offline

```bash
export WORK="$HOME/.cache/mlx-porting-work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cp -aL "$HOME/.cache/huggingface/hub/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/." "$WORK/t5-small/"
jq -S '. + {tie_word_embeddings: true}' "$WORK/t5-small/config.json" > "$WORK/t5-small/config.overlay.json"
mv "$WORK/t5-small/config.overlay.json" "$WORK/t5-small/config.json"
cp mlx-model-porting/LICENSE "$WORK/t5-small/LICENSE"
shasum -a 256 "$WORK/t5-small/config.json" "$WORK/t5-small/LICENSE"
python3 mlx-model-porting/scripts/inspect_model.py "$WORK/t5-small" --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --output "$WORK/t5-small-inspection.json"
python3 mlx-model-porting/scripts/scaffold_port.py "$WORK/t5-small-inspection.json" --artifact-root "$WORK/t5-small" --output "$WORK/t5-small-mlx"
python3 mlx-model-porting/scripts/convert_checkpoint.py --source "$WORK/t5-small" --mapping mlx-model-porting/examples/worked-port-t5-small/WEIGHT_MAP.json --output "$WORK/t5-small-converted"
python3 mlx-model-porting/scripts/validate_weight_map.py --source "$WORK/t5-small-inspection.json" --target "$WORK/t5-small-mlx/scaffold-manifest.json" --mapping mlx-model-porting/examples/worked-port-t5-small/WEIGHT_MAP.json
python3 mlx-model-porting/scripts/run_parity.py --source-model "$WORK/t5-small" --package "$WORK/t5-small-mlx" --weights "$WORK/t5-small-converted" --token-ids 13959 1566 12 2968 10 37 629 19 1627 5 1 --generate-steps 8 --atol 1e-3 --rtol 1e-4 --cosine-min 0.999999 --output "$WORK/t5-small-parity.json"
```

## Result

All 27 encoder-decoder rungs pass at the recorded FP32 cross-framework policy:
exact input IDs and attention mask; encoder embedding, six blocks, and final norm; decoder start and
embedding; all six cross-attention and post-block hidden states; decoder final
norm; first-step logits; and eight exact greedy tokens. The fixture produces the
nontrivial continuation `[644, 4598, 229, 19250, 5, 1, 1, 40]`. The largest
absolute difference is `0.0048828125`; the actual minimum floating-rung cosine
in `parity-report.json` is `0.9999999999999092`. Generated token IDs match
exactly. This supersedes the earlier report whose true minimum was
`0.9999999999993505`, not the previously claimed `> 0.9999999999998`.

The unmodified decoder-default tolerance (`atol=1e-5`, `rtol=1e-4`) is retained
in `parity-report-tight.json` and honestly stops at encoder layer 0 with
`max_abs=0.00048828125`. This packet does not change the tool default or hide
that stricter result.
