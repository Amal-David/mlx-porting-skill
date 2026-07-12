# Worked port: HuBERT base LS960 acoustic encoder

This example records a real offline FP32 port of the transformer-encoder core
of `facebook/hubert-base-ls960` on Apple M4 Pro / Metal. It contains no model
weights or tensor archives.

## Scope

Implemented:

- the HuBERT/Wav2Vec2 feature projection (`LayerNorm(512)` plus `512 -> 768`);
- the weight-normalized, 16-group convolutional positional embedding;
- all 12 bidirectional self-attention layers;
- GELU `768 -> 3072 -> 768` feed-forward blocks and post LayerNorms;
- source and MLX capture at extracted features, encoder input, every layer, and
  final hidden state.

Not implemented:

- the seven-layer raw-waveform convolutional feature extractor;
- CTC heads, greedy/beam CTC decoding, tokenization, transcripts, or WER;
- Whisper or any other autoregressive encoder-decoder graph.

The source oracle creates a seeded 16,000-sample waveform, runs the real Torch
HuBERT feature extractor, and freezes its `[1, 49, 512]` output as
`input_features`. That exact tensor is fed to both the Torch remainder and the
MLX encoder, so the passing ladder proves the real projection, positional
convolution, and transformer blocks without claiming raw-waveform parity.

## Local artifact

The cached Hugging Face snapshot was split across two local revisions. The
working copy was assembled without network access:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WORK="$HOME/.cache/mlx-porting-work"
mkdir -p "$WORK/hubert-base-ls960"
cp -L "$HOME/.cache/huggingface/hub/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a/config.json" "$WORK/hubert-base-ls960/config.json"
cp -L "$HOME/.cache/huggingface/hub/models--facebook--hubert-base-ls960/snapshots/af46f65f540dc3ca7aa59f46c6c3d5dbb4374fa8/model.safetensors" "$WORK/hubert-base-ls960/model.safetensors"
```

`inspect_model.py` correctly routed the model to
`automatic-speech-recognition`, but the cached bytes contain no model card or
license sidecar, so the inspection remains provenance-blocked. The checked-in
inspection preserves that blocker. The graph was generated directly from the
same inspected config/tensor inventory for local validation only; this example
must not be treated as permission to redistribute converted weights.

## Reproduce the validated stages

With a provenance-complete copy of the same checkpoint, the normal scaffold
command is:

```bash
python3 mlx-model-porting/scripts/scaffold_port.py \
  "$WORK/hubert-run/inspection.json" \
  --artifact-root "$WORK/hubert-base-ls960" \
  --output "$WORK/hubert-base-ls960-mlx"
```

Resolve the emitted draft to the checked-in `WEIGHT_MAP.json`, validate, and
convert in FP32:

```bash
python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source "$WORK/hubert-run/inspection.json" \
  --target "$WORK/hubert-base-ls960-mlx/scaffold-manifest.json" \
  --mapping mlx-model-porting/examples/worked-port-hubert-base-ls960/WEIGHT_MAP.json \
  --output "$WORK/hubert-run/weight-map-validation.json"

python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source "$WORK/hubert-base-ls960" \
  --mapping mlx-model-porting/examples/worked-port-hubert-base-ls960/WEIGHT_MAP.json \
  --output "$WORK/hubert-base-ls960-converted"
```

Run the real ladder:

```bash
python3 mlx-model-porting/scripts/run_parity.py \
  --source-model "$WORK/hubert-base-ls960" \
  --package "$WORK/hubert-base-ls960-mlx" \
  --weights "$WORK/hubert-base-ls960-converted" \
  --mode asr --waveform-samples 16000 --seed 7006 \
  --atol 1e-4 --rtol 1e-4 --cosine-min 0.99999 \
  --output "$WORK/hubert-run/parity-report.json"
```

## Result

All 15 rungs passed. `input_features` was exact. Maximum absolute error was
`4.0531e-06` at `embed`, grew gradually through the 12 real encoder layers,
and reached `6.4671e-05` at `layer.11.hidden` / `final_hidden`; final cosine
similarity was `0.9999999993744736`. See `parity-report.json` for every rung.
