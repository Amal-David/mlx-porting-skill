# HuBERT base LS960 port plan

- Source: local cached `facebook/hubert-base-ls960` config and safetensors,
  with the two cache revisions recorded in `README.md`.
- License: missing from cached artifact; redistribution remains blocked.
- Task: inference-time acoustic encoder parity only.
- Source runtime: Transformers 5.3.0 and Torch 2.9.1, built-in `HubertModel`,
  offline and without remote code.
- Architecture: seven-layer waveform convolutional frontend, feature
  projection, convolutional positional embedding, 12 post-norm transformer
  encoder layers; no decoder in the base model.
- Target: standalone eager MLX, FP32, Apple M4 Pro / Metal.
- Fixture: seeded mono waveform (`seed=7006`, 16,000 samples); Torch-extracted
  features are the shared model input.
- Correctness gate: `input_features`, `embed`, all 12 post-layer hidden states,
  and `final_hidden` at `atol=1e-4`, `rtol=1e-4`, cosine >= `0.99999`.
- Rollback: any failed rung, unmapped encoder tensor, unsupported config, or
  attempt to claim waveform/CTC/Whisper support.

