# Model support and reference map

This file guides source selection; it is not a promise that every upstream checkpoint works unchanged.

| Family | First MLX reference | Secondary reference | Primary runbook |
|---|---|---|---|
| Dense decoder LM | MLX-LM model implementations and generation/cache code | MLX-VLM language backbones | `runbook-decoder-transformer.md` |
| Sparse MoE LM | MLX-LM MoE implementations | serving projects with expert policies | `runbook-moe-transformer.md` |
| Encoder LM | MLX examples / embedding projects | standalone ports | `runbook-encoder-transformer.md` |
| Encoder-decoder / Whisper | MLX Whisper example, MLX-Audio STT | MLX-VLM audio paths | `runbook-encoder-decoder.md`, `runbook-asr.md` |
| Mamba/SSM/hybrid | MLX-LM supported hybrid models | standalone MLX ports | `runbook-ssm-hybrid.md` |
| Vision-language / omni | MLX-VLM | MLX-Audio for speech branches | `runbook-multimodal-omni.md` |
| Diffusion / flow | MLX examples and active image/video ports | model-specific repositories | `runbook-diffusion-flow.md` |
| Neural audio codec | MLX-Audio codec implementations; MLX EnCodec example | original codec repository | `runbook-audio-codec.md` |
| Autoregressive audio/TTS | MLX-Audio TTS implementations | MLX-LM backbones | `runbook-autoregressive-audio.md` |
| Flow/diffusion TTS | MLX-Audio model implementations | original architecture repository | `runbook-flow-tts.md` |
| Vocoder | MLX-Audio codec/vocoder implementations | MLX examples and original repo | `runbook-vocoder.md` |
| ASR | MLX-Audio STT implementations | MLX Whisper example | `runbook-asr.md` |
| Streaming speech | MLX-Audio realtime/STT/STS implementations | original streaming model repo | `runbook-streaming-speech.md` |
| Separation/enhancement | MLX-Audio STS models | original model repository | `runbook-separation-enhancement.md` |

Before copying a pattern, inspect its revision, tests, config schema, and weight conversion path. Similar class names do not imply identical cache or tensor semantics.
