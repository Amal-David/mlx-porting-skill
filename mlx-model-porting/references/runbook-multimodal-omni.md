# Runbook: vision-language, audio-language, and omni models

## Applies to

VLMs and omni models with image/video/audio encoders, projectors/resamplers, a language backbone, and optional speech/audio output tracks.

## Architecture fingerprint

Record:

- modality encoders and processor revisions;
- patch/frame/audio tokenization and rates;
- projector/resampler/perceiver/Q-former;
- placeholder tokens and prompt template;
- modality token insertion order and position encoding;
- shared/separate attention or cross-attention;
- language backbone family;
- output heads/tracks and synchronization;
- modality feature caching and serving behavior.

## Source oracle checkpoints

Capture:

1. exact processed pixel/video/audio tensors;
2. modality encoder features;
3. projector/resampler outputs;
4. placeholder/token replacement sequence;
5. position IDs and masks over mixed modalities;
6. language backbone hidden states/logits;
7. output speech/image/audio branch intermediates;
8. cached modality features and repeated-request behavior.

## Weight conversion

- port each modality encoder with its native runbook;
- map projector/resampler dimensions and normalization;
- preserve placeholder IDs and chat template;
- preserve temporal/spatial position conventions;
- route the language backbone through dense/MoE/hybrid runbook;
- port output codec/vocoder/decoder separately;
- keep processor versions pinned to the weights.

## Minimal MLX path

1. Port text-only backbone and verify.
2. Port one modality encoder and verify features.
3. Port projector and mixed-token assembly.
4. Verify conditioned logits with a tiny input.
5. Add multiple items/frames/audio segments.
6. Add output modalities and streaming.
7. Add serving feature cache and batching.

## Parity traps

- image normalization/crop/resize mismatch;
- audio frontend frame mismatch;
- projector output inserted at wrong placeholder;
- variable number of modality tokens changes positions incorrectly;
- processor chat template differs from source;
- modality encoder dtype/normalization causes amplified drift;
- cached features reused under a changed processor/model;
- mixed batches route wrong feature to request;
- output tracks lose synchronization.

## Optimization ladder

1. Cache immutable modality features with a strong namespace.
2. Batch encoder work separately from language decode.
3. Use fast attention and compile stable encoder/projector regions.
4. Compress/prune visual/audio tokens only as a measured quality mode.
5. Quantize modality encoder, projector, and LM independently.
6. Reuse prefix/cache for repeated context.
7. Add continuous batching with mixed-modality admission tests.
8. Use speculative decoding only on compatible autoregressive output and include conditioning cost.
9. Stream output branches with explicit synchronization.

### VLM cache taxonomy

- Pixel/preprocess cache: key by decoded media bytes plus resize, crop, FPS, max-frame, pixel-budget, processor revision, dtype, and tenant namespace.
- Projected-feature cache: key by model revision, processor revision, projector version, adapter, quantization, media content hash, and processing knobs. Compare cold/warm logits before trusting generated answers.
- Prompt KV/APC cache: require token prefix hash plus multimodal extra hashes. Reject reuse when the suffix contains media placeholders that are not fully covered by the restored cache.
- Disk warm cache: treat as persistence with privacy, deletion, versioning, checksum, and storage-pressure gates.

### Model-family gates

For Qwen-style VLMs, assert placeholder count equals projected feature count; preserve `image_grid_thw` and `video_grid_thw`; verify M-RoPE deltas under chunked prefill and cache reuse; and keep image/video ordering stable in mixed batches.

For LLaVA-style VLMs, assert projector output shape, feature select layer/strategy, image newline behavior, single-image versus multi-image limits, and merge positions. Replacing the projector or visual encoder is a new model, not an optimization of an existing checkpoint.

Generic visual-token pruning, merging, or FastVLM-style encoder replacement stays research-only until there is a reproducible MLX implementation, model-specific quality suite, benchmark metadata, and rollback threshold.

## Completion gates

- text-only and each single-modality path pass independently;
- mixed-modality ordering and multiple-item tests;
- processor/tokenizer/template pinned;
- cache namespace and cross-request isolation tested;
- cold/warm cache parity for projected features and prefix KV/APC;
- task quality for every advertised modality;
- benchmark includes modality preprocessing/encoding and not only LM decode.
