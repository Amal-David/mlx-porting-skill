# Runbook: autoregressive audio LM and codec-token TTS

## Applies to

Bark/VALL-E/SoundStorm-like token generators, Qwen3-TTS, Fish Speech, Moshi/Sesame, VibeVoice, Zonos, Spark, Chatterbox, MOSS-style systems, and text/speech LMs that predict semantic or acoustic tokens autoregressively.

## Architecture fingerprint

Record:

- text tokenizer and audio tokenizer/codec revisions;
- semantic, coarse, fine, acoustic, or multi-stream token spaces;
- codebook count, rate, offsets, and special IDs;
- flattened, delayed, interleaved, parallel, or dual-track schedule;
- text/audio conditioning and reference/ICL encoding;
- speaker/style embeddings;
- transformer family, cache, and positional encoding;
- stopping rules and length/duration controls;
- codec/vocoder decode pipeline;
- streaming unit and minimum lookahead.

## Source oracle checkpoints

Capture:

1. normalized text and token IDs;
2. reference audio features/tokens/embeddings;
3. constructed multimodal prompt sequence;
4. delay/interleave mask and position IDs;
5. first-step and several-step logits for every stream/head;
6. selected tokens before any offset conversion;
7. reconstructed code matrix;
8. codec/vocoder output and streaming chunks.

## Weight conversion

- route the language backbone through the dense/MoE/hybrid runbook;
- preserve multiple embeddings/heads and token offsets;
- map stream-specific projections and conditioning adapters;
- preserve speaker/reference encoders;
- map codec/vocoder weights with their dedicated runbooks;
- store prompt templates, delay pattern, codebook metadata, and special IDs in config;
- do not merge components from different upstream revisions without compatibility tests.

## Minimal MLX path

1. Port/load codec and verify source token decode.
2. Reconstruct exact prompt and schedule without generation.
3. Port model forward for fixed token sequences.
4. Verify per-stream logits.
5. Implement deterministic generation one stream/schedule step at a time.
6. Reconstruct codebooks and decode offline.
7. Add reference/voice conditioning.
8. Add streaming output and chunk flush.

## Parity traps

- audio token offsets applied twice or omitted;
- codebook order or delay pattern shifted;
- semantic and acoustic rates confused;
- text/reference prompt template differs;
- reference audio resampling/loudness differs;
- independent stream caches advanced inconsistently;
- stopping one stream prematurely;
- sampling parameters shared when source uses per-stream settings;
- codec version incompatible with generated tokens;
- streaming decoder lacks enough future/receptive context.

## Optimization ladder

1. Cache reference/ICL encodings and static prompt state with a versioned key.
2. Compile stable decode step with explicit multi-stream cache.
3. Batch or parallelize codebook heads where schedule allows.
4. Use native fast attention and weight quantization on the LM backbone.
5. Stream codec/vocoder decode concurrently only when dependencies and chunk quality are proven.
6. Use overlap-add and state reuse to reduce first-audio latency.
7. Evaluate MTP/Medusa/draft-model speech speculation only with schedule-aware acceptance.
8. Quantize codec/vocoder separately from the LM; retain sensitive output layers.
9. For dialogue models, cache speaker/context state and test long turn stability.

## Quality and performance gates

- token/logit parity before waveform review;
- intelligibility, speaker similarity, duration, language, and prosody checks;
- no codebook/delay corruption under long generation;
- time to first audio, steady RTF, and total RTF reported separately;
- offline and streaming outputs compared at boundaries;
- reference cache invalidates on codec/model/processor changes;
- any speculative path reports acceptance and quality, not only speed.
