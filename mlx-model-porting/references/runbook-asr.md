# Runbook: automatic speech recognition

## Applies to

Whisper, wav2vec2/HuBERT/CTC, Conformer/Parakeet/Canary, RNNT/transducers, Moonshine, Voxtral, Qwen ASR, and hybrid speech encoders with autoregressive or non-autoregressive decoders.

## Architecture fingerprint

Record:

- sample rate, channel handling, resampling and VAD;
- feature frontend: waveform conv, log-mel, STFT, learned encoder;
- encoder type and subsampling factor;
- decoder type: CTC, RNNT, attention seq2seq, LLM decoder, NAR editor;
- tokenizer, language/task prompts, timestamp tokens;
- streaming chunk, lookahead, left context, and state;
- beam/search/logit filtering and endpointing;
- alignment/timestamp postprocessing.

## Source oracle checkpoints

Capture:

1. resampled/normalized waveform;
2. frontend features and frame count;
3. encoder output at selected layers;
4. CTC logits or decoder/cross-attention logits;
5. RNNT predictor/joiner states where applicable;
6. selected token IDs before text normalization;
7. timestamps/alignments;
8. streaming state after each chunk.

## Weight conversion

- port frontend convolution and feature normalization exactly;
- preserve subsampling padding and frame-rate math;
- map encoder attention/conv modules;
- port decoder via encoder-decoder/dense runbook;
- preserve CTC blank ID, RNNT predictor/joiner, timestamp IDs;
- store tokenizer and language/task config;
- map adapters/projectors between audio encoder and LLM.

## Minimal MLX path

1. Verify waveform preprocessing and features.
2. Port encoder and compare hidden states.
3. Implement simplest deterministic decoder: greedy CTC, greedy seq2seq, or greedy RNNT.
4. Compare token IDs and transcript.
5. Add timestamps/beam/search.
6. Add batching/long-audio chunking.
7. Add streaming state and endpointing.

## Parity traps

- resampler or log-mel convention;
- frame padding/centering and subsampling off-by-one;
- CTC blank collapse/order;
- RNNT state update and blank handling;
- Whisper forced language/task/timestamp tokens;
- no-speech or suppression processors omitted;
- chunk timestamp offsets drift;
- text normalization differences mistaken for acoustic errors;
- VAD clipping initial/final phonemes.

## Optimization ladder

1. Cache frontend windows/filterbanks and remove host transfers.
2. Compile encoder for length buckets.
3. Batch offline audio with masks; bucket by duration.
4. Stream encoder with state/cache where trained for it.
5. Quantize encoder large linears/convs conservatively and evaluate WER.
6. Cache cross-attention projections for autoregressive decoder.
7. Use decoder cache/speculation where compatible.
8. Add VAD gating as a product pipeline with recall checks, not a model speed claim.
9. Optimize beam/RNNT search only after greedy parity.

## Completion gates

- WER/CER and language coverage versus source;
- timestamps and segment boundaries checked;
- short, long, silence, noise, and code-switch fixtures;
- offline versus streaming quality and latency;
- time to first partial/final transcript and RTF;
- VAD/search/preprocessing time included in end-to-end metrics.
