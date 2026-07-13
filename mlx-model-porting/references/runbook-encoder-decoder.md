# Runbook: encoder-decoder Transformer

## Applies to

T5/BART-style text-to-text models, Whisper-like sequence-to-sequence ASR, translation models, and other architectures with an encoded source and autoregressive decoder cross-attention.

## Architecture fingerprint

Record:

- encoder and decoder block counts/dimensions;
- shared or separate embeddings;
- self- and cross-attention head geometry;
- relative/absolute/rotary positions;
- decoder-start and forced tokens;
- encoder output projection/adaptor;
- cross-attention KV caching policy;
- beam search or timestamp logic;
- tied output head and logit scaling.

## Source oracle checkpoints

Capture:

1. preprocessed encoder input;
2. encoder frontend/embedding;
3. encoder hidden states;
4. decoder input IDs;
5. decoder self-attention Q/K/V;
6. cross-attention Q and encoder K/V;
7. self and cross caches after prefill/steps;
8. final norm/logits;
9. generated token sequence before text decoding.

## Weight conversion

- separate encoder, decoder self-attention, and cross-attention keys;
- preserve shared embeddings/tied head identity;
- map relative-position buckets and biases;
- verify cross-attention projection dimensions;
- preserve frontend convolution and feature normalization for speech;
- record forced/suppressed token configuration separately from weights.

## Minimal MLX path

1. Port encoder and achieve standalone hidden-state parity.
2. Port decoder without cache using fixed encoder states.
3. Add decoder self-cache.
4. Add reusable cross-attention K/V cache when source does so.
5. Implement deterministic greedy generation.
6. Add beam/timestamp/sampling behavior only after token-logit parity.

The built-in scaffolder currently implements the non-gated ReLU T5 path. It
uses mean-square-only T5 LayerNorm, first-layer-owned relative-position bias
shared across each self-attention stack, tied embeddings with T5 output
scaling, causal decoder self-cache, and reusable encoder cross-attention K/V.
BART, NLLB, Whisper, gated T5 variants, and other aliases still require their
runbook-specific graph implementation and fail closed in the generator.

The encoder-decoder capture contract orders `encoder.embed`,
`encoder.layer.{i}.hidden`, `encoder.final_norm`, `decoder_input_ids`,
`decoder.embed`, `decoder.layer.{i}.cross_attention`,
`decoder.layer.{i}.hidden`, `decoder.final_norm`, `logits`, and exact greedy
IDs. Decoder captures represent the fixed first step beginning at
`decoder_start_token_id`; generation continues against the same encoder memory.

## Parity traps

- encoder padding mask versus decoder causal mask confusion;
- decoder-start token or shifted labels;
- relative-position buckets differ for encoder, decoder, cross paths;
- cross-attention K/V recomputed or cached at wrong precision;
- source rescales embeddings or logits;
- Whisper-style timestamp/no-speech/logit filters omitted;
- multilingual task/language prompt tokens omitted;
- beam-search length penalty or cache reorder wrong.

## Optimization ladder

1. Cache encoder output and cross-attention projections.
2. Use fast SDPA independently for encoder, decoder self, and cross attention where valid.
3. Compile encoder and single-step decoder as separate stable regions.
4. Chunk long encoder input if architecture supports it.
5. Quantize encoder and decoder large linears separately; assess quality contribution.
6. Batch encoder work and continuously batch decoder requests if serving warrants it.
7. Use speculative decoding only for the autoregressive decoder and only when conditioning/cross-attention is compatible.

## Completion gates

- encoder parity passes independently;
- decoder full versus incremental parity passes;
- cross-cache reuse does not alter logits;
- forced/suppressed token and beam reorder tests pass;
- task quality is measured on representative source lengths;
- benchmark separates encoder, first decoder token, and steady decode.
