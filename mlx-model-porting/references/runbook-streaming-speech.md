# Runbook: streaming speech and speech-to-speech

## Applies to

Realtime ASR, streaming TTS, Moshi/full-duplex speech LMs, Voxtral realtime, RNNT/streaming Conformer, cascaded STS, and duplex conversational models.

## Architecture fingerprint

Record:

- input/output sample rate and chunk size;
- algorithmic lookahead and receptive field;
- encoder, decoder, codec, and vocoder states;
- clock/token-rate relationship across streams;
- duplex/interruption/turn-taking policy;
- VAD/endpointing and buffering;
- overlap-add/windowing;
- partial result revision policy;
- warmup and first-chunk behavior.

## Source oracle checkpoints

Feed the same waveform under multiple chunk partitions. Capture per chunk:

1. buffered input and consumed samples;
2. frontend state/features;
3. encoder/recurrent state;
4. emitted token IDs or latents;
5. decoder/codec state;
6. output samples and overlap buffer;
7. timestamps and end-of-stream flush.

A correct stream should be invariant, within the architecture’s declared tolerance, to legal chunk partitioning.

## State contract

Define a serializable state object with:

- tensor name, shape, dtype;
- initialization and reset;
- update order;
- ownership and batch dimension;
- maximum history/window;
- flush/finalization behavior;
- version/model compatibility.

Do not hide state in module globals.

## Minimal MLX path

1. Build offline source parity.
2. Build a slow streaming oracle using explicit buffers.
3. Compare legal chunk partitions.
4. Add persistent MLX state and remove recomputation.
5. Add partial emissions and final flush.
6. Add real-time I/O only after deterministic file-based tests.

## Parity traps

- samples consumed/emitted off by one chunk;
- context reset or duplicated;
- output overlap window not complementary;
- different frontend padding online/offline;
- token/audio clocks drift;
- flush drops tail or emits duplicate data;
- interruption leaves stale speaker/cache state;
- asynchronous streams race without dependency;
- benchmark excludes buffering or resampling.

## Optimization ladder

1. Keep all persistent state allocated and update in place/functionally without copies.
2. Compile fixed-size chunk functions.
3. Bucket or pad input chunks rather than retracing arbitrary sizes.
4. Overlap independent CPU I/O/preprocessing with GPU work.
5. Cache reference/speaker/context features.
6. Stream codec/vocoder with tested overlap-add.
7. Quantize stable backbone components while checking long-duration drift.
8. Use speculative speech token generation only with streaming-safe acceptance and rollback.
9. Tune chunk size jointly for latency, quality, and throughput.

## Current MLX-Audio surfaces

Validate the exact route before using streaming claims:

- Python TTS generation can yield streaming chunks where supported by the model.
- Qwen3-TTS documents `stream=True` across its generation methods and `batch_generate` for concurrent short-prompt workloads.
- STT streaming surfaces vary by model: some use `generate(..., stream=True)`, while VibeVoice-style routes expose model-specific `stream_transcribe`.
- API server routes include HTTP streaming for TTS/STT and a realtime WebSocket transcription endpoint.

Qwen3-TTS batch throughput numbers in MLX-Audio docs are source-reported for 6-bit short prompts. Reproduce them on the target Mac, text length, voice/reference mix, quantization recipe, and streaming interval before quoting them.

Generic audio prefix caching is not proven by text/VLM prefix-cache results. Cache keys for audio must include model, tokenizer, codec/vocoder revision, quantization, speaker/reference fingerprint, chunking policy, cache format, and privacy namespace.

## Completion gates

- chunk-partition invariance tests;
- reset, interruption, cancellation, and flush tests;
- long-duration memory does not grow unbounded;
- first partial/audio latency, chunk P50/P95, jitter, and end-to-end RTF;
- batch size/concurrency, streaming interval, first chunk frames, codec context, cache state, and API surface;
- boundary clicks, timestamp drift, and quality measured;
- real-time claim includes audio I/O/preprocessing where applicable.
