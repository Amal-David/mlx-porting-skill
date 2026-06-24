# Benchmarking and decision gates

## Benchmark questions

Every benchmark must answer a product question: lower first-token latency, faster steady decode, higher concurrent throughput, lower memory, lower real-time factor, earlier first audio, or lower energy. A kernel benchmark alone is not a product result.

## Mandatory metadata

Record:

- Mac model/chip, GPU cores, unified memory, power mode, thermal state;
- macOS, Python, MLX, MLX-LM/VLM/Audio and package revisions;
- model and exact artifact revision;
- dtype, weight quantization, KV/state quantization;
- prompt/audio/image length, generated length, batch/concurrency;
- cache state, compile state, warm/cold condition;
- seed and sampling parameters;
- measured quality gate.
- for cache tests, cold/warm state, cache key fields, hit/miss counts, eviction policy, namespace/tenant setting, and persistence path;
- for compiled tests, first-run compile time, warm-run timing, suspected retrace count, and shape/dtype variants;
- for audio, streaming interval, first chunk frames, codec/vocoder context, API surface, and batch/concurrency.

## Metric decomposition

### Language and autoregressive models

- load time;
- compile/first-run time;
- time to first token (TTFT);
- prefill tokens/s;
- decode tokens/s and inter-token latency;
- end-to-end request latency;
- throughput under concurrency;
- peak and steady memory;
- cache bytes/token/layer.

### Audio and speech

- load and warmup;
- time to first audio/transcript;
- real-time factor for each stage and end-to-end;
- semantic/acoustic token rates;
- codec/vocoder time;
- chunk latency and jitter;
- peak and steady memory;
- quality and boundary-continuity metrics.

### Diffusion/flow

- preprocess/encode;
- first compiled step;
- per-step latency distribution;
- decode/postprocess;
- total latency and memory;
- fixed-seed quality comparison.

## Experimental protocol

1. Isolate baseline and candidate in separate processes when allocator/cache carryover matters.
2. Run warmups explicitly and report cold performance separately.
3. Use identical inputs and generation length.
4. Run enough repetitions to report median and tail, not only the best run.
5. Capture raw results in JSON/CSV.
6. Pair performance with correctness/quality results.
7. Re-run the baseline after the candidate to detect thermal drift.
8. Benchmark realistic and adversarial sizes; many kernels switch paths by shape.
9. Write the rollback condition before running the candidate, then apply the same condition to the raw results.

## Claim record

For each accepted or rejected optimization, record:

- hypothesis and expected bottleneck;
- evidence class: official API, source code, pinned MLX repo, paper, technical blog, or local benchmark artifact;
- support scope: official MLX, official MLX project, third-party pinned, paper-only, context-only, or locally reproduced;
- target hardware/software/workload;
- correctness and quality result;
- raw benchmark artifact path;
- keep/revert decision and rollback condition.

## Keep/revert gate

Keep an optimization only when:

- all correctness gates pass;
- the target end-to-end metric improves beyond noise;
- memory does not violate budget;
- quality remains within the declared bound;
- complexity and maintenance cost are justified;
- the result survives at least two relevant workload sizes.

Revert when the win exists only in a microbenchmark, only after excluding first-run cost without disclosure, or only on an unrepresentative shape.

## Using the command harness

`benchmark_command.py` measures command wall time, return status, optional process RSS when `psutil` is present, stdout/stderr summaries, and environment metadata. Model-specific scripts should additionally emit TTFT, tokens/s, RTF, and quality metrics to files referenced in the report.
