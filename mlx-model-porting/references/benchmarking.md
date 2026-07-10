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
- MLX active/cache/peak memory counters, and any `set_memory_limit` experiment;
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

## Memory fit gates

Do not say a model "fits" because one prompt completed once. Record MLX peak memory and run an explicit memory-budget experiment when memory is part of the claim. `mx.get_peak_memory()` can support peak measurement, and `mx.set_memory_limit()` can test graph-evaluation behavior under pressure, but the limit is not a quality, latency, or system-stability proof by itself.

Treat operator-level system settings such as macOS `iogpu.wired_limit_mb` as environment notes only. Never run or recommend them as an automatic optimization, and keep them out of reproducible benchmark commands unless the user explicitly chose that operating configuration.

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

For a controlled family-neutral observation, pass `--receipt-spec` and a schema-2
`--quality-contract`. The spec's argv template may contain only exact literals
or allowlisted sources from `models.target`, workload parameters/artifact paths,
and `variant_config`; it must bind `models.target.id`,
`models.target.revision`, and exactly one digest-pinned Python runner as argv
position 1. The adapter hashes the resolved interpreter and installed-package
metadata, sanitizes and binds the ambient environment, launches Python with
`-I -B`, and executes from the
receipt root with bounded
run/warmup/output settings, and records only parent-measured `wall_seconds`.
Text printed by the child process is raw evidence, never a performance metric.
Every measured run must recreate the exact quality candidate under
`quality/outputs/<label>/`; the raw report binds that per-run digest to the
built-in quality contract.

Receipt mode statically rejects symlinked artifact/output components, snapshots the quality
contract before the child starts, and verifies the source contract did not
change. Target and source revisions, the exact root-level baseline, and its
structural gates are checked before the measured command runs; stability is
still assessed from the complete pair. A compatible
repetition must keep the exact baseline file/digest, semantic experiment
(including warmup/run counts and timeout), and exact-output quality contract. Receipt-specific raw
measurements remain in each full fingerprint; the conservative minimum
repetition supplies any future promoted ratio and fingerprint.

These controls do not prove arbitrary runner semantics or installed dependency
bytes. The `execution_attested` gate therefore remains false for the generic
runner and legacy MLX-LM lanes, making their receipts observations. Promotion
requires a future reviewed built-in adapter that independently binds executed
package bytes, model/workload use, and the output generated by every measured
run.

The receipt spec has exactly these nine top-level fields; artifact paths and
digests are relative to the directory that will contain the receipt:

```json
{
  "schema_version": 1,
  "label": "candidate",
  "argv_template": [
    {"literal": "python3"},
    {"source": ["workload", "artifacts", 0, "path"]},
    {"literal": "--model"},
    {"source": ["models", "target", "id"]},
    {"literal": "--revision"},
    {"source": ["models", "target", "revision"]},
    {"literal": "--input"},
    {"source": ["workload", "artifacts", 1, "path"]},
    {"literal": "--mode"},
    {"source": ["variant_config", "mode"]},
    {"literal": "--output"},
    {"source": ["variant_config", "quality_output_path"]}
  ],
  "models": {
    "target": {
      "id": "owner/model",
      "revision": "<pinned 40-64 hex revision>",
      "lineage_id": "controlled-lineage",
      "source_id": "owner/source-model",
      "source_revision": "<pinned 40-64 hex revision>"
    }
  },
  "workload": {
    "id": "controlled-workload",
    "artifacts": [
      {"role": "runner", "path": "runner.py", "sha256": "<sha256>", "size_bytes": 1234},
      {"role": "input", "path": "input.bin", "sha256": "<sha256>", "size_bytes": 5678}
    ],
    "parameters": {"shape": "fixed"}
  },
  "variant_config": {
    "mode": "candidate",
    "quality_output_path": "quality/outputs/candidate/result.txt"
  },
  "enabled_methods": ["method-id"],
  "comparison_role": "candidate",
  "rollback_condition": "Rollback on quality failure or a gain within noise."
}
```

```bash
python3 mlx-model-porting/scripts/benchmark_command.py \
  --receipt-spec candidate.spec.json \
  --quality-contract quality-contract.json \
  --baseline-receipt baseline.json \
  --warmup 1 --runs 5 --timeout 600 \
  --output candidate.json
```

Receipt mode rejects environment overrides because redacted values cannot be
compared across baseline and candidate. Express a controlled variant through
`variant_config` and bind it into `argv_template` instead. It also requires an
explicit finite positive `--timeout` of at most 3600 seconds. Warmup count,
measured-run count, and timeout are part of the experiment protocol and must
match across compatible repetitions. Receipt mode refuses to overwrite an
existing receipt or collide with a baseline, workload, quality-control, raw, or
bound-output path; use a new label for a new evidence run.

## Receipt harness

`scripts/benchmark_generation.py` wraps `mlx_lm generate`-style commands and writes receipt JSON under `assets/benchmarks/` by default. Each measured run must print prompt tokens/s, generation tokens/s, and peak memory lines; missing metrics or nonzero exits fail loudly. Receipts include environment metadata, exact command arguments, config notes, per-run metrics, aggregate median/min/max values, and a labeled `ttft_proxy` computed as prompt tokens divided by prompt throughput. The harness records `speedup_vs_baseline` only when `--baseline-receipt` is provided, so standalone receipts never contain speedup numbers.
