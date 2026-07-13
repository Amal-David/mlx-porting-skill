# Optimized port: Qwen2.5-0.5B-Instruct

This worked example applies the structured optimization loop to the cached
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`. It contrasts the naive MLX-LM
4-bit default with the best candidate that actually held the declared quality
bar.

No model weights, tensors, benchmark receipts, or sealed evidence are checked
in. Converted artifacts remained under `$WORK`.

## Local protocol

The measurement ran on 2026-07-13 on an Apple M4 Pro with 48 GiB unified
memory, macOS 26.3, Python 3.14.4, MLX 0.32.0, and MLX-LM 0.31.3. Metal was
available and MLX selected `Device(gpu, 0)`.

- Reference and candidates were converted from the same cached source with
  `trust_remote_code=False`; non-quantized tensors were saved as bf16.
- Decode used a fixed 14-token prompt, greedy generation, one warmup, and five
  timed 64-token runs. Timing excluded the first yielded token/prefill, so each
  run timed 63 decode tokens.
- Throughput is the median of five timed runs; CV is the population coefficient
  of variation across those runs.
- Peak memory is the MLX Metal peak after resetting the counter before each
  model load.
- Quality reused `quant_quality_gate.py`: 72 held next-token positions plus four
  fixed greedy prompts at 32 tokens each.
- The declared bar was perplexity ratio at most 1.10 and zero candidate-only
  degenerate outputs. Exact output and first-token agreement remained reported
  diagnostics rather than extra default gates.

## Measured local observations

| Config | Median tok/s | CV | Speedup vs bf16 (×) | Peak GiB | Memory reduction | Perplexity ratio | Exact output | First token | Candidate-only degenerate | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 | 132.75 | 1.86% | 1.00 | 0.956 | 0.0% | 1.0000 | 4/4 | 4/4 | 0/4 | reference |
| 8-bit, group 64 | 199.07 | 0.73% | 1.50 | 0.517 | 45.9% | 1.0009 | 1/4 | 3/4 | 0/4 | pass |
| 4-bit, group 64 | 292.43 | 0.22% | 2.20 | 0.287 | 70.0% | 1.2964 | 0/4 | 2/4 | 0/4 | **fail** |
| 4-bit, group 32 | 264.74 | 0.52% | 1.99 | 0.318 | 66.8% | 1.1848 | 0/4 | 4/4 | 2/4 | **fail** |

These values are local user-diagnostic observations, not portable speed or
memory claims. Throughput has run-to-run variance and will change with model,
prompt shape, generated length, thermal state, Mac, and software version.

The bf16 held-text perplexity was 82.3697. The 8-bit candidate measured
82.4431, while 4-bit/group-64 measured 106.7827 and 4-bit/group-32 measured
97.5889.

## Naive default versus structured optimum

`mlx_lm.convert --quantize` defaults to affine 4-bit/group-64 for this runtime.
That naive candidate produced the highest measured local throughput and lowest
memory, but its 1.2964 perplexity ratio exceeded the 1.10 quality limit. It was
therefore ineligible.

The loop recommends **8-bit/group-64**. It was the only quantized candidate to
pass: median decode rose from 132.75 to 199.07 tok/s, peak Metal memory fell
from 0.956 to 0.517 GiB, the perplexity ratio was 1.0009, and no candidate-only
degeneration was detected. Its 1/4 exact-output score is retained explicitly;
the default bar holds perplexity and degeneration, not exact continuation
identity. An operator who requires closer token identity can raise
`--min-firsttoken-agreement` or substitute a stricter task-specific gate.

Rollback condition: keep bf16 if the 8-bit artifact fails the declared quality
bar on the user's representative prompts or if its measured advantage falls
within local variance. Do not select either 4-bit artifact from this run.

## Reproduce locally

```bash
export MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
export WORK="$HOME/.cache/mlx-porting-work/opt"

python3 mlx-model-porting/scripts/optimize_port.py \
  --model "$MODEL" \
  --work-dir "$WORK"
```

The tool creates `$WORK/bf16`, `$WORK/8bit`, `$WORK/4bit-g64`, and
`$WORK/4bit-g32`. Strict JSON goes to stdout; progress and the readable table go
to stderr. The output carries `promotable_claim=false` and
`writes_sealed_evidence=false`. It must not be placed in the benchmark-receipt,
assessment, effective-claim, or sealed-evidence pipeline.

