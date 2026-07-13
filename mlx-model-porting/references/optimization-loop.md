# Structured optimization loop

Use this loop when an already-supported `mlx_lm` model is correct but still
unoptimized. It turns “make this port fast” into a reproducible local decision:
establish one bf16 reference, change one quantization recipe at a time, pair
every performance result with the same quality workload, and keep only a
quality-passing candidate.

This is a user diagnostic. Its measurements describe one model, Mac, software
stack, and workload. They are not benchmark receipts, receipt assessments,
effective claims, or inputs to the sealed promotion pipeline.

## 1. Freeze the experiment

Before measuring:

- require an unquantized local model that loads through a built-in `mlx_lm`
  implementation without remote code;
- record the source revision, MLX and MLX-LM versions, Metal device, fixed
  prompt workload, seed, generated-token count, warmups, and timed-run count;
- choose the quality bar and rollback condition before seeing candidates; and
- keep the converted artifacts outside the repository.

The default tool protocol uses one fixed decode prompt, one warmup, five timed
64-token greedy generations, and the built-in `quant_quality_gate.py` workload.
Decode timing starts after the first yielded token, excluding prefill. The
report retains every timed throughput value, their median, and population CV.
Peak memory comes from the MLX Metal peak counter reset before each model load.

## 2. Establish bf16

Convert the source to bf16 with `mlx_lm.convert`, then load and measure it before
creating candidates. Capture the four fixed greedy reference outputs and the
held-text reference perplexity through `quant_quality_gate.py`'s scoring path.
Stop if the reference does not load, generation is not deterministic across
timed runs, perplexity is nonsensical, or Metal memory cannot be measured.

## 3. Sweep one dimension at a time

The default documented sweep is:

| Config ID | MLX-LM recipe | Role |
| --- | --- | --- |
| `8bit` | affine, 8-bit, group size 64 | conservative candidate |
| `4bit-g64` | affine, 4-bit, group size 64 | naive `mlx_lm.convert --quantize` default |
| `4bit-g32` | affine, 4-bit, group size 32 | smaller-group 4-bit candidate |

Each candidate is converted from the same source bytes, measured with the same
decode prompt and run counts, and evaluated against the same bf16 artifact.
Existing conversion directories are refused by default. `--reuse-existing`
accepts only loadable artifacts whose model type and quantization recipe match
the requested slot.

## 4. Gate quality

Do not implement a second perplexity or generation evaluator. Invoke
`quant_quality_gate.py` and retain its:

- candidate/reference perplexity ratio;
- greedy exact-match count and first-token agreement;
- candidate-only degenerate-output rate; and
- individual threshold checks.

The optimization loop defaults to perplexity ratio at most 1.10 and no
candidate-only degenerate output. Exact-output divergence is always reported;
it is diagnostic unless the operator raises `--min-firsttoken-agreement`.

## 5. Recommend or roll back

Failed candidates are ineligible regardless of speed or memory. Among passing
candidates, rank measured decode tokens per second per peak Metal byte. This is
an explicit local balance between throughput and memory, not a universal model
ranking.

If no candidate passes, leave `recommended_config_id` empty. Report the
least-degrading candidate separately as a warning-only fallback, ordered by
candidate-only degeneration, perplexity ratio, exact match, then local
efficiency. A failed fallback is not a quality-held recommendation. Roll back
to bf16 whenever the chosen candidate fails the declared bar on the user's real
task workload.

## 6. Run and interpret

```bash
export MODEL="$HOME/.cache/huggingface/hub/models--ORG--MODEL/snapshots/REVISION"
export WORK="$HOME/.cache/mlx-porting-work/opt"

python3 mlx-model-porting/scripts/optimize_port.py \
  --model "$MODEL" \
  --work-dir "$WORK"
```

Strict JSON is printed to stdout. Conversion progress and the human-readable
table are printed to stderr. The JSON carries
`promotable_claim=false` and `writes_sealed_evidence=false`; do not copy it into
`assets/benchmarks/`, claim generation, evidence indexing, or receipt
assessment. Throughput varies across runs, thermal state, prompt shape, model,
and software version, so inspect both the median and CV and repeat the baseline
when drift matters.
