# Quantization quality gate

`scripts/quant_quality_gate.py` is a local, user-facing diagnostic for an
already-supported `mlx_lm` model. It does not create a benchmark receipt,
receipt assessment, effective claim, evidence-index input, or other promotable
artifact. Its strict JSON is printed to stdout only.

## Correct scoring path

The held text is tokenized with special tokens, with BOS inserted when the
tokenizer declares one. The model is loaded through `mlx_lm.load`; the scorer
feeds `tokens[:-1]` through the same cached model call used by
`mlx_lm.generate_step`, converts logits to float32, applies log-softmax, and
gathers the probabilities of `tokens[1:]`. There is no padding in the
single-sequence workload, so all 72 next-token positions are scored.

The cache is material for Gemma 3n in the tested runtime. The fixed plain-English
text scored at perplexity 210.5175 through the generation-compatible cached
path. A no-cache probe scored about 3,151 and was rejected as the wrong path;
the tool also stops before candidate measurement if reference perplexity is not
finite and between 1 and 512.

## Local Gemma 3n E2B measurement

Measured 2026-07-13 on an Apple M4 Pro with Metal, Python 3.14.4, MLX 0.32.0,
and mlx-lm 0.31.3. The built-in workload used four greedy 32-token prompts and
the default gates: perplexity ratio at most 1.10, first-token agreement at least
0.75, and candidate-only degenerate-output rate 0.

| Artifact | Perplexity | Ratio vs bf16 | Exact outputs | First token | Candidate-only degenerate | Verdict | Run ID |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bf16 reference | 210.5175 | 1.0000 | 4/4 self-match | 4/4 self-match | 0/4 | reference | shared by all runs |
| uniform 4-bit (`gemma-3n-e2b-4bit-naive`) | 260.2654 | 1.2363 | 0/4 | 4/4 | 1/4 | fail | `369307f19bbef98bbdf7f36e904e4b0eb805d33c9d6d26c6a9e94fab8fcf69b5` |
| PLE-safe (`gemma-3n-e2b-4bit-plesafe`) | 238.5126 | 1.1330 | 0/4 | 3/4 | 1/4 | fail | `2e2eef74c7df44615f5b6c2f34384b51e8c3c6043ec5c577326db981ea7ae128` |
| PLE-forced probe (`gemma-3n-e2b-4bit-ple-forced`) | 240.0381 | 1.1402 | 0/4 | 4/4 | 0/4 | fail | `4221f0260d8e9bba87ac40295b9ce79e6764efad0b513b38889ba663c56c8e2a` |

The commands were:

```bash
python3 mlx-model-porting/scripts/quant_quality_gate.py \
  --reference ~/.cache/mlx-porting-work/gemma-3n-e2b \
  --candidate ~/.cache/mlx-porting-work/gemma-3n-e2b-4bit-naive

python3 mlx-model-porting/scripts/quant_quality_gate.py \
  --reference ~/.cache/mlx-porting-work/gemma-3n-e2b \
  --candidate ~/.cache/mlx-porting-work/gemma-3n-e2b-4bit-plesafe

python3 mlx-model-porting/scripts/quant_quality_gate.py \
  --reference ~/.cache/mlx-porting-work/gemma-3n-e2b \
  --candidate ~/.cache/mlx-porting-work/gemma-3n-e2b-4bit-ple-forced
```

## What “PLE forced” means

The ordinary group-64 “naive” conversion was already quantizing PLE: 98 of 158
PLE/AltUp quantizable modules had packed scales, including the per-layer
embedding, projections, gates, and modality routers. The 60 skipped modules
were the 4-column `prediction_coefs` and `correction_coefs` tables. MLX 0.32.0
supports native affine group sizes 32, 64, and 128, so those tables cannot be
represented as native packed affine layers.

The forced diagnostic artifact therefore used native group-64 packed 4-bit for
the 98 eligible PLE modules and explicit per-row affine 4-bit
quantize/dequantize for the 60 incompatible coefficient tables, which remain
stored as bf16. All 60 tables changed numerically; the artifact's `config.json`
records the method and module list under `quant_quality_probe`. This isolates
4-bit rounding sensitivity in the otherwise-unrepresentable tables, but it is
not evidence for a native 4-bit kernel path for those tables.

Force-quantizing the PLE coefficient values did **not** reproduce a catastrophic
or gibberish failure on this workload. The outputs were largely coherent, and
the forced artifact triggered no degeneration detector. However, every 4-bit
variant exceeded the strict 1.10 perplexity-ratio limit, so these measurements
also do not establish lossless or gate-passing quality. This is one small local
diagnostic workload, not a general Gemma quality claim.
