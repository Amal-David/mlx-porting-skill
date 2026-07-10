# Worked port: Qwen2.5-0.5B-Instruct

This is a real, offline, end-to-end port of
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775` to the generated standalone
MLX dense-decoder package. The checked-in example contains decisions,
manifests, reports, and receipt specifications only. Model weights and NPZ
tensor captures remain outside the repository.

## Result

- Static intake: 290 BF16 tensors, 494,032,768 represented parameters, no
  remote code, dense-decoder route, artifact identity
  `sha256:2607205aab4b7ec45982916b028479d9926b5d7027c8a9aafc5acd2d2ef15517`.
- Conversion: 290 source tensors to 290 targets; 290 explicit renames and 290
  F32 casts; no ignored, unresolved, generated, transposed, split, or merged
  tensors.
- Parity: all 29 ordered rungs passed, including exact input IDs and exact
  eight-token greedy output.
- Independent check: Torch, the standalone scaffold cache path, and local
  MLX-LM 0.31.1 produced the same eight IDs and continuation.
- Benchmark: F32 and BF16 schema-2 receipts are both honest
  `performance_observation` records with `execution_attested=false`.

## Local setup and provenance

The original Hugging Face cache snapshot used symlinks into its blob store, so
the seven cached model/tokenizer files were materialized with `cp -L`. The
cached snapshot did not contain its model card or license file. A local
artifact-bound `README.md` containing `license: apache-2.0` was copied from the
repository's existing intake fixture so static inspection could record the
operator-supplied license declaration. This sidecar is part of the inspection
artifact identity; it is not presented as a downloaded upstream model card.

```bash
export WORK="$HOME/.cache/mlx-porting-work/qwen2.5-0.5b-instruct"
export SNAPSHOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
export MODEL="$WORK/source"
export RUN="$WORK/run"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "$MODEL" "$RUN"
cp -L "$SNAPSHOT"/* "$MODEL"/
cp tests/fixtures/models/decoder/README.md "$MODEL/README.md"
chmod -R a-w "$MODEL"
```

All model and tokenizer loads used the local path and
`trust_remote_code=False`; no Hub ID was passed to an execution loader.

## Inspection, advice, and plan

```bash
python3 mlx-model-porting/scripts/inspect_model.py "$MODEL" \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output "$RUN/inspection.json"

python3 mlx-model-porting/scripts/recommend_optimizations.py \
  "$RUN/inspection.json" \
  --output "$RUN/recommendations.json" \
  --markdown "$RUN/RECOMMENDATIONS.md"

python3 mlx-model-porting/scripts/make_port_plan.py \
  "$RUN/inspection.json" \
  --artifact-root "$MODEL" \
  --recommendations "$RUN/recommendations.json" \
  --output "$RUN/PORT_PLAN.md"
```

The checked-in [`inspection.json`](inspection.json) is the portable static
report. [`PORT_PLAN.md`](PORT_PLAN.md) began as the generator output and records
the completed target and validation matrix.

## Fixed Torch oracle

The prompt `Explain why the sky is blue in one sentence.` tokenizes to the
following fixed IDs. Token-ID mode removes tokenizer behavior from the parity
runner while keeping the original text documented.

```text
[840, 20772, 3170, 279, 12884, 374, 6303, 304, 825, 11652, 13]
```

```bash
python3 mlx-model-porting/scripts/capture_oracle.py "$MODEL" \
  --token-ids 840 20772 3170 279 12884 374 6303 304 825 11652 13 \
  --generate-steps 8 \
  --output "$RUN/source-oracle.npz" \
  --manifest "$RUN/source-oracle.manifest.json"
```

The default capture storage policy is F32. The source model itself executes
with its pinned BF16 weights, which matters for the later tolerance decision.
Only [`source-oracle.manifest.json`](source-oracle.manifest.json) is checked in;
the NPZ is deliberately excluded.

## Scaffold defects exposed by the real model

The first scaffold attempt failed on `sliding_window=32768` and the previously
unclassified `max_window_layers` key even though the config explicitly has
`use_sliding_window=false`. The generator now treats that exact combination as
full attention and continues to fail closed when sliding-window use is true or
ambiguous.

The inspection also showed 72 Q/K/V bias tensors: every Qwen2 layer has Q, K,
and V biases but no O-projection bias. The old generator had one all-or-none
attention-bias switch. The scaffold now derives Q/K/V/O bias presence
independently from complete inspected layer coverage and rejects partial layer
coverage.

```bash
python3 mlx-model-porting/scripts/scaffold_port.py \
  "$RUN/inspection.json" \
  --artifact-root "$MODEL" \
  --output "$RUN/mlx_port"
```

The generated target manifest contains 290 tensors: 72 attention biases, 216
per-layer weights/norms, the token embedding, and the final norm.

## Weight map decisions and conversion

The source and target use the same MLX/Hugging Face linear layout and identical
parameter names, so the draft resolved all 290 entries automatically.

```bash
python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source "$RUN/inspection.json" \
  --scaffold-manifest "$RUN/mlx_port/scaffold-manifest.json" \
  --emit-draft-map "$RUN/WEIGHT_MAP.draft.json"
```

The authored parity map changed `draft` to `false` and the global dtype policy
to `f32`. Every entry retains an explicit `rename` transform and declared
source/target shape.

- Q/K/V biases map by the same name; no O-projection bias is invented.
- `tie_word_embeddings=true` means `model.embed_tokens.weight` is the shared
  owner. Neither source nor target has a separate `lm_head.weight`.
- RoPE is config-only; no nonexistent rotary tensor is generated.
- `ignore=[]` and `unresolved=[]`; there are no silent coverage exceptions.

```bash
python3 mlx-model-porting/scripts/validate_weight_map.py \
  --source "$RUN/inspection.json" \
  --target "$RUN/mlx_port/scaffold-manifest.json" \
  --mapping "$RUN/WEIGHT_MAP.json" \
  --output "$RUN/weight-map-validation.json"

python3 mlx-model-porting/scripts/convert_checkpoint.py \
  --source "$MODEL" \
  --mapping "$RUN/WEIGHT_MAP.json" \
  --output "$RUN/converted-f32"
```

The F32 artifact is 1,976,163,149 bytes with SHA-256
`7df68182b704fb0933625cdb44cc4e89e8f4cfdb5cb75b3e2251f608807f65f0`.
The checked-in [`WEIGHT_MAP.json`](WEIGHT_MAP.json) contains the complete map;
no converted weights are checked in.

## Parity ladder and tolerance decision

The default `atol=1e-5, rtol=1e-4` run passed exact inputs and embeddings, then
stopped at `layer.0.hidden` with max absolute error 0.0548452 and cosine
0.9998527. This was not treated as a pass.

The source performs BF16 arithmetic while the converted parity artifact stores
the same BF16-quantized values as F32 and executes F32 arithmetic. The model has
residual outliers around 1,600, where BF16 spacing is large; the largest
cross-runtime absolute drift was 17.052856 at layer 21, while the worst cosine
was 0.9969896. The final symmetric policy was therefore `atol=18`, `rtol=0`,
and cosine >= 0.996. The same bounds govern both pass and fail paths; integer
rungs remain exact. No default tool tolerance was weakened globally.

```bash
python3 mlx-model-porting/scripts/run_parity.py \
  --source-model "$MODEL" \
  --package "$RUN/mlx_port" \
  --weights "$RUN/converted-f32" \
  --token-ids 840 20772 3170 279 12884 374 6303 304 825 11652 13 \
  --generate-steps 8 \
  --atol 18 --rtol 0 --cosine-min 0.996 \
  --timeout-seconds 600 \
  --output "$RUN/parity-report.json"
```

All 29 rungs passed. The exact generated IDs were:

```text
[576, 12884, 374, 6303, 1576, 432, 25963, 39020]
```

The continuation is ` The sky is blue because it reflects sunlight`.
[`parity-report.json`](parity-report.json) contains every per-rung metric, and
[`target-f32.manifest.json`](target-f32.manifest.json) records the target
capture without its NPZ tensors.

The generated cache path was also executed directly:

```bash
python3 "$RUN/mlx_port/generate.py" \
  --weights "$RUN/converted-f32/model.safetensors" \
  --token-ids 840 20772 3170 279 12884 374 6303 304 825 11652 13 \
  --max-new-tokens 8
```

It returned the same eight IDs as the full-recompute Torch oracle.

## Local MLX-LM cross-check

MLX-LM 0.31.1 loaded `$WORK/source` completely offline with
`local_files_only=True` and `trust_remote_code=False`. Its Qwen2 implementation
returned the same eight greedy IDs and decoded text. The exact transcript is in
[`mlx_lm-cross-check.txt`](mlx_lm-cross-check.txt).

## Benchmark receipts

A second conversion changed only the global map policy from F32 to BF16. The
benchmark workload measures separate-process load plus six cached greedy tokens;
six tokens were selected because F32 and BF16 preserve that exact fixed output.
The seventh BF16 token diverges, so no broader BF16 quality equivalence is
claimed.

The receipt harness used isolated `python3.13 -I -B`, whose system site contains
MLX on this machine. Isolated Python 3.14 correctly excluded the user-site MLX
installation and failed before measurement; no hidden environment override was
used to bypass that isolation.

```bash
python3 mlx-model-porting/scripts/benchmark_command.py \
  --receipt-spec "$WORK/run/benchmark-f32.spec.json" \
  --quality-contract "$WORK/run/benchmark-f32.quality.json" \
  --warmup 1 --runs 5 --timeout 120 \
  --output mlx-model-porting/assets/benchmarks/qwen2.5-0.5b-port-f32.json

python3 mlx-model-porting/scripts/benchmark_command.py \
  --receipt-spec "$WORK/run/benchmark-bf16.spec.json" \
  --quality-contract "$WORK/run/benchmark-bf16.quality.json" \
  --baseline-receipt qwen2.5-0.5b-port-f32.json \
  --warmup 1 --runs 5 --timeout 120 \
  --output mlx-model-porting/assets/benchmarks/qwen2.5-0.5b-port-bf16.json
```

Receipt outputs are immutable and these exact commands now refuse to overwrite
them. Use new labels for a repetition. See
[`benchmark-receipts.md`](benchmark-receipts.md) for the measured values,
digests, classifications, and checked-in evidence pointers.

Derived artifacts were regenerated in this order:

```bash
python3 mlx-model-porting/scripts/validate_benchmarks.py generate
python3 mlx-model-porting/scripts/generate_claim_catalog.py
python3 mlx-model-porting/scripts/generate_evidence_index.py
python3 mlx-model-porting/scripts/generate_site_data.py
```

## Checked-in files and exclusions

This directory contains the portable inspection, annotated plan, complete
weight map, parity report, source/target capture manifests, receipt specs and
quality contracts, receipt pointers, and MLX-LM transcript. It contains no
weights, NPZ/NPY tensors, tokenizer payloads, absolute private paths, or files
larger than 1 MB.
