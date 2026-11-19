# Core porting method

## Phase A: freeze behavior

1. Pin source code and artifacts.
2. Set inference/eval mode and seeds.
3. Disable stochastic sampling for the parity path.
4. Save post-preprocessing inputs.
5. Capture intermediate outputs at semantic boundaries.
6. Record source dtypes, accumulation behavior, epsilon values, masking conventions, and positional indexing.

## Phase B: model the graph explicitly

Write down:

- input and output contracts;
- module tree and repeated block parameters;
- tensor layout at every modality boundary;
- persistent state, recurrent state, and KV/cache state;
- static versus dynamic dimensions;
- control flow driven by data or config;
- tied/shared parameters;
- preprocessing and postprocessing outside the neural graph.

Keep Python control flow outside compiled regions until semantics are proven.

## Phase C: implement primitives and blocks

Port in this order:

1. normalization and activation;
2. positional encoding;
3. linear/convolution/FFT primitives;
4. attention, SSM, recurrent, or quantizer block;
5. one complete repeated block;
6. cache/state update;
7. frontend/encoder;
8. decoder/head;
9. complete model;
10. generation/sampling/streaming loop.

For every primitive, make layout conversions visible. Do not hide transposes inside unrelated weight-loading code.

## MLX-specific correctness habits

- MLX is lazy: materialize at deliberate boundaries with `mx.eval` rather than accidentally through printing, host conversion, or scalar extraction.
- Treat compiled state explicitly, including model state, optimizer state, random state, and mutable caches.
- Keep dynamic Python strings, token decoding, file I/O, and callbacks outside compiled functions.
- Confirm whether an operation expects NHWC or another layout; source PyTorch code often assumes NCHW.
- Use explicit dtypes for numerically sensitive operations such as normalization, softmax, recurrences, FFTs, and quantizer distances.
- Avoid repeated host/device conversions. Unified memory does not make Python/NumPy synchronization free.
- Verify slice/update semantics and duplicate-index accumulation.

### Precision and interop preflight

Use [runtime compatibility](runtime-compatibility.md) before adopting a newer
native primitive. A float32 output dtype alone does not prove full-precision
matrix arithmetic: MLX may use reduced-precision matrix hardware. For a strict
float32 baseline, launch the process with `MLX_ENABLE_TF32=0` and record the
setting. Metal 3x3 stride-one convolutions may select Winograd based on total
input size, including batch size; diagnose batch-dependent drift with
`MLX_CONV_WINOGRAD=0`. Compare each optimized path against the strict baseline
and restore the chosen settings only after parity/quality gates pass. See the
[pinned numerical-precision guide](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/docs/src/usage/precision.rst).

When comparing source and target through DLPack, synchronize the producer and
verify copy/ownership behavior; shared memory does not imply completed work.
Do not use a NumPy or foreign-framework operation inside the MLX gradient
oracle. Preserve BF16 capture values via a float32 export rather than an
unsupported direct NumPy conversion. Follow the
[interop contract](runtime-compatibility.md#framework-interoperability).

## Weight conversion principles

A conversion is data transformation, not model execution.

Each mapping entry should record:

- source tensor key and revision;
- target tensor key;
- source and target shape;
- transform sequence: rename, transpose/permute, reshape, split, concat, squeeze, cast;
- fused or unfused QKV/gate rules;
- quantization/dequantization state;
- tie/share owner;
- expected checksum or summary statistics.

Model-specific `sanitize()` functions are useful, but the transformations must remain testable and documented.

Use `scripts/convert_checkpoint.py` as the executable conversion gate for
safetensors checkpoints. Its schema-version 2 map keeps the one-to-one
`source`/`target` entries and adds:

- declared `source_shape` and `target_shape` for one-to-one entries;
- `targets: [{target, shape, dtype_policy?}, ...]` plus one explicit `split`
  transform with `axis` and `sizes`;
- `sources: [{source, shape}, ...]` plus one explicit `merge` transform with
  `axis` and a declared `target_shape`;
- global or per-entry `dtype_policy` (`keep`, `f16`, `bf16`, or `f32`);
- `ignore: [{source, reason}, ...]` and `unresolved: [...]` instead of silent
  coverage exceptions.

The converter also accepts the existing unary transform vocabulary (`rename`,
`transpose`/`permute`, `reshape`, `squeeze`, `unsqueeze`, `slice`, and `cast`).
It validates safetensors with a bounded pure-Python header reader even when the
optional `safetensors` package is installed. It uses that package lazily for
payload reads when possible, otherwise the validated NumPy reader. Output is
`model.safetensors`: `mlx.core.save_safetensors` is used when MLX is importable;
the dependency-free fallback is a deterministic pure-Python safetensors writer,
not NPZ. `target-manifest.json` and `conversion-report.json` record shapes,
dtypes, coverage counts, applied transforms, policies, writers, and SHA-256
digests. Draft maps emitted from source and scaffold manifests are intentionally
non-runnable until `draft` is false and `unresolved` is empty.

The dependency-free reader allowlist is `BOOL`, signed and unsigned 8/16/32/64
bit integers, and `F16`/`BF16`/`F32`/`F64`. Float8 and complex tensors are
rejected because NumPy-only CI cannot round-trip them into the declared MLX
checkpoint contract. BF16 payloads are decoded losslessly through float32 and
rounded back to BF16 by the fallback writer.

`validate_weight_map.py` accepts normal inspection reports with safetensors
headers and static source-format reports that expose tensor shapes, currently
ONNX initializers and GGUF tensor tables. That validation proves deterministic
shape coverage only; it does not prove operator lowering, source-framework
behavior, or MLX numeric parity.

## Completion criteria for the minimal port

- all required weights accounted for;
- no unexplained source or target keys;
- primitive/block tests pass;
- end-to-end deterministic output passes;
- cache/state incremental path matches full recomputation;
- model can save and reload without changing output;
- no quantization, compile, or custom-kernel dependency is required for correctness.
