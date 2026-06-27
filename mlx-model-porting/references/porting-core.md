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
