# MLX runtime, compilation, and custom kernels

## Optimization order

### 1. Evaluation boundaries

Find accidental materialization caused by `.item()`, printing, NumPy conversion, Python conditionals on arrays, frequent synchronization, or tiny per-step `mx.eval` calls. Build larger useful graphs while preventing unbounded graph growth.

### 2. Dtype and layout

Choose dtype by numerical role rather than applying one global cast. Check that contiguous/layout transformations do not dominate. Move recurring transposes to boundaries or convert weights once.

### 3. Native fast operations

Prefer MLX-provided fused primitives where semantics match, including fast attention, normalization, RoPE, quantized matmul, and available segmented/gather/block-masked operations. Confirm masks, head grouping, sinks, and causal semantics.

After making a large matmul cheaper, profile the operations around it again.
Normalization, gated activation, casts, reshapes, and intermediate
materialization can become the new bandwidth or launch bottleneck. Prefer a
native fast operation or compilation of a stable region before custom Metal.

Do not call a separate post-op Metal dispatch a fused quantized-matmul
epilogue. That claim requires one implementation that keeps the accumulator or
intermediate on chip across the combined operation, plus evidence that the
built-in MLX path did not already fuse it. If a one-time weight permutation or
packing change makes paired gate/up values contiguous, record a reversible
conversion map and test load/save and unfused-reference parity.

The CUTLASS visitor, tile-revisit, Tensor Core, register, and TMEM mechanisms in
`fal-ideogram-v4-serving-2026-07-09` are NVIDIA-specific. The portable learning
is to measure memory traffic around a newly cheap matmul; any Metal kernel must
be redesigned and validated for MLX.

### 4. Stable-region compilation

Compile pure, shape-stable, frequently repeated numerical regions. Return updated cache/state explicitly or capture it with `outputs=`. Keep I/O, logging, tokenizer work, dynamic strings, and irregular Python control flow outside. Token decode steps may be compiled only when cache containers, shapes, static arguments, dtype/device, and closure constants are stable enough to avoid retracing.

Watch for recompilation caused by:

- changing static arguments;
- Python values captured in closures;
- shape/rank changes;
- variable container structure;
- dtype/device changes;
- hidden mutation or random state.

Use shapeless compilation only when operation semantics permit and benchmark both compile overhead and steady execution. If the target relies on very recent MLX fixes for shapeless compile, gather/reduce, or custom-kernel behavior, record the exact MLX commit or release. A fix merged on main is not publishable support for users pinned to the latest PyPI package until that package contains it.

For compiled decode or streaming steps, make stale-state failure testable: run at least two sequence lengths or chunk sizes, verify cache/state mutation across consecutive calls, and include a closure-constant change or explicit state-refresh test when the source path mutates generation settings.

### 5. Streams and overlap

Use separate streams only for genuinely independent work with enough granularity. Validate dependencies explicitly. Candidate use cases include overlapping CPU preprocessing with GPU work, independent encoder branches, or asynchronous materialization. Avoid stream complexity for tiny operations.

Treat asynchronous evaluation as an explicit scheduling choice. Put `mx.synchronize` only at measurement or dependency boundaries; otherwise hidden synchronization can erase the expected overlap or make timings misleading.

### 6. Memory behavior

Measure active, cache, and peak memory. Reuse caches and scratch buffers when safe, donate/discard intermediates where supported, and avoid retaining lazy graphs through Python references. Unified memory can still page or pressure the system.

## Custom Metal kernel gate

A custom kernel is justified only if profiling shows a stable hotspot that cannot be removed by graph, layout, cache, or native-op changes.

Required deliverables:

- readable MLX reference implementation;
- operation contract, supported shapes/dtypes, and error behavior;
- forward numerical tests and gradient tests if trainable;
- boundary and nonmultiple-size tests;
- benchmark across relevant Apple chip families where possible;
- compile-cache behavior;
- fallback path;
- end-to-end benchmark proving material value.

Common candidates in model ports include unusual fused projections, codec quantizer distance/search, recurrent scan, specialized convolution/upsampling, PyTorch-style spatial or volume sampling, sparse expert dispatch, and streaming overlap-add. None is automatically worthwhile.

For MoE, `gather_mm`/`gather_qmm` are native indexed matmul paths, while `segmented_mm` and block-masked MoE layouts need an oracle and shape legality check before being advertised as supported.

For `grid_sample`-style ports, write down the exact coordinate order, `align_corners`, padding mode, interpolation mode, dtype, and layout contract. Treat a custom kernel as valid only for the modes it tests; do not silently extend a bilinear/trilinear forward kernel to reflection padding, nearest sampling, or training gradients.

For graph, point-cloud, and scientific ports, treat scatter/segment reductions,
neighbor search, radius graphs, ragged batching, spherical harmonics, tensor
products, and energy/force heads as separate validation surfaces. Dense adjacency
rewrites can be useful for tiny correctness fixtures, but they are not a
general support claim. Equivariant layers need explicit rotation, reflection,
translation, and permutation probes before benchmark metrics matter.

## Anti-patterns

- writing a Metal kernel before establishing parity;
- fusing operations that alter accumulation order beyond quality tolerance;
- compiling a function that retraces every token/chunk;
- using host scalar extraction inside the decode loop;
- copying a CUDA tile/block design without considering Metal execution and MLX dispatch;
- measuring only the second invocation while hiding compilation cost for an interactive workload.
