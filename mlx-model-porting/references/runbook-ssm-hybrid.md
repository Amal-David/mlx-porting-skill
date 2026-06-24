# Runbook: state-space, recurrent, and hybrid models

## Applies to

Mamba/Mamba2, RWKV, RecurrentGemma, Jamba, Zamba, DeltaNet/linear-attention, Kimi-linear, LFM-style and other recurrent/SSM/attention hybrids.

## Architecture fingerprint

Document each layer type and sequence:

- convolutional state and kernel width;
- recurrent/SSM state shape and update equations;
- discretization, scan, gates, and normalization;
- attention layers mixed into the stack;
- chunked parallel training path versus recurrent inference path;
- state initialization/reset and dtype;
- position handling and cache/state sharing;
- fused upstream kernels whose math must be reconstructed.

## Source oracle checkpoints

For a single layer capture:

1. projected/gated inputs;
2. convolution state before/after update;
3. SSM parameters after input dependence;
4. one-step recurrent state update;
5. chunk/scan output;
6. gate/output projection;
7. attention cache for hybrid layers;
8. full-sequence versus recurrent outputs.

## Weight conversion

- map packed projections with explicit split order;
- preserve convolution orientation and padding;
- preserve A/B/C/D or equivalent parameter transforms;
- verify log/exponential parameterizations;
- preserve per-head/group state dimensions;
- distinguish training-only fused packing from canonical weights;
- map layer-type schedule exactly.

## Minimal MLX path

1. Write a readable one-step recurrence.
2. Verify it against source for several steps.
3. Write a full-sequence loop oracle.
4. Compare source parallel/chunk path to the loop oracle.
5. Assemble hybrid stack with explicit state object.
6. Add generation/reset/save/reload.
7. Only then implement vectorized scan or custom kernel.

## Parity traps

- off-by-one state update;
- convolution state rolled in wrong direction;
- continuous-to-discrete parameter transform mismatch;
- state initialized in wrong dtype/device/value;
- packed projection split order;
- chunk boundary state not carried;
- source fused kernel uses numerically stabilized formulation;
- hybrid layer schedule differs from simple periodic assumption;
- cache/state accidentally copied each token.

## Optimization ladder

1. Keep recurrent state in MLX arrays and avoid host extraction.
2. Preallocate/update state without concatenation.
3. Compile the one-step recurrent cell with explicit state.
4. Use chunked/parallel path for prefill and recurrent path for decode.
5. Evaluate native scan/vectorization primitives.
6. Write a custom Metal scan only after a correct loop and end-to-end profile.
7. Quantize large projections first; keep recurrence parameters/state high precision initially.
8. Combine with attention cache optimizations only for hybrid attention layers.

## Completion gates

- full-sequence and recurrent execution agree;
- arbitrary chunk partitions produce the same output/state;
- reset and batch state behavior pass;
- long sequences remain numerically stable;
- performance report distinguishes prefill scan from recurrent decode;
- custom scan has fallback and odd-length tests.
