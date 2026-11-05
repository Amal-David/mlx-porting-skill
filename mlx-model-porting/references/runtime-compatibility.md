# Runtime compatibility and current MLX capabilities

Reviewed **2026-11-05** against MLX **0.32.4**, released 2026-10-14 at
[`36969c094354086e15d2a327a9c50feb20b9cf11`](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11).
This is source/API verification, not execution on the target Mac. Treat 0.32.4
as a conservative source-reviewed floor for the new candidates below, not a
claim about the first version that implemented each feature.

## Contents

- [Resolve the actual runtime](#resolve-the-actual-runtime)
- [Native operation gates](#native-operation-gates)
- [Quantization format contract](#quantization-format-contract)
- [Metal and M5 boundaries](#metal-and-m5-boundaries)
- [Framework interoperability](#framework-interoperability)
- [Distributed execution](#distributed-execution)
- [Sources and review scope](#sources-and-review-scope)

## Resolve the actual runtime

Record installed package versions, source/build revision when available, macOS,
chip, device/backend, and relevant environment settings in the port plan and
benchmark. Check `mx.is_available(mx.gpu)` and `mx.metal.is_available()` before
calling a path Metal-specific. Use `mx.device_info(mx.gpu)` for device metadata;
its keys vary by backend. A Linux CPU/CUDA test is not Metal validation.

Do not turn a new upstream symbol into a model-support claim. Require the
installed symbol, a small legal-shape execution probe, source-oracle parity,
and a fallback. Keep old runtime support when its existing path still works.
See [core porting](porting-core.md) and [benchmarking](benchmarking.md).

Provide declared target facts to `recommend_optimizations.py --target-profile`
using this shape (replace the example with the actual target):

```json
{
  "schema_version": 1,
  "hardware": {"chip": "Apple M5 Pro"},
  "software": {"mlx": "0.32.4", "macos": "26.2"},
  "capabilities": [],
  "workloads": ["long-context"]
}
```

Version fields use numeric release strings with one to three components.
Unknown, prerelease, development, or locally suffixed versions require review;
do not strip their suffixes to clear a hold. Chip matching accepts whole Apple
M-family labels with optional Pro/Max/Ultra suffixes, but a method must list
each eligible family explicitly. The M5 recipe does not silently authorize M6.
Profile facts are declarations, not device measurements. Runtime holds are
separate from numeric claim holds and survive `--allow-blocked`.

## Native operation gates

| Candidate | Source-reviewed gate | Contract and validation |
| --- | --- | --- |
| `mx.fast.gated_delta_update` | MLX >=0.32.4 and callable present | Match the recurrence, normalization, scaling, state layout and mask semantics; compare both output and final state. |
| `mx.fft.stft` / `mx.fft.istft` | MLX >=0.32.4 and both callables present | Match window values, centering, padding, hop, normalization, complex layout and reconstruction length. |
| Native NAX matmul dispatch | MLX >=0.32.4, M5 target, macOS >=26.2, Metal/build availability | Profile the actual shape/dtype; chip/version eligibility does not establish that a particular call uses NAX. |

The gated-delta implementation accepts `q,k` shaped `[B,T,Hk,Dk]` and `v`
shaped `[B,T,Hv,Dv]`. Its state is **`[B,Hv,Dv,Dk]` in float32**. Follow the
implementation and upstream oracle tests: the binding's state-shape docstring
does not match them. Pass decay in linear space; a log-decay source needs an
explicit conversion. Preserve the model's query scaling outside the primitive.
Test full sequence versus consecutive chunks, nonzero initial state, masks,
grouped heads, and training gradients separately. An API fallback can be
correct without being the fused path. Do not substitute this recurrence for a
Mamba/selective-SSM block or declare every hybrid checkpoint supported.

Native STFT returns complex `(..., n_freq, n_frames)` arrays. Its default window
is rectangular, not a model-specific Hann window. Pass `n_fft` explicitly for
odd-size inverse transforms and `length` when reconstructing a fixed waveform.
Keep the existing frontend oracle until waveform and downstream feature parity
pass; the new primitive does not establish a complete ASR/TTS frontend.

For attention, recheck non-square cached decoding: `mask="causal"` uses
lower-right alignment in this snapshot. Keep grouped K/V heads unexpanded.
`force_fused=True` is a diagnostic option that raises if no fused path exists;
it may trade speed for memory and is not a universal optimization. Compare the
explicit mask reference on unequal query/key lengths before enabling it.

## Quantization format contract

Use the exact recipe supported by the pinned runtime, not the label “4-bit”.

| `mode` | Legal `group_size` | Legal `bits` | Per-group scale | Quantization bias |
| --- | --- | --- | --- | --- |
| `affine` | 32, **64 default**, 128 | 1, 2, 3, **4 default**, 5, 6, 8 | Same dtype as input | Yes |
| `mxfp4` | 32 | 4 | E8M0 | No |
| `mxfp8` | 32 | 8 | E8M0 | No |
| `nvfp4` | 16 | 4 | E4M3 | No |

`mx.quantize` requires rank >=2 and a last dimension divisible by the group
size. It returns three arrays for affine and two for floating formats. Preserve
mode, grouping, packed weights, scales and any NVFP4 global scale through
conversion/reload. Do not unpack every mode into `(weight, scale, bias)`.

`nn.quantize(..., quantize_input=True)` applies only to supported Linear paths
with `nvfp4` or `mxfp8`; select Linear modules explicitly. `mx.qqmm` quantizes
its input on the fly and is distinct from weight-only `mx.quantized_matmul`.
Measure that overhead and quality separately. One-bit affine availability does
not demonstrate a useful one-bit checkpoint or a trained ternary-model port.
Use [quantization](quantization.md) for exclusions and quality gates.

## Metal and M5 boundaries

M5 GPU Neural Accelerators are GPU matrix hardware, separate from the Neural
Engine. MLX's NAX path also depends on OS and build availability; builds may
disable it with `MLX_METAL_NO_NAX`. Prefer native operations before writing a
TensorOps kernel. Keep an M1-M4-compatible path and measure prefill and decode
independently; Apple’s M5 measurements do not transfer to another chip/workload.

WWDC26 describes quantized TensorOps, scale planes, and cooperative-tensor
matmul inputs. Some of those interfaces require macOS/iOS 27. Treat them as
Metal SDK capabilities, not evidence that every MLX Python custom kernel can
use them. Check SDK/OS availability, layout compatibility and alignment before
a custom implementation; retain its earlier-OS fallback.

Keep the default safe custom-kernel math mode for masks and special values.
Only try `compile_options={"math_mode": "relaxed"}` or `"fast"` after testing
NaN/Inf and masked rows. See [compile and kernels](compile-and-kernels.md).
For a strict float32 oracle, inspect reduced-precision matrix dispatch and
Winograd behavior as described in [core porting](porting-core.md).

## Framework interoperability

In this snapshot, `mx.asarray` / `mx.from_dlpack` can reuse a non-private Metal
buffer; `copy=False` requires sharing and fails when a copy is needed.
`mx.array` copies. The documented PyTorch >=2.12 ordinary MPS path uses shared
storage; verify the actual buffer/runtime rather than promising universal
zero-copy conversion. A NumPy bridge is a separate, potentially copying path.

DLPack does not synchronize pending Metal work. Complete work in the producer
before consuming the imported array:

```python
# torch_tensor is an already-created, trusted MPS tensor.
torch.mps.synchronize()
mlx_array = mx.asarray(torch_tensor)

mx.eval(mlx_array)
torch_view = torch.as_tensor(mlx_array)
```

Keep ownership/lifetimes valid and forbid external mutation during MLX use.
Do not place host conversions in an autodiff path: external work is not tracked
by MLX. Cast BF16 captures to float32 before NumPy export. Verify shape, dtype,
strides and values for each boundary; interop sharing is not parity evidence.

## Distributed execution

Use distributed execution only for a measured capacity/throughput need. Keep
rank count, topology, per-rank versions, shard map and collective time in the
benchmark. Distinguish depth/pipeline splits from width/tensor splits.

JACCL requires Thunderbolt 5 RDMA, macOS >=26.2 and a fully connected mesh in
the reviewed guide. Recovery-mode RDMA enablement and network reconfiguration
are operator setup, not automatic porting steps. TCP ring, MPI and CUDA NCCL
have different contracts; do not infer JACCL from a generic distributed import.

Initialize the requested backend strictly and verify the expected group size:
`mx.distributed.init(strict=True, backend="jaccl")`. With the default `any`
and non-strict behavior, failed setup can become a singleton group; that must
not be reported as successful distributed execution.

Leave `MLX_METAL_FAST_SYNCH` unset for the baseline. Apple's October 2026 blog
uses it, but the pinned MLX guide warns of deadlocks/GPU hangs and recommends
leaving it off. A blog invocation does not override the runtime warning.
Roll back on parity failure, unexpected singleton execution, hangs, or a gain
erased by communication. Never infer speedup from fitting more parameters.

## Sources and review scope

All MLX links below pin the same 0.32.4 commit; registry records beginning
`mlx-0324-` record the affected guide/method and source-only claim boundary.

- [Device and NAX gates](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/device.cpp), [fast API](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/src/fast.cpp), [gated-delta implementation](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/fast.cpp) and [oracle tests](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/tests/test_fast_gated_delta.py).
- [Spectral API](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/src/fft.cpp), [quantization API](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/src/ops.cpp) and [quantized layers](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/mlx/nn/layers/quantized.py).
- [Interop guide](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/docs/src/usage/numpy.rst), [distributed guide](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/docs/src/usage/distributed.rst) and [strict initialization](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/python/src/distributed.cpp).
- [Apple M5 article, 2025-11-19](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), [WWDC26 Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/), and [Apple distributed inference, 2026-10-15](https://machinelearning.apple.com/research/mlx-distributed-inference): hardware/approach context only; no local performance claim.
