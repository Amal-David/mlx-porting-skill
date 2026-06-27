# Practitioner Blog Miner Research Blog

## Assignment
Mine implementation writeups and practitioner reports for leads while keeping unsupported claims out of supported guidance.

## Sources sampled
- ltx-2-mlx BlockStreamer implementation (https://github.com/dgrauet/ltx-2-mlx/blob/3f5897a9582b/packages/ltx-core-mlx/src/ltx_core_mlx/loader/block_streaming.py, accessed 2026-06-27)
- ltx-2-mlx block streaming tests (https://github.com/dgrauet/ltx-2-mlx/blob/3f5897a9582b/tests/test_block_streaming.py, accessed 2026-06-27)
- mlx-grid-sample custom Metal implementation (https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84/mlx_grid_sample.py, accessed 2026-06-27)
- mlx-grid-sample correctness tests (https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84/test_grid_sample.py, accessed 2026-06-27)
- mlx-grid-sample benchmark harness (https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84/bench_grid_sample.py, accessed 2026-06-27)

## Candidate findings
- top1000-promoted-block-streaming: Repeated-block weight streaming has pinned implementation and rollback gates [adopted] - dgrauet/ltx-2-mlx provided inspectable safetensors block streaming code, tests, and a regression report, so the skill now treats repeated-block weight streaming as a proven MLX-port technique for memory-bound diffusion or flow-style models.
- top1000-promoted-grid-sample: PyTorch-compatible grid_sample custom kernels have enough evidence for scoped guidance [adopted] - katlun-lgtm/mlx-grid-sample provided custom Metal code, correctness tests, and a benchmark harness for 2D/3D grid_sample. The skill now scopes this as a proven MLX-port technique for spatial or volume warping after parity and end-to-end benchmark gates.

## Decision notes
- Two contributor-derived techniques already crossed the promotion bar because they had pinned source, tests, and rollback gates.

## Open validation
- top1000-promoted-block-streaming: Require local parity and memory/latency measurement on the target model before recommending it in a port plan.
- top1000-promoted-grid-sample: Run model-level parity and benchmark before using the kernel in a port.
