# Runbook: Mixture-of-Experts Transformer

## Applies to

Mixtral, DeepSeek-MoE, Qwen-MoE, DBRX, Grok, MiniMax-style sparse MoE decoders and hybrids with routed experts. Use the dense decoder runbook for shared attention and non-MoE blocks.

## Architecture fingerprint

Record:

- number of experts, active top-k, shared experts, expert groups/nodes;
- router input, dtype, bias, normalization, noise, and auxiliary-loss behavior;
- top-k selection before/after softmax;
- expert capacity or token dropping;
- expert MLP activation and fused gate/up layout;
- shared expert scaling/gating;
- expert parallel/offload assumptions in source;
- whether experts differ by layer;
- dense first/last layers or periodic MoE pattern.

## Source oracle checkpoints

For one MoE layer capture:

1. router input;
2. router logits/probabilities;
3. selected expert IDs and weights per token;
4. token-to-expert dispatch indices;
5. each selected expert output for a tiny fixture;
6. weighted combine result;
7. shared expert output and gate;
8. final residual.

Construct a fixture that routes tokens to multiple experts and creates duplicate expert assignments.

## Weight conversion

- preserve expert index ordering exactly;
- map fused expert tensors only when target packing is documented;
- verify gate/up split order for every expert;
- distinguish shared versus routed experts;
- preserve router bias and dtype;
- validate expert dimensions independently of dense MLP dimensions;
- record any expert stacking axis and target segmented/gather layout;
- do not discard inactive experts to reduce size unless producing an explicitly altered model.

## Minimal MLX path

1. Port router in high precision.
2. Implement a simple readable loop over experts.
3. Verify selected IDs/weights and combined output.
4. Assemble one block and full model.
5. Add cache/generation parity from dense decoder runbook.
6. Only then replace loops with vectorized, gather, segmented, or custom dispatch.

The loop baseline is intentionally slow but provides a trustworthy oracle for optimized dispatch.

## Parity traps

- applying softmax before versus after top-k;
- renormalizing top-k weights when the source does not;
- wrong expert stacking axis;
- duplicate-token scatter semantics;
- router calculation in reduced precision;
- missing shared expert or wrong shared-expert gate;
- source expert bias folded into a fused tensor;
- token dropping/capacity behavior ignored;
- distributed source code hiding the canonical single-device semantics.

## Optimization ladder

1. Batch tokens per expert to replace per-token calls.
2. Use `gather_mm` for indexed expert projections when loop-oracle parity passes. If indices are sorted, benchmark the sorted path separately.
3. Use `gather_qmm` only for quantized expert matrices after recording group size, bit width, mode, scales/biases layout, and quality drift.
4. Treat `segmented_mm` as a gated candidate: the C++ primitive is documented, but the exact Python binding and MoE segment mapping must be verified on the target MLX version.
5. Treat `block_masked_mm` as a primitive, not full MoE support. It needs legal block size, dense oracle parity, mask-build timing, and an end-to-end win.
6. Fuse expert gate/up projections only as an experiment when checkpoint conversion is reversible and a merged/tested runtime path exists.
7. Compile router + dispatch metadata only if dynamic token routing does not cause harmful retracing.
8. Reuse dispatch buffers and avoid host extraction of expert IDs.
9. Quantize large expert weights; keep router and sensitive shared components higher precision.
10. Evaluate expert memory mapping/offload only when the model does not fit unified memory.
11. For serving, batch requests to increase tokens per active expert while monitoring tail latency.
12. Treat expert reduction (`moe-top-k` lower than trained value) as a lossy product mode, never a default optimization.

## Quantization policy

Experts dominate memory and are prime weight-quantization candidates. Measure per-layer/router sensitivity and rare-expert behavior. Router, shared expert gates, norms, and small projections should remain high precision initially. Quantized expert dispatch must be benchmarked because small token groups can underutilize low-bit kernels.

For `gather_qmm`, scales and biases must follow the quantized expert weight batch dimensions. Training or differentiable-weight-update paths need extra review because indexed quantized gathers are not a blanket replacement for ordinary trainable matmuls.

## Completion gates

- router IDs and weights match on deterministic fixtures;
- loop oracle and optimized dispatch match;
- duplicate/scatter cases pass;
- all experts load with correct shapes;
- quality is checked on prompts that activate diverse experts;
- benchmarks report active experts/tokens per expert and batch/concurrency;
- benchmarks report sorted flag, expert matmul time, dispatch/combine time, compile count, and peak memory;
- any top-k reduction or expert pruning is labeled lossy.
