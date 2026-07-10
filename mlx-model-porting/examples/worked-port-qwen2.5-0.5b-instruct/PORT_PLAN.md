# MLX port plan

This plan began as the exact `make_port_plan.py` output and was annotated after
the worked execution completed. Detailed commands and evidence are in `README.md`.

## Source and target

- Source: `source`
- Requested/pinned revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Artifact identity: `sha256:2607205aab4b7ec45982916b028479d9926b5d7027c8a9aafc5acd2d2ef15517`
- Local inspection path: `source`
- Static tensor count: 290
- Static parameter count: 494,032,768
- License evidence: [{'source': 'README.md', 'value': 'apache-2.0'}]
- Target package: standalone eager MLX scaffold, with MLX-LM used as an independent local cross-check.
- Target Mac/workload: Apple M4 Pro, 48 GB unified memory, macOS 26.3, MLX 0.30.4; batch 1, 11-token prompt, 8-token greedy parity fixture and 6-token receipt workload.

## Architecture route

- Primary family: `dense-decoder-transformer`
- Secondary families: none
- Required runbooks: `references/runbook-decoder-transformer.md`
- Architecture traits: not declared; verify manually
- Detection evidence: exact model_type=qwen2, architecture class patterns: ForCausalLM, CausalLM, config keys: num_hidden_layers, num_attention_heads, hidden_size, vocab_size, weight signals: self_attn.q_proj.weight, mlp.down_proj.weight
- State/cache contracts:
  - `dense-decoder-transformer`: per-layer KV cache

## Static risk gates

- No high-signal static risk was found; still review code, license, and provenance.

## Source oracle

- [x] Pin source model and tokenizer at revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- [x] Freeze the documented 11-token prompt fixture.
- [x] Capture embeddings, attention/MLP branches, all 24 blocks, final norm, logits, and greedy IDs.
- [x] Capture an 8-token deterministic end-to-end output.
- [x] Match the scaffold cache-based greedy path to the full-recompute Torch oracle and MLX-LM continuation.

## Weight conversion

- [x] Export the 290-tensor source manifest without executing model code.
- [x] Create 290 explicit same-name/same-shape entries in `WEIGHT_MAP.json`.
- [x] Preserve Q/K/V-only attention biases and the tied embedding owner; no split, merge, transpose, ignore, or generated target is needed.
- [x] Explain the absence of ignored/generated tensors in `README.md`.
- [x] Validate 100% source and target shape coverage with `validate_weight_map.py`.

## Implementation phases

1. Implement config and preprocessing contract.
2. Implement a readable primitive/block oracle in eager floating point.
3. Achieve block and staged intermediate parity.
4. Assemble end-to-end model and deterministic output.
5. Add exact cache/recurrent/streaming state and compare against full recomputation.
6. Save/reload and package a smoke-test fixture.
7. Establish a baseline profile and benchmark report.
8. Run one optimization experiment at a time.
9. Quantize only after unquantized quality and performance are recorded.
10. Publish only after license/provenance and clean-environment load tests.

## Optimization advice

- Canonical source: `recommend_optimizations.py` schema-2 output

- Canonical planning stack: `dense-decoder-inference`

### Validated locally

None.

### Validated by source or theory

- `fast-sdpa` (`native-mlx`): Profile-required. Treat it as a likely attention-kernel win only when attention dominates the measured profile. Gate: Prefill tensor parity
- `native-low-bit-weight-quantization` (`native-mlx`): No numeric effect is claimed because the effective-claim catalog withholds this observation. Gate: task quality against unquantized baseline
- `compile-stable-region` (`native-mlx`): Profile-required. Compile can help repeated hot regions but cold compile cost and retracing can erase wins. Gate: compile count or cold/warm timing captured
- `lazy-eval-boundaries` (`native-mlx`): Profile-required. This removes false synchronization and graph-growth pathologies rather than guaranteeing a fixed speedup. Gate: same outputs
- `draft-model-speculation` (`official-mlx-project`): No numeric effect is claimed because the effective-claim catalog withholds this observation. Gate: lossless distribution or quality-preserving test
- `uniform-kv-quantization` (`official-mlx-project`): No numeric effect is claimed because the effective-claim catalog withholds this observation. Gate: logit/perplexity drift over context lengths

### Benchmark required

None.

### Experimental approaches

- `eagle-medusa-mtp-drafters` (`research-candidate`): Source-reported speedups are drafter- and workload-specific; MLX support must be validated per artifact. Gate: drafter compatibility Opt-in required: This is an experimental approach. Do you want to try it? Execution remains held until explicit consent.


## Validation matrix

| Stage | Fixture | Metric/tolerance | Status |
|---|---|---|---|
| Preprocessing | fixed 11-token IDs | exact | passed |
| Weight coverage | 290 source/target tensors | 100% explained | passed |
| Primitive/block | embedding + 24 block outputs | `atol=18`, `rtol=0`, cosine >= 0.996 | passed |
| End-to-end | 8-token greedy fixture | exact IDs | passed |
| Cache/state | scaffold cache decode vs full-recompute oracle | exact 8-token IDs | passed |
| Task quality | fixed continuation and MLX-LM cross-check | exact IDs/text | passed |
| Performance | F32 vs BF16, load + 6 greedy tokens | schema-2 wall-time receipts | observation only |

## Stop conditions

- Missing or incompatible license/provenance.
- Required behavior exists only in unreviewed remote code.
- First divergence cannot be localized through staged checkpoints.
- Candidate optimization fails correctness/quality or lacks end-to-end benefit.
- Custom kernel lacks a reference fallback and shape/dtype coverage.
