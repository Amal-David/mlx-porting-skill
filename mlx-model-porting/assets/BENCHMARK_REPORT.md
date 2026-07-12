# Benchmark evidence assessment

This report is generated deterministically from `assets/benchmarks/receipt_assessments.json`.
Historical schema-1 measurements are retained as observations; only schema-2 candidate receipts that pass every gate are promotion-ready.
MLX-LM 0.31.1 speculative-decoding observations are held because v0.31.2 fixed silent output corruption.

## Summary

- Receipts: 13
- Performance observations: 12
- Promotion-ready: 0
- Rejected: 1
- Integrity errors: 0

## Assessments

| Receipt | Classification | Enabled methods | CV | Recomputed primary ratio | Reasons |
|---|---|---|---:|---:|---|
| `kv-4bit-8k.json` | `performance_observation` | uniform-kv-quantization | 0.081 | 1.1144x | legacy-schema-1<br>missing-baseline-digest<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash |
| `kv-baseline-8k.json` | `performance_observation` | none | 0.121 | n/a | legacy-schema-1<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>unstable-primary-metric |
| `pcache-cold.json` | `performance_observation` | none | 0.072 | n/a | legacy-schema-1<br>missing-checked-in-input<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>nonportable-ephemeral-path |
| `pcache-warm.json` | `performance_observation` | prompt-prefix-cache | 0.006 | 0.9178x | baseline-workload-incompatible<br>legacy-schema-1<br>missing-baseline-digest<br>missing-checked-in-input<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>nonportable-ephemeral-path |
| `quant-4bit.json` | `performance_observation` | native-low-bit-weight-quantization | 0.109 | 2.3966x | incompatible-quant-baseline<br>legacy-schema-1<br>missing-baseline-digest<br>missing-checked-in-input<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>nonportable-ephemeral-path<br>unstable-primary-metric |
| `quant-baseline-bf16.json` | `performance_observation` | none | 0.012 | n/a | legacy-schema-1<br>missing-checked-in-input<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>nonportable-ephemeral-path |
| `qwen2.5-0.5b-port-bf16.json` | `performance_observation` | bf16-weight-cast | 0.009 | 1.8122x | missing-external-attestation-signature |
| `qwen2.5-0.5b-port-f32.json` | `performance_observation` | none | 0.006 | n/a | baseline-role-not-promotable<br>missing-external-attestation-signature |
| `spec-baseline.json` | `performance_observation` | none | 0.005 | n/a | legacy-schema-1<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash |
| `spec-draft-k2.json` | `performance_observation` | draft-model-speculation | 0.021 | 1.2540x | legacy-schema-1<br>missing-baseline-digest<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>mlx-lm-0.31.1-speculative-correctness-fix |
| `spec-draft-k3.json` | `performance_observation` | draft-model-speculation | 0.443 | 0.6728x | legacy-schema-1<br>missing-baseline-digest<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>mlx-lm-0.31.1-speculative-correctness-fix<br>unstable-primary-metric |
| `spec-draft-k4.json` | `performance_observation` | draft-model-speculation | 0.122 | 0.5498x | legacy-schema-1<br>missing-baseline-digest<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>mlx-lm-0.31.1-speculative-correctness-fix<br>unstable-primary-metric |
| `stack-measured-together.json` | `rejected` | uniform-kv-quantization, prompt-prefix-cache, draft-model-speculation | 0.169 | 0.3254x | baseline-workload-incompatible<br>legacy-schema-1<br>missing-baseline-digest<br>missing-checked-in-input<br>missing-model-lineage<br>missing-output-digests<br>missing-quality-artifact<br>missing-rollback-condition<br>missing-target-hash<br>missing-workload-hash<br>mlx-lm-0.31.1-speculative-correctness-fix<br>nonportable-ephemeral-path<br>partial-invalid-stack-configuration<br>unstable-primary-metric |

## Promotion rule

A candidate is `promotion_ready` only when aggregate recomputation, pinned target/source lineage, the canonical experiment invariant, normalized target/workload hashes, bounded raw evidence, controlled quality, stability, rollback, baseline compatibility, and externally signed execution attestation all pass. The retained Qwen challenge/evidence lane establishes internal consistency and reproducibility-on-request, but SHA-256 digests are not signatures. Promotion requires a protected Apple-Silicon signer and an out-of-repository trust anchor covering the repository commit/tree, challenge, reviewed dependency manifest, raw output, promotion policy, and timing. No such signer exists today, so every checked-in receipt remains sealed. The primary-metric ratio must also exceed `1 + max(2%, 2 x max(candidate CV, baseline CV))`. Missing evidence is never inferred.
