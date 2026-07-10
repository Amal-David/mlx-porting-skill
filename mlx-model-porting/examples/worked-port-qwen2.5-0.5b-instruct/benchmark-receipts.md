# Benchmark receipt pointers

The checked-in receipts measure separate-process model load plus six greedy
tokens on Apple M4 Pro with 48 GB unified memory. Both variants reproduce the
same exact six-token quality artifact.

| Variant | Receipt | Median wall time | CV | Classification | State |
|---|---|---:|---:|---|---|
| F32 baseline | [`qwen2.5-0.5b-port-f32.json`](../../assets/benchmarks/qwen2.5-0.5b-port-f32.json) | 1.6550946250 s | 0.0060855 | `performance_observation` | `execution_attested=true`; baseline role is non-promotable |
| BF16 candidate | [`qwen2.5-0.5b-port-bf16.json`](../../assets/benchmarks/qwen2.5-0.5b-port-bf16.json) | 0.9133066250 s | 0.0090013 | `promotion_ready` | every promotion gate passes |

- `1.8122003933x` is the observed receipt inverse-wall-time ratio, not a portable guarantee.
- `1.02x` is the required receipt noise floor.
- `1.0x-1.8122x` is the generated catalogued range, exposed only with
  the exact canonical experiment fingerprint.

Both receipts use the repository-owned `attested-mlx-port-wall-time` runner.
The timing includes model hashing and evidence capture, so this is an attested
adapter wall-time claim rather than a pure decode-speed claim. Exact output
passes for six generated tokens; the known seventh-token BF16 divergence keeps
the quality scope narrow.

Receipt SHA-256:

- F32: `c7775a8a3bce14dbbca143ca622ae14e1d7a81a986bae6304d6ea90c307ad04a`
- BF16: `11ede57ed58d4623214f852bf7827fece79fd77daac70f2beb11d68d076f333d`

The trusted runner lives in
[`assets/benchmarks/runners/`](../../assets/benchmarks/runners/); the input and
exact-output reference remain in
[`assets/benchmarks/qwen2.5-0.5b/`](../../assets/benchmarks/qwen2.5-0.5b/).
Per-run challenges, dependency snapshots, evidence bundles, and outputs live
under [`assets/benchmarks/attestations/`](../../assets/benchmarks/attestations/).
