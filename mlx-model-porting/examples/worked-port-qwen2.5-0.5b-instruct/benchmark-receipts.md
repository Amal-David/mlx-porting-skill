# Benchmark receipt pointers

The checked-in receipts measure separate-process model load plus six greedy
tokens on Apple M4 Pro with 48 GB unified memory. Both variants reproduce the
same exact six-token quality artifact.

| Variant | Receipt | Median wall time | CV | Classification | State |
|---|---|---:|---:|---|---|
| F32 baseline | [`qwen2.5-0.5b-port-f32.json`](../../assets/benchmarks/qwen2.5-0.5b-port-f32.json) | 1.6550946250 s | 0.0060855 | `performance_observation` | `execution_attested=false`; external signature missing |
| BF16 candidate | [`qwen2.5-0.5b-port-bf16.json`](../../assets/benchmarks/qwen2.5-0.5b-port-bf16.json) | 0.9133066250 s | 0.0090013 | `performance_observation` | `execution_attested=false`; external signature missing |

- `1.8122003933x` is the observed receipt inverse-wall-time ratio, not a portable guarantee.
- `1.02x` is the required receipt noise floor.
- No effective range is generated; the observed ratio remains non-promotable.

Both receipts use the repository-owned `attested-mlx-port-wall-time` runner.
The timing includes model hashing and evidence capture. The challenge and
digest-bound bundles establish internal consistency and
reproducibility-on-request, not authenticity: SHA-256 is not a signature. A
protected Apple-Silicon signer and out-of-repository trust anchor are future
work. Exact output is captured for six generated tokens; no quality claim is
made beyond that artifact.

Receipt SHA-256:

- F32: `c7775a8a3bce14dbbca143ca622ae14e1d7a81a986bae6304d6ea90c307ad04a`
- BF16: `11ede57ed58d4623214f852bf7827fece79fd77daac70f2beb11d68d076f333d`

The trusted runner lives in
[`assets/benchmarks/runners/`](../../assets/benchmarks/runners/); the input and
exact-output reference remain in
[`assets/benchmarks/qwen2.5-0.5b/`](../../assets/benchmarks/qwen2.5-0.5b/).
Per-run challenges, dependency snapshots, evidence bundles, and outputs live
under [`assets/benchmarks/attestations/`](../../assets/benchmarks/attestations/).
