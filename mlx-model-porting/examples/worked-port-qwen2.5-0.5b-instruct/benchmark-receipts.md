# Benchmark receipt pointers

The checked-in receipts measure separate-process model load plus six greedy
tokens on Apple M4 Pro with 48 GB unified memory. Both variants reproduce the
same exact six-token quality artifact.

| Variant | Receipt | Median wall time | CV | Classification | Hold |
|---|---|---:|---:|---|---|
| F32 baseline | [`qwen2.5-0.5b-port-f32.json`](../../assets/benchmarks/qwen2.5-0.5b-port-f32.json) | 0.3247475000 s | 0.0250812 | `performance_observation` | baseline role; execution semantics unattested |
| BF16 candidate | [`qwen2.5-0.5b-port-bf16.json`](../../assets/benchmarks/qwen2.5-0.5b-port-bf16.json) | 0.2490912910 s | 0.0730578 | `performance_observation` | execution semantics unattested |

- `1.3037288406x` is the withheld inverse-wall-time performance observation.
- `1.1461156112x` is the receipt noise floor, not a speedup claim.

The measured change clears the noise gate. It is still not promotion-ready
because the generic external runner is deliberately
`execution_attested=false`.

Receipt SHA-256:

- F32: `47e573066f0537fd92bd9f31925c36bee645a93ce332b929bdddd719b0774902`
- BF16: `4710976c8c5df13e87ab3186168364487e0309e7f93896c847327a12a9692f86`

The digest-pinned runner, input, and exact-output reference live in
[`assets/benchmarks/qwen2.5-0.5b/`](../../assets/benchmarks/qwen2.5-0.5b/).
