# Optimize receipts — measured quantization sweeps with the quality gate

Evidence for "optimized MLX conversion" as *"prove which optimization holds,"* not *"make it fast."*
Each receipt is a real `optimize_port.py` sweep on real Metal: bf16 baseline vs 8-bit vs 4-bit-g64,
with the full quality gate (perplexity ratio, first-token agreement, degenerate-output rate).

| Model | 4-bit-g64 (naive default) | 8-bit (structured optimal) | Verdict |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 70.6 tok/s — **quality FAIL** | 56.0 tok/s — **quality PASS** | 8-bit chosen (held quality) |
| SmolLM2-360M-Instruct | 94.2 tok/s — **quality FAIL** | 87.7 tok/s — **quality PASS** | 8-bit chosen (held quality) |

**Consistent finding:** the naive default (4-bit-g64) is faster/smaller but **fails the quality gate** on
these small models; the structured pick is 8-bit, the highest throughput-per-peak-memory config that
*held quality*. The gate earns its keep — it stops you shipping a faster port that quietly degraded.

Full per-config detail (baseline, perplexity ratios, memory, methodology, non-promotable boundary flags)
is in each `*-quant-receipt.json`.

Scope/honesty: local observations for these models + this Mac + this workload; not promotion inputs or
benchmark receipts. A larger receipt set (7B+ models, more architectures) and an outlier-aware parity
gate are the follow-ups to make "optimized MLX conversion" a fully general headline.
