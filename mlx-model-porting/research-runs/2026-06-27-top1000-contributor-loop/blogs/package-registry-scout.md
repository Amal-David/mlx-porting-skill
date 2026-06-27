# Package Registry Scout Research Blog

## Assignment
Track packages and release metadata that change install, conversion, runtime, or optional dependency guidance.

## Sources sampled
- tiny-llm MLX serving course repository (https://github.com/skyzh/tiny-llm/tree/efb0c89fd236, accessed 2026-06-27)
- mlx-moe dense-to-MoE assembly scripts (https://github.com/mzbac/mlx-moe/tree/11088f36eaea, accessed 2026-06-27)

## Candidate findings
- top1000-paged-kv-moe-reinforces-existing: Contributor serving repos reinforce paged-KV and MoE gates [held] - tiny-llm and mlx-moe show useful paged-KV, continuous batching, quantized expert, and synthetic MoE assembly patterns, but they reinforce existing validation gates rather than introducing a new supported optimization.

## Decision notes
- Contributor-owned serving repos can reinforce existing cache and MoE gates, but not replace official package/release evidence.

## Open validation
- top1000-paged-kv-moe-reinforces-existing: Keep as reinforcing context until a concrete port needs these patterns and has local parity fixtures.
