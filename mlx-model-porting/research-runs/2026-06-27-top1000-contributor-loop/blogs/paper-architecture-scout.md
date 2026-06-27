# Paper Architecture Scout Research Blog

## Assignment
Find architecture-specific algorithm constraints and evaluation methods for model families the skill supports or should support.

## Sources sampled
- mlx-rwkv WIP recurrent implementation (https://github.com/dc-dc-dc/mlx-rwkv/tree/45d9f309afa3, accessed 2026-06-27)
- mlx-ssm state-space prototype (https://github.com/j-csc/mlx-ssm/tree/51e2eb3285f7, accessed 2026-06-27)
- mlx-conformer Torch-to-MLX port scaffold (https://github.com/FL33TW00D/mlx-conformer/tree/88809a360184, accessed 2026-06-27)

## Candidate findings
- top1000-rwkv-ssm-conformer-held: RWKV, SSM, and Conformer prototypes reinforce validation gaps [needs-validation] - Contributor-owned RWKV, SSM/Mamba, and Conformer repositories are useful architecture leads, but current snapshots show WIP or incomplete kernels and no packaged parity suite strong enough for supported guidance.

## Decision notes
- Long-tail contributors exposed recurrent, SSM, MoE, and Conformer prototypes, but most remain prototype or training-context evidence.

## Open validation
- top1000-rwkv-ssm-conformer-held: Add tiny SSM/RWKV/Conformer golden scenarios before promoting any contributor-derived implementation pattern.
