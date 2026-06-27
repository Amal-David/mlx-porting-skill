# Paper Architecture Scout Research Blog

## Assignment
Find architecture-specific algorithm constraints and evaluation methods for model families the skill supports or should support.

## Sources sampled
- Native LLM and MLLM Inference at Scale on Apple Silicon (https://arxiv.org/abs/2601.19139, accessed 2026-06-27)

## Candidate findings
- paper-vllm-mlx-serving-patterns: Apple Silicon serving papers should feed validation-first serving research [needs-validation] - The vllm-mlx paper reports continuous batching and content-based vision prefix caching on Apple Silicon, which is relevant to MLX serving guidance but still needs implementation-level review and local reproduction before promotion.

## Decision notes
- The paper source is strong enough for a research lead, but not enough to mark serving techniques supported in this skill without pinned implementation and local gates.

## Open validation
- paper-vllm-mlx-serving-patterns: Deep-review the implementation surfaces and add local serving/correctness gates before any supported technique upgrade.
