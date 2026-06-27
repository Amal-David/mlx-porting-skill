# Coverage Skeptic Research Blog

## Assignment
Look for blind spots, unsupported architecture families, missing validation gates, and overclaimed optimizations.

## Sources sampled
- tests/test_scenarios.py expected coverage (file:/Users/amal/Downloads/mlx-porting-skill/tests/test_scenarios.py, accessed 2026-06-27)
- architectures.yaml declared families (file:/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/architectures.yaml, accessed 2026-06-27)
- SKILL.md source scope (file:/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/SKILL.md, accessed 2026-06-27)
- intake-and-routing.md (file:/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/references/intake-and-routing.md, accessed 2026-06-27)
- Subagent coverage audit for MLX porting skill (codex-subagent://019f08a0-1e40-7b31-ad73-4328676b9018, accessed 2026-06-27)
- Subagent harness architecture audit (codex-subagent://019f08a0-03eb-7861-a33a-6927175c9b7b, accessed 2026-06-27)
- maintenance-and-provenance.md review-only update pipeline (file:/Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/references/maintenance-and-provenance.md, accessed 2026-06-27)

## Candidate findings
- golden-scenario-family-coverage-gap: Golden scenarios cover only 3 of 14 declared architecture families [needs-validation] - The skill declares 14 architecture families, but the golden scenario harness covers dense decoder, MoE decoder, and neural audio codec only; comprehensive MLX porting needs scenario gates for every declared family.
- non-hf-source-format-gap: Source-format coverage should expand beyond PyTorch and Hugging Face [needs-validation] - The front-door skill description and intake flow emphasize PyTorch and Hugging Face; comprehensive MLX porting also needs static intake for ONNX, JAX/Flax, TensorFlow/Keras, GGUF, Core ML, and checkpoint-only repos.
- missing-domain-runbooks-gap: Non-generative CV, structured/time-series/recsys, graph/scientific ML, and training need dedicated tracks [needs-validation] - Contributor mining did not cover several important MLX porting domains. The skill should add research tracks for non-generative computer vision, structured and time-series models, graph/scientific ML, and training/fine-tuning as port targets.
- review-only-agent-harness-design: Dynamic research harness should scaffold prompts and record execution state [adopted] - A separate review artifact lane fits the repo better than automatic asset mutation: default runs should scaffold prompts and mark agents not run, while explicit returned findings are ingested with blogs and synthesis.

## Decision notes
- The repo-local audit found the largest strategic gap: validation coverage is much narrower than declared architecture coverage.
- Source-format and non-generative-domain coverage need dedicated tracks that GitHub contributor mining will not reliably surface.

## Open validation
- golden-scenario-family-coverage-gap: Create a follow-up task graph for golden scenarios across the 11 uncovered families.
- non-hf-source-format-gap: Shape a source-format intake workstream with fixtures for at least ONNX and JAX/Flax first.
- missing-domain-runbooks-gap: Prioritize CV and source-format intake before adding lower-demand families.
- review-only-agent-harness-design: Add executor subcommands only when the repo needs to run child processes directly.
