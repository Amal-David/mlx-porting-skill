# Research Loop 2026-06-27-deep-research-loop

Objective: Deepen MLX porting evidence beyond GitHub with subagent research, non-GitHub sampling, and auditable blogs

Review only: True
Findings: 16
Non-GitHub lanes covered: hugging_face, official_docs, packages, papers, repo_local_audit, technical_blogs

## Decision Counts
- adopted: 10
- held: 1
- rejected: 0
- needs-validation: 5

## Adopted
- official-custom-metal-validation: Custom Metal kernels need official API plus local fallback gates - The official MLX docs expose custom Metal kernel surfaces, but the skill should keep treating every custom kernel as mode-scoped until a readable fallback, parity oracle, and end-to-end benchmark exist.
- official-memory-fit-gate: MLX memory APIs should back fit claims - MLX exposes peak-memory measurement and a graph-evaluation memory limit API, so model-fit claims should include peak memory and budget-pressure experiments instead of relying on a single successful run.
- official-mlxfn-export-smoke: .mlxfn export is an optional shape/dtype-specific smoke gate - MLX function export can support a release smoke test, but exported behavior is tied to example shapes/dtypes unless shapeless export is used deliberately.
- official-compile-state-checklist: Compiled decode state needs explicit stale-state checks - The MLX compile documentation makes shape/type/input-count and captured-state behavior important enough to test concretely in compiled decode loops.
- hf-model-card-publication-lint: Hugging Face MLX publications need model-card linting - MLX model cards expose mutable but important metadata such as library name, tags, base model, license, quantization, and custom-code/package requirements.
- pypi-version-drift-sampling: Package registries should be sampled in research loops - Current PyPI metadata exposes active releases for mlx, mlx-lm, mlx-vlm, and mlx-audio; the skill should capture package versions during research so guidance does not silently rely on stale CLI or API assumptions.
- pypi-batched-cache-isolation: Batched cache isolation needs explicit contamination tests - mlx-lm package metadata includes a yanked release note for batched KV-cache cross contamination, which is enough to require mixed-batch isolation tests in serving guidance.
- pypi-audio-dependency-frontends: Audio ports need optional dependency and frontend matrices - mlx-audio package metadata names optional language/frontend and compression dependencies, so audio and ASR runbooks should capture dependency-driven frontend differences.
- pypi-vlm-cache-and-speculation-caveats: VLM package metadata should tighten cache and speculative gates - mlx-vlm package metadata describes path-keyed vision feature caching, speculative decoding pairings/windows, thinking budgets, and TurboQuant caveats; the skill should retain stronger cache keys and require acceptance/window/tokenizer gates.
- review-only-agent-harness-design: Dynamic research harness should scaffold prompts and record execution state - A separate review artifact lane fits the repo better than automatic asset mutation: default runs should scaffold prompts and mark agents not run, while explicit returned findings are ingested with blogs and synthesis.

## Held
- technical-blog-benchmark-caveat: Technical blog claims need benchmark protocol capture - Technical posts can reveal MLX deployment or optimization tactics, but the skill should not ingest speed claims unless hardware, software versions, workload, baseline, and quality gates are recorded.

## Rejected
- None

## Needs Validation
- paper-vllm-mlx-serving-patterns: Apple Silicon serving papers should feed validation-first serving research - The vllm-mlx paper reports continuous batching and content-based vision prefix caching on Apple Silicon, which is relevant to MLX serving guidance but still needs implementation-level review and local reproduction before promotion.
- hf-mlx-demand-sampling: Hugging Face MLX metadata should drive coverage priorities, not support claims - Current Hugging Face metadata shows heavily downloaded MLX-tagged quantized multimodal models and MLX ASR models, so the skill should use HF as a demand and artifact-shape sampler while requiring separate implementation validation.
- golden-scenario-family-coverage-gap: Golden scenarios cover only 3 of 14 declared architecture families - The skill declares 14 architecture families, but the golden scenario harness covers dense decoder, MoE decoder, and neural audio codec only; comprehensive MLX porting needs scenario gates for every declared family.
- non-hf-source-format-gap: Source-format coverage should expand beyond PyTorch and Hugging Face - The front-door skill description and intake flow emphasize PyTorch and Hugging Face; comprehensive MLX porting also needs static intake for ONNX, JAX/Flax, TensorFlow/Keras, GGUF, Core ML, and checkpoint-only repos.
- missing-domain-runbooks-gap: Non-generative CV, structured/time-series/recsys, graph/scientific ML, and training need dedicated tracks - Contributor mining did not cover several important MLX porting domains. The skill should add research tracks for non-generative computer vision, structured and time-series models, graph/scientific ML, and training/fine-tuning as port targets.

## Limitations
- This loop used live host subagents for repo-local audit and harness architecture, but source-collection automation is still scaffold/fixture-based.
- The Hugging Face result was gathered through the installed Hugging Face connector; detailed model-card parsing remains a follow-up.
- The arXiv source was sampled through the arXiv API and should be deep-reviewed before any serving technique promotion.
- Some guessed MLX doc URLs returned 404 and were excluded from evidence.
