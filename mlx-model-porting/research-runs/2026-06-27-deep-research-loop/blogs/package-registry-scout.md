# Package Registry Scout Research Blog

## Assignment
Track packages and release metadata that change install, conversion, runtime, or optional dependency guidance.

## Sources sampled
- PyPI mlx package metadata (https://pypi.org/pypi/mlx/json, accessed 2026-06-27)
- PyPI mlx-lm package metadata (https://pypi.org/pypi/mlx-lm/json, accessed 2026-06-27)
- PyPI mlx-vlm package metadata (https://pypi.org/pypi/mlx-vlm/json, accessed 2026-06-27)
- PyPI mlx-audio package metadata (https://pypi.org/pypi/mlx-audio/json, accessed 2026-06-27)
- mlx-lm PyPI package metadata (https://pypi.org/project/mlx-lm/, accessed 2026-06-27)
- mlx-audio PyPI package metadata (https://pypi.org/project/mlx-audio/, accessed 2026-06-27)
- mlx-vlm PyPI package metadata (https://pypi.org/project/mlx-vlm/, accessed 2026-06-27)
- TurboQuant paper (https://arxiv.org/abs/2504.19874, accessed 2026-06-27)

## Candidate findings
- pypi-version-drift-sampling: Package registries should be sampled in research loops [adopted] - Current PyPI metadata exposes active releases for mlx, mlx-lm, mlx-vlm, and mlx-audio; the skill should capture package versions during research so guidance does not silently rely on stale CLI or API assumptions.
- pypi-batched-cache-isolation: Batched cache isolation needs explicit contamination tests [adopted] - mlx-lm package metadata includes a yanked release note for batched KV-cache cross contamination, which is enough to require mixed-batch isolation tests in serving guidance.
- pypi-audio-dependency-frontends: Audio ports need optional dependency and frontend matrices [adopted] - mlx-audio package metadata names optional language/frontend and compression dependencies, so audio and ASR runbooks should capture dependency-driven frontend differences.
- pypi-vlm-cache-and-speculation-caveats: VLM package metadata should tighten cache and speculative gates [adopted] - mlx-vlm package metadata describes path-keyed vision feature caching, speculative decoding pairings/windows, thinking budgets, and TurboQuant caveats; the skill should retain stronger cache keys and require acceptance/window/tokenizer gates.

## Decision notes
- Package registry state changes faster than static repo guidance; research runs should record versions sampled.

## Open validation
- pypi-version-drift-sampling: Add package-version fields to future live collector records when network execution is implemented.
- pypi-batched-cache-isolation: Add a concrete batch-cache regression when a serving harness exists.
- pypi-audio-dependency-frontends: Validate per model and language path.
- pypi-vlm-cache-and-speculation-caveats: Pin MLX-VLM source surfaces before upgrading any technique status.
