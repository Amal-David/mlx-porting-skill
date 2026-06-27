# Official Docs Cartographer Research Blog

## Assignment
Find MLX or Apple-supported APIs, runtime constraints, packaging rules, and validation gates that should constrain porting guidance.

## Sources sampled
- MLX Custom Metal Kernels documentation (https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html, accessed 2026-06-27)
- mlx.core.fast.metal_kernel API (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html, accessed 2026-06-27)
- mlx.core.get_peak_memory API (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html, accessed 2026-06-27)
- mlx.core.set_memory_limit API (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html, accessed 2026-06-27)
- MLX exporting functions documentation (https://ml-explore.github.io/mlx/build/html/usage/export.html, accessed 2026-06-27)
- MLX compilation documentation (https://ml-explore.github.io/mlx/build/html/usage/compile.html, accessed 2026-06-27)

## Candidate findings
- official-custom-metal-validation: Custom Metal kernels need official API plus local fallback gates [adopted] - The official MLX docs expose custom Metal kernel surfaces, but the skill should keep treating every custom kernel as mode-scoped until a readable fallback, parity oracle, and end-to-end benchmark exist.
- official-memory-fit-gate: MLX memory APIs should back fit claims [adopted] - MLX exposes peak-memory measurement and a graph-evaluation memory limit API, so model-fit claims should include peak memory and budget-pressure experiments instead of relying on a single successful run.
- official-mlxfn-export-smoke: .mlxfn export is an optional shape/dtype-specific smoke gate [adopted] - MLX function export can support a release smoke test, but exported behavior is tied to example shapes/dtypes unless shapeless export is used deliberately.
- official-compile-state-checklist: Compiled decode state needs explicit stale-state checks [adopted] - The MLX compile documentation makes shape/type/input-count and captured-state behavior important enough to test concretely in compiled decode loops.

## Decision notes
- Official MLX docs are adoption-quality for API existence, but performance claims still need local benchmarks.
- Earlier guessed custom-extension URLs returned 404; only reachable docs are cited.

## Open validation
- official-custom-metal-validation: Keep each custom kernel finding tied to mode coverage, fallback, local parity tests, and model-level benchmark evidence.
- official-memory-fit-gate: Add live benchmark examples only when a real port run provides raw memory counters.
- official-mlxfn-export-smoke: Use only for ports with a stable callable inference surface.
- official-compile-state-checklist: Add a concrete compiled decode fixture if the repo later ships runtime examples.
