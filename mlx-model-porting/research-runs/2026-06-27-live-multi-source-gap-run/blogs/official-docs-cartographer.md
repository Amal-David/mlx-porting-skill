# Official Docs Cartographer Research Blog

## Assignment
Find MLX or Apple-supported APIs, runtime constraints, packaging rules, and validation gates that should constrain porting guidance.

## Planned sampling
- official_docs: MLX documentation index [official-doc] - https://ml-explore.github.io/mlx/build/html/index.html
- official_docs: MLX custom Metal kernels [official-doc] - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- official_docs: Apple Metal developer documentation [official-doc] - https://developer.apple.com/metal/
- packages: PyPI mlx metadata [package-metadata] - https://pypi.org/pypi/mlx/json
- packages: PyPI mlx-lm metadata [package-metadata] - https://pypi.org/pypi/mlx-lm/json
- packages: PyPI mlx-vlm metadata [package-metadata] - https://pypi.org/pypi/mlx-vlm/json
- packages: PyPI mlx-audio metadata [package-metadata] - https://pypi.org/pypi/mlx-audio/json

## Sources sampled
- mlx.core.value_and_grad (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html, accessed 2026-06-27)
- mlx.nn.value_and_grad (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html, accessed 2026-06-27)
- MLX Module documentation (https://ml-explore.github.io/mlx/build/html/python/nn/module.html, accessed 2026-06-27)
- MLX Module documentation (https://ml-explore.github.io/mlx/build/html/python/nn/module.html, accessed 2026-06-27)
- mlx.nn.Module.load_weights (https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.load_weights.html, accessed 2026-06-27)
- MLX Optimizers (https://ml-explore.github.io/mlx/build/html/python/optimizers.html, accessed 2026-06-27)
- Saving and Loading Arrays (https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html, accessed 2026-06-27)
- MLX Compilation (https://ml-explore.github.io/mlx/build/html/usage/compile.html, accessed 2026-06-27)
- MLX Examples MNIST training loop (https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py, accessed 2026-06-27)
- MLX-LM trainer (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py, accessed 2026-06-27)
- mlx.core.get_peak_memory (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html, accessed 2026-06-27)
- mlx.core.checkpoint (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.checkpoint.html, accessed 2026-06-27)
- MLX-LM LoRA documentation (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md, accessed 2026-06-27)
- MLX examples LoRA script (https://github.com/ml-explore/mlx-examples/blob/main/lora/lora.py, accessed 2026-06-27)

## Candidate findings
- official-autograd-value-and-grad-contract: Autograd gives a concrete training-loop contract [adopted] - MLX exposes scalar-loss gradient APIs for arbitrary callables and Module trainable parameters, so training guidance can require loss and gradient-tree parity without claiming a trained port works.
- official-training-checkpoint-resume-contract: Training checkpoints need model, optimizer, random, scheduler, and data state [adopted] - MLX Module and optimizer docs support serializing weights and optimizer state, but optimizer state alone is not sufficient for exact resume because configuration, scheduler, random state, and data cursor live outside a plain state tree.
- official-compiled-training-state-gate: Compiled training requires explicit state capture [adopted] - MLX compile guidance and official examples show compiled training must capture model state, optimizer state, and random state for stochastic modules; eager parity should precede compiled training claims.
- official-training-memory-and-lora-gates: Memory APIs and official LoRA paths define conservative training gates [adopted] - MLX memory counters, memory limits, gradient checkpointing, and MLX-LM LoRA code provide reference patterns for training-memory and adapter validation, but not a general non-LLM training claim.

## Decision notes
- Official MLX docs sampled are current MLX 0.31.2 documentation pages accessed on 2026-06-27.
- Adopted findings are adoption-quality for backlog and reference guidance only; they do not establish training parity for any model family.
- The existing training-as-port-target backlog item should be enriched with concrete MLX API contracts and validation gates before any support claim.

## Open validation
- official-autograd-value-and-grad-contract: Add a tiny local fixture for loss and gradient parity in one frozen-base adapter case and one full-train case.
- official-training-checkpoint-resume-contract: Define a training checkpoint manifest and test strict full-weight reload plus explicit partial-adapter reload on a tiny MLX Module.
- official-compiled-training-state-gate: Create a tiny eager-versus-compiled train-step fixture with dropout enabled and disabled.
- official-training-memory-and-lora-gates: Add a local training-memory probe and a first-class LLM adapter validation slice using a tiny dataset.
