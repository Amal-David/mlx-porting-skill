# Hugging Face Ecosystem Sampler Research Blog

## Assignment
Sample model cards, library tags, downloads, tasks, and license metadata to identify porting demand and artifact shapes.

## Sources sampled
- Hugging Face MLX model search (https://huggingface.co/models?library=mlx&search=mlx, accessed 2026-06-27)
- mlx-community/parakeet-tdt-0.6b-v3 model card (https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3, accessed 2026-06-27)
- lmstudio-community/gemma-4-E4B-it-MLX-4bit model card (https://huggingface.co/lmstudio-community/gemma-4-E4B-it-MLX-4bit, accessed 2026-06-27)
- mlx-community/Nemotron-Labs-Diffusion-3B-4bit model card (https://huggingface.co/mlx-community/Nemotron-Labs-Diffusion-3B-4bit, accessed 2026-06-27)

## Candidate findings
- hf-mlx-demand-sampling: Hugging Face MLX metadata should drive coverage priorities, not support claims [needs-validation] - Current Hugging Face metadata shows heavily downloaded MLX-tagged quantized multimodal models and MLX ASR models, so the skill should use HF as a demand and artifact-shape sampler while requiring separate implementation validation.
- hf-model-card-publication-lint: Hugging Face MLX publications need model-card linting [adopted] - MLX model cards expose mutable but important metadata such as library name, tags, base model, license, quantization, and custom-code/package requirements.

## Decision notes
- Hugging Face metadata is useful for prioritization, especially when GitHub contributor mining misses demand signals.

## Open validation
- hf-mlx-demand-sampling: Turn top HF demand clusters into representative golden scenarios or runbook gaps.
- hf-model-card-publication-lint: Add a model-card lint script only when publication automation is introduced.
