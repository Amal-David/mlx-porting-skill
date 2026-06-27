# Hugging Face Ecosystem Sampler Research Blog

## Assignment
Sample model cards, library tags, downloads, tasks, and license metadata to identify porting demand and artifact shapes.

## Sources sampled
- mlx-onnx export library (https://github.com/skryl/mlx-onnx/tree/c9aec867c76b, accessed 2026-06-27)
- ONNX IR specification (https://github.com/onnx/onnx/blob/main/docs/IR.md, accessed 2026-06-27)
- TensorFlow SavedModel guide (https://www.tensorflow.org/guide/saved_model, accessed 2026-06-27)
- Hugging Face Hub GGUF documentation (https://huggingface.co/docs/hub/en/gguf, accessed 2026-06-27)

## Candidate findings
- top1000-source-format-onnx-adjacent: MLX ONNX export tooling is adjacent evidence, not intake support [held] - A top-1000 contributor-owned mlx-onnx repository shows MLX-to-ONNX export ecosystem work, but it does not prove ONNX-to-MLX model intake. Keep ONNX/JAX/TF/GGUF/Core ML intake as a P0 validation backlog.
- source-format-static-intake-expanded: Static intake must cover source formats beyond PyTorch and Hugging Face [needs-validation] - Primary docs for ONNX, Flax/Orbax, TensorFlow/Keras, GGUF, Core ML, and safetensors show inspectable metadata surfaces. These can support intake triage and unsupported-op reporting, but not automatic conversion claims.

## Decision notes
- The contributor sweep should feed demand and gap mapping, while Hugging Face metadata remains a separate prioritization lane.

## Open validation
- top1000-source-format-onnx-adjacent: Build source-format fixtures and route them without importing source framework code.
- source-format-static-intake-expanded: Implement source-format fixtures and make ambiguous checkpoint-only repos fail loud.
