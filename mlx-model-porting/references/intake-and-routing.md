# Intake and routing

## Required intake record

Create one immutable record before coding:

| Field | Required content |
|---|---|
| Source | Repository/model identifier and pinned revision |
| License | Code, weight, dataset, and derivative/distribution terms |
| Task | Exact inference/training task, not a broad label |
| Source runtime | Framework, version, custom extensions, remote code |
| Architecture | Family, submodules, state/cache, modality boundaries |
| Artifacts | Configs, weights, tokenizer/processor, generation config, code |
| Target | MLX core, MLX-LM, MLX-VLM, MLX-Audio, Swift, or standalone |
| Hardware | Mac chip, memory, OS, MLX version, power mode |
| Objective | TTFT, decode latency, throughput, RTF, memory, quality |
| Constraints | Context/audio length, batch, streaming, precision, license |

## Static-first inspection

Do not import the model package during initial inspection. Read:

- `config.json`, `generation_config.json`, processor/preprocessor configs;
- safetensors headers and shard index;
- ONNX `ModelProto` metadata: IR version, opsets, graph inputs/outputs,
  initializers, node operator/domain counts, external-data references, and
  unsupported-op hold conditions;
- GGUF headers and metadata: version, metadata keys, architecture, tokenizer
  model, source/base-model provenance keys, file type, quantization version, and
  tensor table;
- Flax/Orbax checkpoint metadata: msgpack artifacts, checkpoint metadata files,
  tree paths, and params-tree hold conditions;
- TensorFlow/Keras metadata: SavedModel signature keys, method names, variable
  file presence, Keras archive config, layer classes, and weight members;
- Core ML package metadata: package manifest version, root model identifier,
  model files, and weight files;
- repository tree and Python filenames;
- model card frontmatter and license files;
- `auto_map`, custom architecture names, and nonstandard config keys;
- tokenizer, feature extractor, sample rate, hop/window sizes, codebook counts;
- tied embeddings, shared submodules, and adapter metadata.

Run `scripts/inspect_model.py`. Network download is opt-in.

ONNX, GGUF, Flax/Orbax, TensorFlow SavedModel, Keras archive, and Core ML
package support here is static intake only. It can identify routing signals and
hold conditions, but it is not conversion support. Before using any of these
formats for a port, record unsupported operators, source/base-model provenance,
tokenizer or processor compatibility, source-framework oracle outputs, and an
MLX parity fixture.

## Architecture routing signals

Use the machine-readable aliases in `assets/architectures.yaml`. Signals should be combined rather than trusted individually:

1. exact `model_type`;
2. `architectures` class names;
3. config-key fingerprints;
4. layer/module names in the weight header;
5. task metadata and processor class;
6. source code graph when custom code is unavoidable.

Return multiple candidates when confidence is low. A wrong family route is more expensive than an explicit uncertainty.

## Target selection

- **MLX-LM**: decoder LMs, compatible MoE/hybrid language models, standard generation and LoRA workflows.
- **MLX-VLM**: image/video/audio-conditioned language models and multimodal serving patterns.
- **MLX-Audio**: TTS, STT, speech-to-speech, codecs, vocoders, and audio-specific conversion/serving.
- **Standalone MLX**: architectures outside existing package contracts, research prototypes, training-first ports, or custom applications.
- **MLX Swift**: only after a Python MLX oracle is stable unless Swift is the canonical source.

## Red flags

Pause for review when:

- weights require arbitrary pickle execution and no safe conversion path exists;
- the only implementation is gated or unpinned remote code;
- license metadata is missing or conflicts across code and weights;
- source preprocessing is undocumented;
- model output depends on nondeterministic external services;
- architecture class and weight key patterns disagree;
- hidden state/cache semantics cannot be inferred from source tests.
