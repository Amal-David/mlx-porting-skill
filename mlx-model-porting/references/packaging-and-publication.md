# Packaging and publication

## Required artifact set

- model weights in a safe format;
- complete config and generation config;
- tokenizer/processor/feature extractor/vocoder/codec configs;
- conversion manifest with source revision and transforms;
- quantization configuration and module exclusions;
- smoke-test fixtures and expected summaries;
- model card with original attribution and license;
- compatibility table for MLX and package versions;
- Hugging Face model-card metadata when publishing there: `library_name`, tags, base model, license, quantization, custom-code requirement, and minimum package versions;
- raw benchmark and quality reports;
- known limitations and unsupported paths.

## Reproducibility manifest

Record SHA-256 for source files where practical, converter commit, command line, environment lock, and date. For sharded weights, record shard index and all shard hashes.

## Safe formats

Prefer safetensors or similarly inspectable data formats. Avoid distributing pickled Python objects. If upstream only provides pickle-based weights, perform conversion in an isolated reviewed environment and publish the safe output only when licensing permits.

## Model card claims

Acceptable:

- “Converted from revision X with converter Y.”
- “On M4 Max 128 GB, workload Z, median decode was N under configuration C.”
- “Quality metric changed from A to B on dataset D.”

Not acceptable:

- “Lossless” without a defined quality/equivalence test.
- “2× faster” without baseline, hardware, workload, and raw results.
- “Works on MLX” when only one untested path loads.
- “Official” for a community port.

## Optional `.mlxfn` export smoke

For ports that expose a stable callable inference surface, add an optional MLX function-export smoke test. Record the example input shapes, dtypes, enclosed arrays, and whether shapeless export is used. Treat `.mlxfn` as a release smoke artifact, not a replacement for normal Python/config/weight publication, and do not imply it supports untested shapes or processors.

## Hugging Face publication lint

Before uploading a converted checkpoint or adapter, lint the model card for:

- source model and revision;
- `library_name`/tags that match the actual loader;
- base model and quantization fields;
- license and gated-use terms;
- custom-code or main-branch package requirements;
- tokenizer/processor compatibility;
- unsupported modes such as batching, streaming, training, or speculative decoding.

Hugging Face cards are mutable metadata. Use them for publication hygiene and artifact-shape sampling, not as benchmark evidence.

## Publication checklist

- license permits derivative weights and redistribution;
- gated/source terms preserved;
- no secrets, local paths, or private samples;
- no remote code required unless explicitly documented and reviewed;
- loading from a clean environment passes;
- deterministic smoke test passes;
- README commands use pinned/minimum compatible versions;
- limitations include unsupported batching, streaming, training, or quantization modes.
