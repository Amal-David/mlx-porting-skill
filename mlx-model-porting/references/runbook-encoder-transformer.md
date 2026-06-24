# Runbook: encoder Transformer

## Applies to

BERT/RoBERTa/DeBERTa-like text encoders, embedding/reranker models, ViT-like encoders where modality frontend is simple, and encoder components reused by speech/audio systems.

## Architecture fingerprint

Confirm:

- bidirectional self-attention and padding-mask semantics;
- absolute, relative, rotary, or disentangled positions;
- pre/post norm;
- pooler or task heads;
- token type/segment embeddings;
- convolutional or patch frontend;
- output hidden-state selection and pooling/normalization;
- classification versus embedding/reranking contract.

## Source oracle checkpoints

Capture preprocessed IDs/features, embedding sum, attention mask after expansion, one block’s Q/K/V and output, selected layer outputs, pooled representation, normalized embedding, and task logits.

## Weight conversion

- preserve embedding component order and padding row behavior;
- map QKV fused/separate projections;
- preserve relative-position tables/biases;
- map LayerNorm epsilon and affine parameters;
- confirm pooler activation;
- preserve task-head label order;
- for image/audio frontends, document NCHW/NHWC or time/channel layout.

## Minimal MLX path

1. Port input embeddings/frontend.
2. Port one bidirectional attention block with padding mask.
3. Port complete stack.
4. Add pooling/head exactly as source.
5. Add batching and variable-length masks.
6. Save/reload and compare embeddings/logits.

## Parity traps

- inverted padding masks;
- mask added before/after scaling;
- token-type embeddings silently omitted;
- source returns a particular hidden layer rather than final;
- mean pooling includes padding;
- embedding normalization or temperature omitted;
- convolution/patch layout mismatch;
- position IDs reset incorrectly in packed inputs;
- classifier label mapping reordered.

## Optimization ladder

1. Use fast SDPA for full bidirectional attention where compatible.
2. Compile the encoder for stable batch/sequence buckets.
3. Bucket or pad lengths to control recompilation.
4. Batch requests for throughput.
5. Quantize large linears after embedding/task-quality checks.
6. Cache immutable embeddings/features for repeated inputs at the application layer.
7. For long inputs, use the architecture’s trained sparse/window strategy; do not invent truncation.

## Completion gates

- masked and unmasked batches match source;
- pooling excludes padding correctly;
- embedding similarity/ranking or classifier metrics pass;
- multiple sequence lengths do not cause uncontrolled recompilation;
- advertised batch throughput includes preprocessing and normalization.
