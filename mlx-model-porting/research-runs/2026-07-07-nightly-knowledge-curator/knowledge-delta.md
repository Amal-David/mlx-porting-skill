# MLX Knowledge Curator Delta

- Run id: `2026-07-07-nightly-knowledge-curator`
- Generated at: `2026-07-07T22:03:01+00:00`
- Graph nodes: 431
- Graph edges: 227

## New Unread Sources
- `candidate:paper:eliteformer-an-efficient-transformer-for-fpgas` - ELiTeFormer: An Efficient Transformer for FPGAs (https://arxiv.org/abs/2607.03652)
- `candidate:paper:lynx-progressive-speculative-quantization-for-accelerating-kv-transfer-in-long-context-inference` - Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference (https://arxiv.org/abs/2607.01831)
- `candidate:paper:rabitqcache-rotated-binary-quantization-for-kvcache-in-long-context-llm-inference` - RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference (https://arxiv.org/abs/2606.31519)

## Already Read Sources
- `candidate:paper:native-llm-and-mllm-inference-at-scale-on-apple-silicon` - Native LLM and MLLM Inference at Scale on Apple Silicon (https://arxiv.org/abs/2601.19139)
- `candidate:paper:rope-aware-bit-allocation-for-kv-cache-quantization` - RoPE-Aware Bit Allocation for KV-Cache Quantization (https://arxiv.org/abs/2606.24033)
- `candidate:paper:agent-assisted-side-channel-attacks-on-non-prefix-kv-cache-in-rag` - Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG (https://arxiv.org/abs/2606.21842)
- `candidate:paper:ultraquant-4-bit-kv-caching-for-context-heavy-agents` - UltraQuant: 4-bit KV Caching for Context-Heavy Agents (https://arxiv.org/abs/2606.20474)
- `candidate:paper:pushing-the-limits-of-on-device-streaming-asr-a-compact-high-accuracy-english-model-for-low-latency-inference` - Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference (https://arxiv.org/abs/2604.14493)
- `candidate:repository:ml-explore-mlx` - ml-explore/mlx (https://github.com/ml-explore/mlx)
- `candidate:repository:ml-explore-mlx-lm` - ml-explore/mlx-lm (https://github.com/ml-explore/mlx-lm)
- `candidate:repository:ml-explore-mlx-examples` - ml-explore/mlx-examples (https://github.com/ml-explore/mlx-examples)
- `candidate:repository:blaizzy-mlx-vlm` - Blaizzy/mlx-vlm (https://github.com/Blaizzy/mlx-vlm)
- `candidate:repository:blaizzy-mlx-audio` - Blaizzy/mlx-audio (https://github.com/Blaizzy/mlx-audio)
- `candidate:repository:waybarrios-vllm-mlx` - waybarrios/vllm-mlx (https://github.com/waybarrios/vllm-mlx)
- `candidate:repository:anthropics-skills` - anthropics/skills (https://github.com/anthropics/skills)

## Updated Sources
- `candidate:repository:openai-codex` - openai/codex (https://github.com/openai/codex)

## New Approach Leads
- `candidate:paper:eliteformer-an-efficient-transformer-for-fpgas` -> approach:adaptive-kv-quantization, approach:native-low-bit-weight-quantization, approach:uniform-kv-quantization
- `candidate:paper:lynx-progressive-speculative-quantization-for-accelerating-kv-transfer-in-long-context-inference` -> approach:adaptive-kv-quantization, approach:multimodal-content-prefix-cache, outcome:decoder-mlx-lm-working-route
- `candidate:paper:rabitqcache-rotated-binary-quantization-for-kvcache-in-long-context-llm-inference` -> approach:adaptive-kv-quantization, backlog:structured-timeseries-recsys, approach:multimodal-content-prefix-cache

## Gap Hints

`quantization`, `adaptive`, `cache`, `content`, `context`, `inference`, `long`, `multimodal`, `prefix`, `accelerating`, `binary`, `bit`

## Policy

- Review-only. Do not auto-promote a candidate source.
- A candidate can update skill/app/CLI guidance only after provenance, validation gate, rollback condition, and tests are recorded.
