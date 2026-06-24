# Evidence index

**Snapshot date:** 2026-06-24  
**Total records:** 320

This is a navigable rendering of `mlx-model-porting/assets/sources.yaml`. Review depth is an engineering claim about this corpus, not a claim that every source was reproduced on MLX.

## Corpus summary

### By source kind

| Kind | Count |
|---|---:|
| `paper` | 226 |
| `repository` | 33 |
| `official-doc` | 30 |
| `source-code` | 19 |
| `technical-blog` | 8 |
| `release` | 3 |
| `issue-report` | 1 |

### By review depth

| Review depth | Count | Meaning |
|---|---:|---|
| `screened` | 171 | Abstract/readme/code path reviewed for relevance and limits. |
| `synthesized` | 125 | Directly informed a runbook, rule, or technique decision. |
| `indexed` | 24 | Catalogued for future review; do not represent as fully reviewed. |

### Most represented topics

| Topic | Records |
|---|---:|
| `speculative-decoding` | 35 |
| `tts` | 33 |
| `mlx` | 32 |
| `quantization` | 32 |
| `audio-codec` | 24 |
| `serving` | 21 |
| `vocoder` | 16 |
| `streaming` | 16 |
| `attention` | 15 |
| `kv-quantization` | 15 |
| `asr` | 14 |
| `kernels` | 11 |
| `moe` | 11 |
| `multimodal` | 10 |
| `vision-language` | 10 |
| `prefix-cache` | 9 |
| `audio` | 9 |
| `decoder-transformer` | 9 |
| `long-context` | 8 |
| `diffusion` | 7 |
| `sparse-attention` | 7 |
| `agent-skills` | 6 |
| `metal` | 6 |
| `continuous-batching` | 6 |
| `stt` | 6 |
| `on-device` | 6 |
| `weight-quantization` | 6 |
| `kernel` | 6 |
| `kv-compression` | 6 |
| `audio-language` | 6 |
| `mlx-audio` | 6 |
| `runtime` | 5 |
| `cache` | 5 |
| `codec` | 5 |
| `recurrent` | 5 |
| `flow-matching` | 5 |
| `kv-pruning` | 5 |
| `vllm-mlx` | 5 |
| `training` | 4 |
| `kv-cache` | 4 |

## Full source list

| ID | Depth | Kind | Topics | Source |
|---|---|---|---|---|
| `agent-skills-spec` | `synthesized` | `official-doc` | `agent-skills`, `format`, `validation` | [Agent Skills specification](https://agentskills.io/specification) |
| `agent-skills-repo` | `screened` | `repository` | `agent-skills`, `examples` | [Anthropic Agent Skills repository](https://github.com/anthropics/skills) |
| `skills-ref-repo` | `screened` | `repository` | `agent-skills`, `validation` | [Agent Skills reference validator](https://github.com/agentskills/agentskills) |
| `codex-skill-loader` | `synthesized` | `source-code` | `agent-skills`, `codex` | [Codex skill loader implementation](https://github.com/openai/codex/blob/main/codex-rs/core-skills/src/loader.rs) |
| `mlx-repo` | `synthesized` | `repository` | `mlx`, `runtime`, `metal`, `cuda` | [MLX array framework repository](https://github.com/ml-explore/mlx) |
| `mlx-docs` | `synthesized` | `official-doc` | `mlx`, `api`, `runtime` | [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html) |
| `mlx-doc-lazy` | `synthesized` | `official-doc` | `mlx`, `lazy-evaluation` | [MLX lazy evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) |
| `mlx-doc-compile` | `synthesized` | `official-doc` | `mlx`, `compile`, `fusion` | [MLX compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html) |
| `mlx-doc-fast-sdpa` | `synthesized` | `official-doc` | `mlx`, `attention`, `kernels` | [MLX fast scaled dot product attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) |
| `mlx-doc-custom-extensions` | `synthesized` | `official-doc` | `mlx`, `metal`, `custom-kernels` | [MLX custom extensions documentation](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) |
| `mlx-doc-fast-rms-norm` | `synthesized` | `official-doc` | `mlx`, `normalization`, `kernels` | [MLX fast RMSNorm API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html) |
| `mlx-doc-fast-layer-norm` | `synthesized` | `official-doc` | `mlx`, `normalization`, `kernels` | [MLX fast LayerNorm API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.layer_norm.html) |
| `mlx-doc-fast-rope` | `synthesized` | `official-doc` | `mlx`, `rope`, `kernels` | [MLX fast RoPE API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html) |
| `mlx-doc-quantized-matmul` | `synthesized` | `official-doc` | `mlx`, `quantization`, `kernels` | [MLX quantized matmul API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) |
| `mlx-doc-gather-mm` | `synthesized` | `official-doc` | `mlx`, `gather`, `kernels`, `moe` | [MLX gather matmul API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html) |
| `mlx-doc-gather-qmm` | `synthesized` | `official-doc` | `mlx`, `gather`, `quantization`, `kernels`, `moe` | [MLX gather quantized matmul API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html) |
| `mlx-doc-block-masked-mm` | `synthesized` | `official-doc` | `mlx`, `block-sparse`, `kernels` | [MLX block-masked matmul API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html) |
| `mlx-doc-hadamard-transform` | `synthesized` | `official-doc` | `mlx`, `hadamard`, `quantization`, `kernels` | [MLX Hadamard transform API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html) |
| `mlx-doc-core-quantize` | `synthesized` | `official-doc` | `mlx`, `quantization`, `low-bit-formats` | [MLX core quantize API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) |
| `mlx-doc-nn-quantize` | `synthesized` | `official-doc` | `mlx`, `quantization`, `linear` | [MLX neural network quantize API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html) |
| `mlx-release-0312` | `synthesized` | `release` | `mlx`, `release`, `threading`, `quantized-matmul` | [MLX v0.31.2 release](https://github.com/ml-explore/mlx/releases/tag/v0.31.2) |
| `mlx-lm-repo` | `synthesized` | `repository` | `mlx`, `language-models`, `training`, `generation` | [MLX-LM repository](https://github.com/ml-explore/mlx-lm) |
| `mlx-lm-release-0313` | `screened` | `release` | `mlx-lm`, `cache`, `threading` | [MLX-LM v0.31.3 release](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3) |
| `mlx-lm-convert` | `synthesized` | `source-code` | `conversion`, `quantization`, `weights` | [MLX-LM model conversion and quantization](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/convert.py) |
| `mlx-lm-cache` | `synthesized` | `source-code` | `kv-cache`, `rotating-cache`, `quantization` | [MLX-LM cache implementations](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py) |
| `mlx-lm-speculative` | `synthesized` | `source-code` | `speculative-decoding`, `generation` | [MLX-LM speculative generation implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) |
| `mlx-lm-models` | `synthesized` | `source-code` | `architectures`, `language-models` | [MLX-LM architecture implementations](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models) |
| `mlx-examples-repo` | `screened` | `repository` | `examples`, `whisper`, `encodec`, `diffusion` | [MLX examples repository](https://github.com/ml-explore/mlx-examples) |
| `mlx-vlm-repo` | `synthesized` | `repository` | `mlx`, `multimodal`, `serving` | [MLX-VLM repository](https://github.com/Blaizzy/mlx-vlm) |
| `mlx-vlm-readme` | `synthesized` | `source-code` | `speculative-decoding`, `prefix-cache`, `continuous-batching`, `kv-quantization` | [MLX-VLM serving, caching, and speculative decoding documentation](https://github.com/Blaizzy/mlx-vlm/blob/main/README.md) |
| `mlx-audio-repo` | `synthesized` | `repository` | `mlx`, `audio`, `tts`, `stt`, `codec` | [MLX-Audio repository](https://github.com/Blaizzy/mlx-audio) |
| `mlx-audio-convert` | `synthesized` | `source-code` | `audio`, `conversion`, `quantization`, `model-detection` | [MLX-Audio conversion implementation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/convert.py) |
| `mlx-audio-codecs` | `synthesized` | `source-code` | `audio-codec`, `vocoder`, `tokenizer` | [MLX-Audio codec implementations](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/codec/models) |
| `mlx-audio-models` | `synthesized` | `source-code` | `tts`, `stt`, `sts`, `audio` | [MLX-Audio model implementations](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio) |
| `mlx-audio-release-044` | `synthesized` | `release` | `audio`, `streaming`, `cache`, `codec` | [MLX-Audio v0.4.4 release](https://github.com/Blaizzy/mlx-audio/releases/tag/v0.4.4) |
| `vllm-mlx-repo` | `screened` | `repository` | `serving`, `continuous-batching`, `paged-kv`, `prefix-cache` | [vLLM-MLX serving repository](https://github.com/waybarrios/vllm-mlx) |
| `mlx-dev-skill` | `synthesized` | `repository` | `agent-skills`, `mlx`, `development` | [General MLX development Agent Skill](https://github.com/ettrickshepherd/mlx-dev-skill) |
| `openclaw-mlx-audio` | `screened` | `repository` | `agent-skills`, `audio`, `tts` | [OpenClaw MLX-Audio integration](https://github.com/cosformula/openclaw-mlx-audio) |
| `mlx-audio-swift` | `indexed` | `repository` | `swift`, `audio`, `mlx` | [MLX-Audio Swift repository](https://github.com/Blaizzy/mlx-audio-swift) |
| `mlx-swift` | `screened` | `repository` | `swift`, `mlx` | [MLX Swift repository](https://github.com/ml-explore/mlx-swift) |
| `mlx-metal-examples` | `screened` | `source-code` | `metal`, `custom-kernels` | [MLX Metal kernel examples and tests](https://github.com/ml-explore/mlx/tree/main/examples/extensions) |
| `apple-metal-guide` | `indexed` | `official-doc` | `metal`, `gpu`, `compute` | [Metal compute documentation](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu) |
| `apple-mpsgraph` | `indexed` | `official-doc` | `metal`, `graph`, `operators` | [Metal Performance Shaders Graph documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph) |
| `safetensors-repo` | `screened` | `repository` | `weights`, `security`, `serialization` | [Safetensors format repository](https://github.com/huggingface/safetensors) |
| `huggingface-transformers` | `screened` | `repository` | `source-oracle`, `architectures`, `model-config` | [Transformers architecture and model repository](https://github.com/huggingface/transformers) |
| `pytorch-repo` | `indexed` | `repository` | `source-oracle`, `operators` | [PyTorch repository](https://github.com/pytorch/pytorch) |
| `paper-1706-03762` | `synthesized` | `paper` | `transformer`, `attention` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| `paper-1810-04805` | `screened` | `paper` | `encoder-transformer` | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| `paper-1910-10683` | `screened` | `paper` | `t5`, `encoder-decoder` | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) |
| `paper-1910-13461` | `screened` | `paper` | `bart`, `encoder-decoder` | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) |
| `paper-2010-11929` | `screened` | `paper` | `vision-transformer` | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) |
| `paper-2204-14198` | `screened` | `paper` | `vision-language`, `resampler` | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) |
| `paper-2301-12597` | `screened` | `paper` | `vision-language`, `projector` | [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) |
| `paper-2302-13971` | `screened` | `paper` | `decoder-transformer` | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) |
| `paper-2304-08485` | `screened` | `paper` | `llava`, `vision-language` | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) |
| `paper-2307-09288` | `screened` | `paper` | `decoder-transformer` | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) |
| `paper-2310-06825` | `screened` | `paper` | `decoder-transformer`, `sliding-window`, `gqa` | [Mistral 7B](https://arxiv.org/abs/2310.06825) |
| `paper-2401-04088` | `screened` | `paper` | `moe`, `decoder-transformer` | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) |
| `paper-2403-08295` | `screened` | `paper` | `decoder-transformer` | [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295) |
| `paper-2404-14219` | `screened` | `paper` | `decoder-transformer`, `on-device` | [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219) |
| `paper-2407-10671` | `screened` | `paper` | `decoder-transformer`, `moe` | [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) |
| `paper-2407-21783` | `screened` | `paper` | `decoder-transformer` | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) |
| `paper-2408-00118` | `screened` | `paper` | `decoder-transformer` | [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) |
| `paper-2405-04434` | `synthesized` | `paper` | `moe`, `mla`, `attention` | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) |
| `paper-2412-19437` | `screened` | `paper` | `moe`, `mtp`, `attention` | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) |
| `paper-2312-00752` | `synthesized` | `paper` | `ssm`, `recurrent` | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) |
| `paper-2405-21060` | `screened` | `paper` | `mamba2`, `ssm` | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| `paper-2403-19887` | `screened` | `paper` | `hybrid`, `ssm`, `attention`, `moe` | [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) |
| `paper-2305-13048` | `screened` | `paper` | `recurrent`, `language-model` | [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) |
| `paper-2404-07839` | `screened` | `paper` | `recurrent`, `griffin` | [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://arxiv.org/abs/2404.07839) |
| `paper-2402-19427` | `screened` | `paper` | `recurrent`, `hybrid` | [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427) |
| `paper-2307-08621` | `indexed` | `paper` | `recurrent`, `linear-attention` | [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) |
| `paper-2302-10866` | `indexed` | `paper` | `long-context`, `convolution` | [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) |
| `paper-2212-09748` | `screened` | `paper` | `dit`, `diffusion` | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) |
| `paper-2112-10752` | `screened` | `paper` | `latent-diffusion` | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) |
| `paper-2209-03003` | `screened` | `paper` | `rectified-flow` | [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) |
| `paper-2210-02747` | `screened` | `paper` | `flow-matching` | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) |
| `paper-2403-03206` | `screened` | `paper` | `stable-diffusion-3`, `flow`, `transformer` | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) |
| `paper-2409-12191` | `screened` | `paper` | `vision-language`, `video` | [Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) |
| `paper-2211-17192` | `synthesized` | `paper` | `speculative-decoding` | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) |
| `paper-2302-01318` | `synthesized` | `paper` | `speculative-decoding` | [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) |
| `paper-2305-09781` | `screened` | `paper` | `speculative-decoding`, `serving` | [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781) |
| `paper-2309-08168` | `screened` | `paper` | `self-speculative` | [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168) |
| `paper-2311-08252` | `screened` | `paper` | `retrieval`, `speculative-decoding` | [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252) |
| `paper-2401-07851` | `screened` | `paper` | `speculative-decoding`, `survey` | [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851) |
| `paper-2401-10774` | `synthesized` | `paper` | `medusa`, `speculative-decoding` | [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) |
| `paper-2401-15077` | `synthesized` | `paper` | `eagle`, `speculative-decoding` | [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) |
| `paper-2402-02057` | `screened` | `paper` | `lookahead-decoding` | [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) |
| `paper-2402-11131` | `screened` | `paper` | `multi-token`, `speculative-decoding` | [Speculative Streaming: Fast LLM Inference without Auxiliary Models](https://arxiv.org/abs/2402.11131) |
| `paper-2402-13720` | `screened` | `paper` | `speculative-decoding` | [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting](https://arxiv.org/abs/2402.13720) |
| `paper-2403-09919` | `screened` | `paper` | `redrafter`, `speculative-decoding`, `mlx` | [ReDrafter: Faster LLM Inference with Recurrent Drafter](https://arxiv.org/abs/2403.09919) |
| `paper-2404-19124` | `screened` | `paper` | `speculative-decoding`, `multi-token` | [Accelerating Production LLMs with Combined Token/Embedding Speculators](https://arxiv.org/abs/2404.19124) |
| `paper-2406-16858` | `screened` | `paper` | `eagle2`, `speculative-decoding` | [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) |
| `paper-2407-01955` | `screened` | `paper` | `speculative-decoding` | [S2D: Sorted Speculative Decoding for More Efficient Deployment of Nested Large Language Models](https://arxiv.org/abs/2407.01955) |
| `paper-2408-08696` | `screened` | `paper` | `token-recycling`, `speculative-decoding` | [Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling](https://arxiv.org/abs/2408.08696) |
| `paper-2409-00142` | `screened` | `paper` | `speculative-decoding` | [Dynamic Depth Decoding: Faster Speculative Decoding for LLMs](https://arxiv.org/abs/2409.00142) |
| `paper-2410-03804` | `screened` | `paper` | `speculative-decoding`, `attention` | [Mixture of Attentions for Speculative Decoding](https://arxiv.org/abs/2410.03804) |
| `paper-2412-12639` | `screened` | `paper` | `speculative-decoding` | [Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting and Custom-Designed Decoding Tree](https://arxiv.org/abs/2412.12639) |
| `paper-2505-07858` | `screened` | `paper` | `speculative-decoding`, `scaling-laws` | [Scaling Laws for Speculative Decoding](https://arxiv.org/abs/2505.07858) |
| `paper-2505-15380` | `synthesized` | `paper` | `speech`, `speculative-decoding` | [Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding](https://arxiv.org/abs/2505.15380) |
| `paper-2505-19201` | `screened` | `paper` | `multimodal`, `speculative-decoding` | [DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding](https://arxiv.org/abs/2505.19201) |
| `paper-2505-19645` | `screened` | `paper` | `moe`, `speculative-decoding` | [MoESD: Unveil Speculative Decoding’s Potential for Accelerating Sparse MoE](https://arxiv.org/abs/2505.19645) |
| `paper-2505-22179` | `synthesized` | `paper` | `speculative-decoding`, `quantization` | [Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design](https://arxiv.org/abs/2505.22179) |
| `paper-2505-24544` | `screened` | `paper` | `speculative-decoding`, `cross-attention` | [Cross-Attention Speculative Decoding](https://arxiv.org/abs/2505.24544) |
| `paper-2506-01979` | `screened` | `paper` | `speculative-decoding` | [Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism](https://arxiv.org/abs/2506.01979) |
| `paper-2509-11815` | `screened` | `paper` | `vlm`, `speculative-decoding` | [SpecVLM: Fast Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2509.11815) |
| `paper-2601-11580` | `synthesized` | `paper` | `speculative-decoding`, `serving`, `benchmark` | [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) |
| `paper-2601-19278` | `screened` | `paper` | `speculative-decoding`, `diffusion-drafter` | [DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference](https://arxiv.org/abs/2601.19278) |
| `paper-2602-06036` | `synthesized` | `paper` | `dflash`, `speculative-decoding` | [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) |
| `paper-2603-08899` | `screened` | `paper` | `speculative-decoding` | [ConFu: Contemplate the Future for Better Speculative Sampling](https://arxiv.org/abs/2603.08899) |
| `paper-2603-11053` | `screened` | `paper` | `speculative-decoding`, `scaling-laws` | [Speculative Decoding Scaling Laws: Throughput Optimization Made Simple](https://arxiv.org/abs/2603.11053) |
| `paper-2605-30852` | `indexed` | `paper` | `speculative-decoding`, `pipeline` | [Speculative Pipeline Decoding: Higher-Accuracy and Zero-Bubble Speculation via Pipeline Parallelism](https://arxiv.org/abs/2605.30852) |
| `paper-2206-01861` | `screened` | `paper` | `quantization` | [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861) |
| `paper-2208-07339` | `screened` | `paper` | `quantization`, `outliers` | [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) |
| `paper-2210-17323` | `screened` | `paper` | `weight-quantization` | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) |
| `paper-2211-10438` | `synthesized` | `paper` | `activation-quantization` | [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) |
| `paper-2305-14314` | `screened` | `paper` | `quantization`, `lora`, `training` | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) |
| `paper-2306-00978` | `screened` | `paper` | `weight-quantization` | [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) |
| `paper-2306-03078` | `screened` | `paper` | `weight-quantization`, `outliers` | [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078) |
| `paper-2306-07629` | `screened` | `paper` | `weight-quantization` | [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629) |
| `paper-2308-13137` | `screened` | `paper` | `quantization` | [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137) |
| `paper-2309-15531` | `screened` | `paper` | `weight-quantization`, `outliers` | [Rethinking Channel Dimensions to Isolate Outliers for Low-bit Weight Quantization of Large Language Models](https://arxiv.org/abs/2309.15531) |
| `paper-2310-16836` | `screened` | `paper` | `fp4`, `quantization` | [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836) |
| `paper-2312-03788` | `screened` | `paper` | `quantization` | [SmoothQuant+: Accurate and Efficient 4-bit Post-Training Weight Quantization for LLM](https://arxiv.org/abs/2312.03788) |
| `paper-2401-06118` | `synthesized` | `paper` | `quantization` | [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118) |
| `paper-2401-18079` | `synthesized` | `paper` | `kv-quantization` | [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079) |
| `paper-2402-02750` | `synthesized` | `paper` | `kv-quantization` | [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) |
| `paper-2402-04396` | `synthesized` | `paper` | `quantization`, `hadamard` | [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) |
| `paper-2402-17764` | `screened` | `paper` | `bitnet`, `quantization` | [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) |
| `paper-2404-00456` | `synthesized` | `paper` | `quantization`, `rotation`, `kv` | [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) |
| `paper-2405-04532` | `synthesized` | `paper` | `quantization`, `serving` | [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532) |
| `paper-2405-16406` | `synthesized` | `paper` | `quantization`, `rotation` | [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406) |
| `paper-2406-03482` | `screened` | `paper` | `kv-quantization` | [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) |
| `paper-2407-00088` | `screened` | `paper` | `low-bit`, `edge` | [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088) |
| `paper-2408-08554` | `screened` | `paper` | `quantization` | [ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for Large Language Models](https://arxiv.org/abs/2408.08554) |
| `paper-2410-07505` | `screened` | `paper` | `activation-quantization` | [CrossQuant: A Post-Training Quantization Method with Smaller Quantization Kernel for Precise Large Language Model Compression](https://arxiv.org/abs/2410.07505) |
| `paper-2410-12168` | `screened` | `paper` | `quantization`, `serving` | [COMET: Towards Practical W4A4KV4 LLMs Serving](https://arxiv.org/abs/2410.12168) |
| `paper-2411-04965` | `screened` | `paper` | `bitnet`, `activation-quantization` | [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965) |
| `paper-2412-14363` | `screened` | `paper` | `mixed-precision`, `quantization` | [ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals](https://arxiv.org/abs/2412.14363) |
| `paper-2501-16383` | `screened` | `paper` | `kv-quantization`, `rotation` | [RotateKV: Accurate and Robust 2-Bit KV Cache Quantization via Outlier-Aware Adaptive Rotations](https://arxiv.org/abs/2501.16383) |
| `paper-2502-04420` | `synthesized` | `paper` | `kv-quantization`, `mixed-precision` | [KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization](https://arxiv.org/abs/2502.04420) |
| `paper-2502-15075` | `screened` | `paper` | `kv-quantization`, `mixed-precision` | [More for Keys, Less for Values: Adaptive KV Cache Quantization](https://arxiv.org/abs/2502.15075) |
| `paper-2503-16257` | `screened` | `paper` | `vlm`, `kv-quantization` | [Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models](https://arxiv.org/abs/2503.16257) |
| `paper-2503-19950` | `screened` | `paper` | `kv-quantization` | [LogQuant: Log-Distributed 2-Bit Quantization of KV Cache](https://arxiv.org/abs/2503.19950) |
| `paper-2505-03745` | `screened` | `paper` | `long-context`, `quantization` | [AccLLM: Accelerating Long-Context LLM Inference via Algorithm-Hardware Co-Design](https://arxiv.org/abs/2505.03745) |
| `paper-2508-15601` | `screened` | `paper` | `mixed-precision`, `serving` | [Efficient Mixed-Precision Large Language Model Inference with TurboMind](https://arxiv.org/abs/2508.15601) |
| `paper-2604-04722` | `screened` | `paper` | `on-device`, `kv-quantization` | [Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs](https://arxiv.org/abs/2604.04722) |
| `paper-2602-23200` | `screened` | `paper` | `kv-quantization`, `hardware-aware` | [InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models](https://arxiv.org/abs/2602.23200) |
| `paper-2606-20474` | `screened` | `paper` | `kv-quantization`, `agent-serving` | [UltraQuant: 4-bit KV Caching for Context-Heavy Agents](https://arxiv.org/abs/2606.20474) |
| `paper-2606-21842` | `screened` | `paper` | `kv-cache`, `side-channel`, `rag` | [Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG](https://arxiv.org/abs/2606.21842) |
| `paper-2606-24033` | `screened` | `paper` | `kv-quantization`, `rope` | [RoPE-Aware Bit Allocation for KV-Cache Quantization](https://arxiv.org/abs/2606.24033) |
| `paper-2604-19157` | `screened` | `paper` | `kv-quantization`, `serving` | [SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving](https://arxiv.org/abs/2604.19157) |
| `paper-2605-17757` | `screened` | `paper` | `kv-quantization`, `rotation` | [OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization](https://arxiv.org/abs/2605.17757) |
| `paper-1911-02150` | `synthesized` | `paper` | `mqa`, `attention` | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| `paper-2205-14135` | `synthesized` | `paper` | `attention`, `kernel` | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) |
| `orca-osdi22` | `synthesized` | `paper` | `serving`, `continuous-batching` | [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) |
| `paper-2305-13245` | `synthesized` | `paper` | `gqa`, `attention` | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) |
| `paper-2305-17118` | `screened` | `paper` | `kv-pruning` | [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression](https://arxiv.org/abs/2305.17118) |
| `paper-2306-14048` | `screened` | `paper` | `kv-pruning`, `long-context` | [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) |
| `paper-2307-08691` | `synthesized` | `paper` | `attention`, `kernel` | [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) |
| `paper-2309-06180` | `synthesized` | `paper` | `paged-attention`, `serving` | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) |
| `paper-2309-17453` | `synthesized` | `paper` | `streamingllm`, `attention-sinks` | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) |
| `paper-2310-01889` | `screened` | `paper` | `long-context`, `distributed-attention` | [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) |
| `paper-2402-05099` | `synthesized` | `paper` | `prefix-sharing`, `serving` | [Hydragen: High-Throughput LLM Inference with Shared Prefixes](https://arxiv.org/abs/2402.05099) |
| `paper-2402-06082` | `screened` | `paper` | `kv-compression` | [SubGen: Token Generation in Sublinear Time and Memory](https://arxiv.org/abs/2402.06082) |
| `paper-2402-15220` | `synthesized` | `paper` | `prefix-cache`, `attention` | [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition](https://arxiv.org/abs/2402.15220) |
| `paper-2404-14469` | `screened` | `paper` | `kv-pruning` | [SnapKV: LLM Knows What You Are Looking for Before Generation](https://arxiv.org/abs/2404.14469) |
| `paper-2405-04437` | `screened` | `paper` | `serving`, `virtual-memory` | [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437) |
| `paper-2406-02069` | `screened` | `paper` | `kv-compression` | [PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069) |
| `paper-2407-08608` | `synthesized` | `paper` | `attention`, `kernel` | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision](https://arxiv.org/abs/2407.08608) |
| `paper-2407-17678` | `screened` | `paper` | `sparse-attention`, `context-sharding` | [Efficient LLM Training and Serving with Heterogeneous Context Sharding among Attention Heads](https://arxiv.org/abs/2407.17678) |
| `paper-2408-05646` | `screened` | `paper` | `low-rank`, `kv-compression` | [Eigen Attention: Attention in Low-Rank Space for KV Cache Compression](https://arxiv.org/abs/2408.05646) |
| `paper-2412-03594` | `synthesized` | `paper` | `batching`, `prefix-sharing` | [BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing](https://arxiv.org/abs/2412.03594) |
| `paper-2412-10319` | `synthesized` | `paper` | `benchmark`, `kv-cache`, `long-context` | [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/abs/2412.10319) |
| `paper-2501-01005` | `synthesized` | `paper` | `attention`, `serving`, `kernels` | [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) |
| `paper-2502-04077` | `screened` | `paper` | `kv-compression`, `attention` | [AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inference](https://arxiv.org/abs/2502.04077) |
| `paper-2503-14376` | `screened` | `paper` | `linear-attention`, `kernel` | [Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](https://arxiv.org/abs/2503.14376) |
| `paper-2505-22913` | `screened` | `paper` | `kv-pruning`, `sparse-attention` | [Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](https://arxiv.org/abs/2505.22913) |
| `paper-2508-02558` | `screened` | `paper` | `diffusion-lm`, `sparse-attention` | [Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction](https://arxiv.org/abs/2508.02558) |
| `paper-2508-18224` | `screened` | `paper` | `sparse-attention`, `kernel` | [Flash Sparse Attention: An Alternative Efficient Implementation of Native Sparse Attention Kernel](https://arxiv.org/abs/2508.18224) |
| `paper-2509-04377` | `screened` | `paper` | `kv-pruning`, `paged-cache` | [PagedEviction: Structured Block-wise KV Cache Pruning for Efficient LLM Inference](https://arxiv.org/abs/2509.04377) |
| `paper-2509-05165` | `screened` | `paper` | `kv-compression` | [KVCompose: Efficient Structured KV Cache Compression with Composite Tokens](https://arxiv.org/abs/2509.05165) |
| `paper-2510-00636` | `screened` | `paper` | `kv-compression` | [Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution](https://arxiv.org/abs/2510.00636) |
| `paper-2512-12087` | `screened` | `paper` | `sparse-attention` | [BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087) |
| `paper-2601-17702` | `screened` | `paper` | `retrieval-attention`, `long-context` | [S3-Attention: Attention-Aligned Endogenous Retrieval for Memory-Bounded Long-Context Inference](https://arxiv.org/abs/2601.17702) |
| `paper-2602-06072` | `screened` | `paper` | `attention`, `batching`, `kernel` | [PackInfer: Compute- and I/O-Efficient Attention for Batched LLM Inference](https://arxiv.org/abs/2602.06072) |
| `paper-2604-07815` | `screened` | `paper` | `sparse-attention`, `offload` | [AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention](https://arxiv.org/abs/2604.07815) |
| `paper-2605-18226` | `screened` | `paper` | `long-context`, `memory` | [Context Memorization for Efficient Long Context Generation](https://arxiv.org/abs/2605.18226) |
| `paper-2606-09079` | `screened` | `paper` | `long-context`, `sparse-attention` | [FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079) |
| `paper-1609-03499` | `indexed` | `paper` | `audio`, `autoregressive`, `vocoder` | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) |
| `paper-1703-10135` | `indexed` | `paper` | `tts` | [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) |
| `paper-1712-05884` | `indexed` | `paper` | `tacotron2`, `tts` | [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) |
| `paper-1802-08435` | `indexed` | `paper` | `wavernn`, `vocoder` | [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) |
| `paper-1810-11846` | `indexed` | `paper` | `vocoder`, `on-device` | [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://arxiv.org/abs/1810.11846) |
| `paper-1811-00002` | `indexed` | `paper` | `vocoder`, `flow` | [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002) |
| `paper-1905-09263` | `indexed` | `paper` | `tts`, `non-autoregressive` | [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) |
| `paper-1910-06711` | `indexed` | `paper` | `vocoder`, `gan` | [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) |
| `paper-1910-11480` | `indexed` | `paper` | `vocoder`, `gan` | [Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial Networks](https://arxiv.org/abs/1910.11480) |
| `paper-2002-02562` | `synthesized` | `paper` | `asr`, `streaming`, `rnnt` | [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562) |
| `paper-2005-08100` | `synthesized` | `paper` | `asr`, `conformer` | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) |
| `paper-2005-11129` | `indexed` | `paper` | `tts`, `flow` | [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129) |
| `paper-2006-04558` | `indexed` | `paper` | `tts`, `non-autoregressive` | [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) |
| `paper-2006-11477` | `screened` | `paper` | `asr`, `speech-encoder` | [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) |
| `paper-2009-00713` | `indexed` | `paper` | `vocoder`, `diffusion` | [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713) |
| `paper-2009-09761` | `indexed` | `paper` | `audio`, `diffusion`, `vocoder` | [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761) |
| `paper-2010-05646` | `synthesized` | `paper` | `vocoder`, `gan` | [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) |
| `paper-2105-06337` | `indexed` | `paper` | `tts`, `diffusion` | [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337) |
| `paper-2106-06103` | `screened` | `paper` | `vits`, `tts` | [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) |
| `paper-2106-07447` | `screened` | `paper` | `asr`, `speech-encoder` | [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) |
| `paper-2106-07889` | `indexed` | `paper` | `vocoder` | [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889) |
| `paper-2107-03312` | `synthesized` | `paper` | `audio-codec`, `rvq`, `streaming` | [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) |
| `paper-2110-13900` | `screened` | `paper` | `speech-encoder`, `asr` | [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) |
| `paper-2203-02395` | `screened` | `paper` | `vocoder`, `istft` | [iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform](https://arxiv.org/abs/2203.02395) |
| `paper-2205-04421` | `indexed` | `paper` | `tts` | [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/abs/2205.04421) |
| `paper-2206-04658` | `synthesized` | `paper` | `vocoder`, `gan` | [BigVGAN: A Universal Neural Vocoder with Large-Scale Training](https://arxiv.org/abs/2206.04658) |
| `paper-2206-08317` | `screened` | `paper` | `asr`, `non-autoregressive` | [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317) |
| `paper-2209-03143` | `synthesized` | `paper` | `audio-lm`, `codec-tokens` | [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143) |
| `paper-2209-15352` | `screened` | `paper` | `audio-generation`, `codec-tokens` | [AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352) |
| `paper-2210-13438` | `synthesized` | `paper` | `encodec`, `audio-codec` | [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) |
| `paper-2212-04356` | `synthesized` | `paper` | `whisper`, `asr` | [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) |
| `paper-2301-02111` | `synthesized` | `paper` | `valle`, `tts`, `audio-codec` | [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111) |
| `paper-2304-09116` | `screened` | `paper` | `tts`, `diffusion` | [NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers](https://arxiv.org/abs/2304.09116) |
| `paper-2305-09636` | `synthesized` | `paper` | `audio-generation`, `parallel-codebooks` | [SoundStorm: Efficient Parallel Audio Generation](https://arxiv.org/abs/2305.09636) |
| `paper-2306-00814` | `synthesized` | `paper` | `vocoder`, `fourier` | [Vocos: Closing the Gap between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis](https://arxiv.org/abs/2306.00814) |
| `paper-2306-05284` | `synthesized` | `paper` | `musicgen`, `audio-lm`, `delay-pattern` | [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) |
| `paper-2306-06546` | `synthesized` | `paper` | `dac`, `audio-codec` | [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546) |
| `paper-2306-15687` | `screened` | `paper` | `speech-generation`, `flow` | [Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale](https://arxiv.org/abs/2306.15687) |
| `paper-2308-11596` | `screened` | `paper` | `speech-to-speech`, `asr`, `translation` | [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596) |
| `paper-2308-16692` | `synthesized` | `paper` | `speech-tokenizer`, `audio-codec` | [SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models](https://arxiv.org/abs/2308.16692) |
| `paper-2309-03199` | `synthesized` | `paper` | `tts`, `flow-matching` | [Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching](https://arxiv.org/abs/2309.03199) |
| `paper-2309-07405` | `screened` | `paper` | `audio-codec`, `toolkit` | [FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec](https://arxiv.org/abs/2309.07405) |
| `paper-2309-15505` | `synthesized` | `paper` | `fsq`, `quantization`, `audio-codec` | [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505) |
| `paper-2310-00014` | `screened` | `paper` | `audio-codec`, `tts` | [Fewer-token Neural Speech Codec with Time-invariant Codes](https://arxiv.org/abs/2310.00014) |
| `paper-2310-13289` | `screened` | `paper` | `audio-language`, `multimodal` | [SALMONN: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/abs/2310.13289) |
| `paper-2311-00430` | `screened` | `paper` | `asr`, `distillation` | [Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430) |
| `paper-2312-05187` | `screened` | `paper` | `speech-to-speech`, `streaming` | [Seamless: Multilingual Expressive and Streaming Speech Translation](https://arxiv.org/abs/2312.05187) |
| `paper-2312-17279` | `synthesized` | `paper` | `asr`, `streaming`, `cache` | [Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition](https://arxiv.org/abs/2312.17279) |
| `paper-2401-04577` | `screened` | `paper` | `audio-generation`, `non-autoregressive` | [MAGNeT: Masked Audio Generation using a Single Non-Autoregressive Transformer](https://arxiv.org/abs/2401.04577) |
| `paper-2402-01831` | `screened` | `paper` | `audio-language`, `multimodal` | [Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities](https://arxiv.org/abs/2402.01831) |
| `paper-2402-13236` | `synthesized` | `paper` | `audio-language`, `survey`, `codec` | [Towards Audio Language Modeling: An Overview](https://arxiv.org/abs/2402.13236) |
| `paper-2404-02781` | `screened` | `paper` | `tts`, `codec-lm` | [CLaM-TTS: Improving Neural Codec Language Model for Zero-Shot Text-to-Speech](https://arxiv.org/abs/2404.02781) |
| `paper-2406-18009` | `synthesized` | `paper` | `tts`, `flow-matching` | [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.18009) |
| `paper-2407-10759` | `synthesized` | `paper` | `audio-language`, `multimodal` | [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759) |
| `paper-2408-16532` | `synthesized` | `paper` | `audio-codec`, `low-token-rate` | [WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](https://arxiv.org/abs/2408.16532) |
| `paper-2410-00037` | `synthesized` | `paper` | `speech-to-speech`, `streaming`, `mimi` | [Moshi: a Speech-Text Foundation Model for Real-Time Dialogue](https://arxiv.org/abs/2410.00037) |
| `paper-2410-06885` | `synthesized` | `paper` | `tts`, `flow-matching` | [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885) |
| `paper-2411-18803` | `synthesized` | `paper` | `audio-codec`, `streaming` | [TS3-Codec: Transformer-Based Simple Streaming Single Codec](https://arxiv.org/abs/2411.18803) |
| `paper-2411-19842` | `screened` | `paper` | `audio-codec`, `fsq` | [Scaling Transformers for Low-Bitrate High-Quality Speech Coding](https://arxiv.org/abs/2411.19842) |
| `paper-2503-01710` | `synthesized` | `paper` | `tts`, `codec-lm` | [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](https://arxiv.org/abs/2503.01710) |
| `paper-2503-20215` | `synthesized` | `paper` | `omni`, `speech`, `streaming` | [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) |
| `paper-2504-10344` | `screened` | `paper` | `audio-codec`, `semantic-tokens` | [ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling](https://arxiv.org/abs/2504.10344) |
| `paper-2504-12339` | `screened` | `paper` | `tts`, `multi-token` | [GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via a Dual-Branch Architecture](https://arxiv.org/abs/2504.12339) |
| `paper-2506-10274` | `synthesized` | `paper` | `audio-codec`, `benchmark`, `survey` | [Discrete Audio Tokens: More Than a Survey!](https://arxiv.org/abs/2506.10274) |
| `paper-2507-12197` | `screened` | `paper` | `tts`, `audio-codec`, `parallel-codebooks` | [Quantize More, Lose Less: Autoregressive Generation from Residually Quantized Speech Representations](https://arxiv.org/abs/2507.12197) |
| `paper-2507-18897` | `screened` | `paper` | `audio-codec` | [HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling](https://arxiv.org/abs/2507.18897) |
| `paper-2509-02020` | `screened` | `paper` | `tts`, `dialogue`, `streaming` | [FireRedTTS-2: Towards Long Conversational Speech Generation for Podcast and Chatbot](https://arxiv.org/abs/2509.02020) |
| `paper-2510-16841` | `screened` | `paper` | `audio-codec`, `semantic-acoustic` | [SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization](https://arxiv.org/abs/2510.16841) |
| `paper-2601-15621` | `synthesized` | `paper` | `tts`, `dual-track`, `streaming` | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| `paper-2601-20094` | `synthesized` | `paper` | `audio-codec`, `on-device`, `quantization` | [T-Mimi: A Transformer-based Mimi Decoder for Real-Time On-Phone TTS](https://arxiv.org/abs/2601.20094) |
| `paper-2601-23174` | `screened` | `paper` | `speech-tokenizer`, `variable-rate` | [Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization](https://arxiv.org/abs/2601.23174) |
| `paper-2602-04683` | `screened` | `paper` | `audio-language`, `audio-codec` | [UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization](https://arxiv.org/abs/2602.04683) |
| `paper-2602-10934` | `synthesized` | `paper` | `audio-codec`, `transformer` | [MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models](https://arxiv.org/abs/2602.10934) |
| `paper-2604-12438` | `screened` | `paper` | `tts`, `streaming`, `low-latency` | [An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture via Block-Wise Generation and Depth-Wise Codec Decoding](https://arxiv.org/abs/2604.12438) |
| `paper-2604-14493` | `screened` | `paper` | `asr`, `on-device`, `quantization` | [Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference](https://arxiv.org/abs/2604.14493) |
| `paper-2604-17852` | `screened` | `paper` | `audio-codec`, `multi-token`, `semantic` | [LLM-Codec: Neural Audio Codec Meets Language Model Objectives](https://arxiv.org/abs/2604.17852) |
| `repo-fish-speech` | `screened` | `repository` | `tts`, `audio-lm`, `codec` | [Fish Speech repository](https://github.com/fishaudio/fish-speech) |
| `repo-moshi` | `screened` | `repository` | `streaming-speech`, `codec`, `duplex` | [Moshi repository](https://github.com/kyutai-labs/moshi) |
| `repo-encodec` | `screened` | `repository` | `audio-codec` | [EnCodec repository](https://github.com/facebookresearch/encodec) |
| `repo-dac` | `screened` | `repository` | `audio-codec` | [Descript Audio Codec repository](https://github.com/descriptinc/descript-audio-codec) |
| `repo-bigvgan` | `screened` | `repository` | `vocoder` | [BigVGAN repository](https://github.com/NVIDIA/BigVGAN) |
| `repo-vocos` | `screened` | `repository` | `vocoder`, `fourier` | [Vocos repository](https://github.com/gemelo-ai/vocos) |
| `repo-f5tts` | `screened` | `repository` | `tts`, `flow-matching` | [F5-TTS repository](https://github.com/SWivid/F5-TTS) |
| `repo-whisper` | `screened` | `repository` | `asr`, `whisper` | [Whisper repository](https://github.com/openai/whisper) |
| `repo-moonshine` | `screened` | `repository` | `asr`, `on-device` | [Moonshine repository](https://github.com/usefulsensors/moonshine) |
| `repo-nemo` | `screened` | `repository` | `asr`, `tts`, `parakeet`, `canary` | [NVIDIA NeMo speech repository](https://github.com/NVIDIA/NeMo) |
| `repo-qwen-audio` | `screened` | `repository` | `audio-language` | [Qwen Audio repository](https://github.com/QwenLM/Qwen-Audio) |
| `repo-qwen-omni` | `screened` | `repository` | `omni`, `speech` | [Qwen2.5-Omni repository](https://github.com/QwenLM/Qwen2.5-Omni) |
| `repo-stable-audio-tools` | `indexed` | `repository` | `audio-generation`, `diffusion` | [Stable Audio Tools repository](https://github.com/Stability-AI/stable-audio-tools) |
| `repo-audiocraft` | `screened` | `repository` | `musicgen`, `audiogen`, `encodec` | [AudioCraft repository](https://github.com/facebookresearch/audiocraft) |
| `repo-funcodec` | `screened` | `repository` | `audio-codec` | [FunCodec repository](https://github.com/alibaba-damo-academy/FunCodec) |
| `paper-2106-09685` | `screened` | `paper` | `lora`, `training`, `parameter-efficient` | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) |
| `dropbox-hqq-repo` | `screened` | `repository` | `weight-quantization`, `hqq` | [Official HQQ implementation](https://github.com/dropbox/hqq) |
| `prompt-lookup-decoding-repo` | `screened` | `repository` | `prompt-lookup`, `speculative-decoding` | [Prompt Lookup Decoding repository](https://github.com/apoorvumang/prompt-lookup-decoding) |
| `paper-2601-19139` | `synthesized` | `paper` | `mlx`, `serving`, `continuous-batching`, `prefix-cache`, `vlm` | [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/abs/2601.19139) |
| `apple-mlx-m5-blog` | `screened` | `technical-blog` | `mlx`, `apple-silicon`, `llm`, `runtime` | [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) |
| `apple-wwdc25-mlx` | `screened` | `technical-blog` | `mlx`, `apple-silicon`, `lazy-evaluation`, `compile` | [Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/) |
| `vllm-blog-anatomy` | `screened` | `technical-blog` | `serving`, `paged-attention`, `continuous-batching`, `prefix-cache`, `speculative-decoding` | [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) |
| `vllm-doc-prefix-caching` | `screened` | `official-doc` | `prefix-cache`, `serving`, `kv-cache` | [vLLM automatic prefix caching documentation](https://docs.vllm.ai/en/stable/design/prefix_caching/) |
| `sankalp-prompt-cache-blog` | `screened` | `technical-blog` | `prompt-cache`, `paged-attention`, `serving` | [How prompt caching works - Paged Attention and Automatic Prefix Caching](https://sankalp.bearblog.dev/how-prompt-caching-works/) |
| `nvidia-nvfp4-blog` | `screened` | `technical-blog` | `nvfp4`, `quantization`, `low-bit-formats` | [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) |
| `spheron-nvfp4-mxfp4-guide` | `screened` | `technical-blog` | `nvfp4`, `mxfp4`, `quantization`, `low-bit-formats` | [NVFP4 vs MXFP4: 4-Bit Quantization Format Decision Guide](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/) |
| `mlx-doc-cpp-ops` | `synthesized` | `official-doc` | `mlx`, `cpp`, `kernels`, `segmented-mm` | [MLX C++ operations reference](https://ml-explore.github.io/mlx/build/html/cpp/ops.html) |
| `mlx-doc-streams` | `synthesized` | `official-doc` | `mlx`, `streams`, `runtime` | [MLX using streams documentation](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html) |
| `mlx-doc-async-eval` | `synthesized` | `official-doc` | `mlx`, `async-eval`, `runtime` | [MLX async_eval API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html) |
| `mlx-doc-fast-metal-kernel` | `synthesized` | `official-doc` | `mlx`, `metal`, `custom-kernels` | [MLX fast Metal kernel API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html) |
| `mlx-doc-custom-function` | `synthesized` | `official-doc` | `mlx`, `custom-function`, `autograd` | [MLX custom_function API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html) |
| `mlx-lm-switch-layers` | `synthesized` | `source-code` | `mlx-lm`, `moe`, `gather-mm`, `gather-qmm` | [MLX-LM Switch/MoE layers implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py) |
| `mlx-lm-issue-956` | `synthesized` | `issue-report` | `mlx-lm`, `moe`, `fusion`, `quantization` | [MLX-LM MoE gate/up fusion proposal](https://github.com/ml-explore/mlx-lm/issues/956) |
| `paper-2211-15841` | `synthesized` | `paper` | `moe`, `block-sparse`, `training` | [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) |
| `mlx-audio-docs` | `synthesized` | `official-doc` | `mlx-audio`, `audio`, `tts`, `stt` | [MLX-Audio documentation](https://blaizzy.github.io/mlx-audio/) |
| `mlx-audio-streaming-guide` | `synthesized` | `official-doc` | `mlx-audio`, `streaming`, `tts`, `stt` | [MLX-Audio streaming audio guide](https://blaizzy.github.io/mlx-audio/guides/streaming/) |
| `mlx-audio-quantization-guide` | `synthesized` | `official-doc` | `mlx-audio`, `quantization`, `tts`, `stt` | [MLX-Audio quantization guide](https://blaizzy.github.io/mlx-audio/guides/quantization/) |
| `mlx-audio-qwen3-tts-docs` | `synthesized` | `official-doc` | `mlx-audio`, `qwen3-tts`, `tts`, `batching`, `streaming` | [MLX-Audio Qwen3-TTS model guide](https://blaizzy.github.io/mlx-audio/models/tts/qwen3-tts/) |
| `mlx-audio-moss-tts-docs` | `synthesized` | `official-doc` | `mlx-audio`, `moss-tts`, `tts`, `streaming` | [MLX-Audio MOSS-TTS model guide](https://blaizzy.github.io/mlx-audio/models/tts/moss-tts/) |
| `mlx-audio-voxtral-realtime-docs` | `synthesized` | `official-doc` | `mlx-audio`, `voxtral`, `stt`, `streaming` | [MLX-Audio Voxtral Realtime STT guide](https://blaizzy.github.io/mlx-audio/models/stt/voxtral-realtime/) |
| `hf-moss-tts-model-card` | `synthesized` | `repository` | `moss-tts`, `model-card`, `streaming`, `tts` | [MOSS-TTS MLX community model card](https://huggingface.co/mlx-community/MOSS-TTS-8B-8bit) |
| `vllm-mlx-continuous-batching-guide` | `synthesized` | `source-code` | `vllm-mlx`, `continuous-batching`, `serving` | [vLLM-MLX continuous batching guide](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/continuous-batching.md) |
| `vllm-mlx-multimodal-guide` | `synthesized` | `source-code` | `vllm-mlx`, `multimodal`, `video`, `serving` | [vLLM-MLX multimodal serving guide](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/multimodal.md) |
| `mlx-vlm-vision-cache-source` | `synthesized` | `source-code` | `mlx-vlm`, `vision-cache`, `multimodal`, `cache` | [MLX-VLM vision feature cache implementation](https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/vision_cache.py) |
| `mlx-vlm-apc-source` | `synthesized` | `source-code` | `mlx-vlm`, `automatic-prefix-cache`, `multimodal`, `cache` | [MLX-VLM automatic prefix cache implementation](https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/apc.py) |
| `mlx-vlm-ar-cache-guards` | `synthesized` | `source-code` | `mlx-vlm`, `prefix-cache`, `media-hash`, `cache-safety` | [MLX-VLM generation cache media guards](https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/generate/ar.py) |
| `vllm-mlx-vision-cache-source` | `synthesized` | `source-code` | `vllm-mlx`, `vision-cache`, `multimodal`, `serving` | [vLLM-MLX vision embedding cache implementation](https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/vllm_mlx/vision_embedding_cache.py) |
| `vllm-mlx-mllm-batch-source` | `synthesized` | `source-code` | `vllm-mlx`, `multimodal`, `batching`, `prefix-cache` | [vLLM-MLX multimodal batch generator implementation](https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/vllm_mlx/mllm_batch_generator.py) |
| `vllm-mlx-memory-cache-source` | `synthesized` | `source-code` | `vllm-mlx`, `prefix-cache`, `memory`, `serving` | [vLLM-MLX memory-aware prefix cache implementation](https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/vllm_mlx/memory_cache.py) |
| `paper-2502-13923` | `synthesized` | `paper` | `vision-language`, `video`, `qwen`, `mrope` | [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) |
| `paper-2412-13303` | `synthesized` | `paper` | `vision-language`, `visual-token-reduction`, `encoder` | [FastVLM: Efficient Vision Encoding for Vision Language Models](https://arxiv.org/abs/2412.13303) |
| `apple-fastvlm-blog` | `synthesized` | `technical-blog` | `vision-language`, `fastvlm`, `visual-token-reduction` | [FastVLM Apple Machine Learning Research blog](https://machinelearning.apple.com/research/fast-vision-language-models) |
| `paper-2412-04467` | `synthesized` | `paper` | `vision-language`, `visual-token-pruning` | [VisionZip: Longer is Better but Not Necessary in Vision Language Models](https://arxiv.org/abs/2412.04467) |
| `paper-2403-15388` | `synthesized` | `paper` | `vision-language`, `visual-token-pruning`, `llava` | [LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](https://arxiv.org/abs/2403.15388) |
| `llava-onevision-blog` | `synthesized` | `technical-blog` | `vision-language`, `video`, `llava` | [LLaVA-OneVision release blog](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) |
