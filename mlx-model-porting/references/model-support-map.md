# Model support and reference map

This file guides source selection; it is not a promise that every upstream checkpoint works unchanged.

| Family | First MLX reference | Secondary reference | Primary runbook |
|---|---|---|---|
| Dense decoder LM | MLX-LM model implementations and generation/cache code | MLX-VLM language backbones | `runbook-decoder-transformer.md` |
| Sparse MoE LM | MLX-LM MoE implementations | serving projects with expert policies | `runbook-moe-transformer.md` |
| Encoder LM | MLX examples / embedding projects | standalone ports | `runbook-encoder-transformer.md` |
| Non-generative CV backbone/classifier | MLX CIFAR/ResNet examples | ConvNeXt/EfficientNet/Swin sources and HF image-classification/feature-extraction cards | `runbook-non-generative-cv.md` |
| Graph message passing | MLX GCN example | PyG/DGL-style GCN, GAT, GraphSAGE, GIN, and MPNN sources | `runbook-graph-message-passing.md` |
| Encoder-decoder / Whisper | MLX Whisper example, MLX-Audio STT | MLX-VLM audio paths | `runbook-encoder-decoder.md`, `runbook-asr.md` |
| Mamba/SSM/hybrid | MLX-LM supported hybrid models | standalone MLX ports | `runbook-ssm-hybrid.md` |
| Vision-language / omni | MLX-VLM | MLX-Audio for speech branches | `runbook-multimodal-omni.md` |
| Diffusion / flow | MLX examples and active image/video ports | model-specific repositories | `runbook-diffusion-flow.md` |
| Neural audio codec | MLX-Audio codec implementations; MLX EnCodec example | original codec repository | `runbook-audio-codec.md` |
| Autoregressive audio/TTS | MLX-Audio TTS implementations | MLX-LM backbones | `runbook-autoregressive-audio.md` |
| Flow/diffusion TTS | MLX-Audio model implementations | original architecture repository | `runbook-flow-tts.md` |
| Vocoder | MLX-Audio codec/vocoder implementations | MLX examples and original repo | `runbook-vocoder.md` |
| ASR | MLX-Audio STT implementations | MLX Whisper example | `runbook-asr.md` |
| Streaming speech | MLX-Audio realtime/STT/STS implementations | original streaming model repo | `runbook-streaming-speech.md` |
| Separation/enhancement | MLX-Audio STS models | original model repository | `runbook-separation-enhancement.md` |

Before copying a pattern, inspect its revision, tests, config schema, and weight conversion path. Similar class names do not imply identical cache or tensor semantics.

## Candidate tracks, not supported families yet

The live multi-source research run on 2026-06-27 found strong demand and
reference patterns outside the declared families. These remaining tracks are not
supported until they gain architecture registry entries, runbooks or reference
sections, synthetic fixtures, golden scenarios, seeded parity failures, and
task-specific validation gates.

| Candidate track | Evidence seed | First validation gate |
|---|---|---|
| Dense and promptable vision | SAM example, DETR/Mask R-CNN/DeepLab/SAM/Depth Anything/HRNet/OCR sources, HF detection/segmentation/depth task pages | fixed-image logits, boxes, masks, prompts, depth maps, keypoints, OCR logits/decoded strings, and postprocessed outputs with IoU/AP/Dice/AbsRel/RMSE/OKS/edit-distance gates |
| Structured, tabular, and forecasting | HF tabular task pages, Chronos/TimesFM/Time Series Transformer metadata | scaler/normalizer parity, lag and context construction, observed masks, known-future covariate leakage checks, fixed forecast tensor/quantile parity |
| Ranking and recommender subfamilies | HF text-ranking task page, cross-encoder and recommender search samples | pair-score parity, top-k ordering stability, NDCG/AUC/top-k task gates, and a taxonomy split between embeddings, cross-encoders, classifiers, graph recommenders, and generative recommenders |
| Point-cloud, equivariant, and scientific ML | PointNet, EGNN/e3nn, Open Catalyst, protein, and chemistry sources | neighbor-list fixtures, ragged batching, rotation/reflection equivariance, units/symmetry constraints, and energy/force or domain-metric gates |
| Training and fine-tuning as a port target | MLX autograd/module/optimizer/compile docs, MLX-LM and examples LoRA trainers | loss/gradient parity, tiny overfit, optimizer-state round trip, exact checkpoint resume, train/eval mode checks, adapter merge/fuse parity, and memory graph-retention checks |

Candidate evidence from Hugging Face metadata must be classified by loader
family, required source package, remote-code risk, fixture type, parity metric,
and rollback condition before it influences implementation guidance.
