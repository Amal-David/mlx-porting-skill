# Research Loop 2026-06-27-live-multi-source-gap-run

Objective: Live multi-source gap research for comprehensive MLX porting coverage

Review only: True
Findings: 13
Non-GitHub lanes covered: hugging_face, official_docs, papers, repo_local_audit
Planned non-GitHub sample targets: 20

## Decision Counts
- adopted: 6
- held: 0
- rejected: 0
- needs-validation: 7

## Adopted
- official-autograd-value-and-grad-contract: Autograd gives a concrete training-loop contract - MLX exposes scalar-loss gradient APIs for arbitrary callables and Module trainable parameters, so training guidance can require loss and gradient-tree parity without claiming a trained port works.
- official-training-checkpoint-resume-contract: Training checkpoints need model, optimizer, random, scheduler, and data state - MLX Module and optimizer docs support serializing weights and optimizer state, but optimizer state alone is not sufficient for exact resume because configuration, scheduler, random state, and data cursor live outside a plain state tree.
- official-compiled-training-state-gate: Compiled training requires explicit state capture - MLX compile guidance and official examples show compiled training must capture model state, optimizer state, and random state for stochastic modules; eager parity should precede compiled training claims.
- official-training-memory-and-lora-gates: Memory APIs and official LoRA paths define conservative training gates - MLX memory counters, memory limits, gradient checkpointing, and MLX-LM LoRA code provide reference patterns for training-memory and adapter validation, but not a general non-LLM training claim.
- hf-library-tag-taxonomy-needed: Library tags and sibling files should drive candidate taxonomy before implementation - Sampled HF categories map to transformers, sentence-transformers, timm, ultralytics, tabpfn, skops, keras, LightGBM, ONNX Runtime, diffusers, and custom-code repos; a single task tag can hide multiple incompatible execution paths.
- coverage-map-beyond-14-guardrail: Comprehensiveness claims must stay bounded until new families are validated - The repo validates 14 declared families, while non-generative CV, structured/time-series/recsys, graph/scientific, and training remain backlog-level tracks that need architecture entries, fixtures, and gates before support language changes.

## Held
- None

## Rejected
- None

## Needs Validation
- cv-backbones-convnet-family: Modern ConvNet backbones should become a first-class validation track - ResNet, ConvNeXt, and EfficientNet cover core non-generative image backbones with residual, depthwise, large-kernel, normalization, activation, pooling, and classifier-head semantics that the current generative/audio map does not exercise.
- dense-prediction-and-promptable-vision-gates: Detection, segmentation, OCR, depth, pose, and SAM-like masks need family-specific gates - Mask R-CNN, DETR, DeepLabv3+, U-Net, CRNN/PP-OCR, SAM/SAM2, Depth Anything, and HRNet introduce postprocessing, prompt, dense map, coordinate, sequence-decoding, and task-metric gates that generic encoder or VLM coverage cannot prove.
- gnn-point-cloud-equivariant-science-gates: Graph, point-cloud, and scientific ports need sparse and symmetry tests - GNNs, point clouds, equivariant molecular models, and protein/chemistry systems require message passing, neighbor search, scatter/segment reductions, ragged batching, permutation invariance, rotation/reflection equivariance, and domain metrics beyond ordinary tensor allclose.
- hf-tabular-time-series-artifact-demand: Tabular and time-series demand exists, but artifact shapes diverge - HF exposes tabular task pages and time-series model/API demand, while sampled TabPFN, Nori, Chronos, TimesFM, and Time Series Transformer sources show heterogeneous checkpoint, config, loader, and output semantics.
- hf-ranking-recsys-split: Ranking is a clearer near-term target than generic recommender support - HF text-ranking has mature task pages and cross-encoder artifacts, while recommender searches are noisy and mix embedding retrieval, reranking, classifiers, Keras, graph, and generative chat recommenders.
- hf-non-generative-cv-artifact-families: HF CV task pages identify concrete early validation targets - Object detection, image segmentation, depth estimation, and image feature extraction have strong HF task and model-card surfaces, with candidates such as Table Transformer/DETR, SegFormer, Depth Anything V2, and DINOv2 exposing distinct preprocessing and postprocessing contracts.
- official-examples-are-seeds-not-support: Official MLX examples seed future tracks but do not prove broad support - MLX examples for CIFAR/ResNet, Segment Anything, GCN, and LoRA are valuable starting points for non-generative CV, graph/scientific, and training tracks, but each needs a porting-specific fixture and task-quality gate before the skill can claim support.

## Limitations
- Findings came from parallel subagent research and were normalized into the research-loop schema before promotion.
- No remote model code was executed and no real MLX-vs-source parity run was performed in this research slice.
- Unsupported evidence remains needs-validation or held; the skill must not claim support for new families until fixtures and gates exist.
