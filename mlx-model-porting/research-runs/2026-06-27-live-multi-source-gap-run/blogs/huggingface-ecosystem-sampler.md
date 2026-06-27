# Hugging Face Ecosystem Sampler Research Blog

## Assignment
Sample model cards, library tags, downloads, tasks, and license metadata to identify porting demand and artifact shapes.

## Planned sampling
- hugging_face: Hugging Face MLX model search [model-index] - https://huggingface.co/models?search=mlx
- hugging_face: mlx-community organization [model-index] - https://huggingface.co/mlx-community
- hugging_face: Hugging Face Transformers MLX search [docs-search] - https://huggingface.co/docs/transformers/search?query=mlx

## Sources sampled
- Hugging Face Tasks: Tabular Classification (https://huggingface.co/tasks/tabular-classification, accessed 2026-06-27)
- Hugging Face Tasks: Tabular Regression (https://huggingface.co/tasks/tabular-regression, accessed 2026-06-27)
- Transformers docs: Time Series Transformer (https://huggingface.co/docs/transformers/model_doc/time_series_transformer, accessed 2026-06-27)
- amazon/chronos-2 model card (https://huggingface.co/amazon/chronos-2, accessed 2026-06-27)
- Hugging Face Tasks: Text Ranking (https://huggingface.co/tasks/text-ranking, accessed 2026-06-27)
- cross-encoder/ms-marco-MiniLM-L6-v2 model card (https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2, accessed 2026-06-27)
- Hugging Face Models API: search=recommender (https://huggingface.co/api/models?search=recommender&sort=downloads&direction=-1&limit=20, accessed 2026-06-27)
- Hugging Face Tasks: Object Detection (https://huggingface.co/tasks/object-detection, accessed 2026-06-27)
- Hugging Face Tasks: Image Segmentation (https://huggingface.co/tasks/image-segmentation, accessed 2026-06-27)
- Hugging Face Tasks: Depth Estimation (https://huggingface.co/tasks/depth-estimation, accessed 2026-06-27)
- facebook/dinov2-small model card (https://huggingface.co/facebook/dinov2-small, accessed 2026-06-27)
- Hugging Face Models API: text-ranking sorted by downloads (https://huggingface.co/api/models?filter=text-ranking&sort=downloads&direction=-1&limit=12, accessed 2026-06-27)
- Hugging Face Models API: object-detection sorted by downloads (https://huggingface.co/api/models?filter=object-detection&sort=downloads&direction=-1&limit=10, accessed 2026-06-27)
- Hugging Face Models API: tabular-regression sorted by downloads (https://huggingface.co/api/models?filter=tabular-regression&sort=downloads&direction=-1&limit=10, accessed 2026-06-27)

## Candidate findings
- hf-tabular-time-series-artifact-demand: Tabular and time-series demand exists, but artifact shapes diverge [needs-validation] - HF exposes tabular task pages and time-series model/API demand, while sampled TabPFN, Nori, Chronos, TimesFM, and Time Series Transformer sources show heterogeneous checkpoint, config, loader, and output semantics.
- hf-ranking-recsys-split: Ranking is a clearer near-term target than generic recommender support [needs-validation] - HF text-ranking has mature task pages and cross-encoder artifacts, while recommender searches are noisy and mix embedding retrieval, reranking, classifiers, Keras, graph, and generative chat recommenders.
- hf-non-generative-cv-artifact-families: HF CV task pages identify concrete early validation targets [needs-validation] - Object detection, image segmentation, depth estimation, and image feature extraction have strong HF task and model-card surfaces, with candidates such as Table Transformer/DETR, SegFormer, Depth Anything V2, and DINOv2 exposing distinct preprocessing and postprocessing contracts.
- hf-library-tag-taxonomy-needed: Library tags and sibling files should drive candidate taxonomy before implementation [adopted] - Sampled HF categories map to transformers, sentence-transformers, timm, ultralytics, tabpfn, skops, keras, LightGBM, ONNX Runtime, diffusers, and custom-code repos; a single task tag can hide multiple incompatible execution paths.

## Decision notes
- Hugging Face task pages, API filters, model-card metadata, and sibling files are demand and artifact-shape evidence only.
- Broad product labels must be split into loader families before implementation work.
- Recommender evidence currently points more to embedding retrieval and reranking than one canonical recommender-task implementation.

## Open validation
- hf-tabular-time-series-artifact-demand: Probe one sklearn/skops baseline, one deep tabular foundation repo, and one Chronos or TimesFM target with fixed fixture inputs.
- hf-ranking-recsys-split: Start with cross-encoder/ms-marco-MiniLM-L6-v2 and compare CrossEncoder.predict scores and rank order.
- hf-non-generative-cv-artifact-families: Pin DINOv2-small, SegFormer-B0, Depth-Anything-V2-Small, and one DETR/Table Transformer target for separate fixture design.
- hf-library-tag-taxonomy-needed: Sample at least five repos per target lane and manually verify one representative per loader family.
