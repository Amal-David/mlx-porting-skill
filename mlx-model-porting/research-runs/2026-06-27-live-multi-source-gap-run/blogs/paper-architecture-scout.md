# Paper Architecture Scout Research Blog

## Assignment
Find architecture-specific algorithm constraints and evaluation methods for model families the skill supports or should support.

## Planned sampling
- papers: arXiv MLX framework search [paper-search] - https://arxiv.org/search/?query=MLX+framework&searchtype=all
- papers: arXiv on-device transformer search [paper-search] - https://arxiv.org/search/?query=on-device+transformer+inference&searchtype=all
- papers: Papers with Code MLX search [paper-index] - https://paperswithcode.com/search?q=MLX
- technical_blogs: Apple Machine Learning Research [technical-blog-index] - https://machinelearning.apple.com/
- technical_blogs: MLX examples discussions and notes [maintainer-notes] - https://ml-explore.github.io/mlx-examples/

## Sources sampled
- Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385, accessed 2026-06-27)
- A ConvNet for the 2020s (https://arxiv.org/abs/2201.03545, accessed 2026-06-27)
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (https://arxiv.org/abs/1905.11946, accessed 2026-06-27)
- mlx.nn.Conv2d documentation (https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html, accessed 2026-06-27)
- Mask R-CNN (https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf, accessed 2026-06-27)
- End-to-End Object Detection with Transformers (https://arxiv.org/abs/2005.12872, accessed 2026-06-27)
- Segment Anything (https://arxiv.org/abs/2304.02643, accessed 2026-06-27)
- Depth Anything V2 (https://arxiv.org/abs/2406.09414, accessed 2026-06-27)
- PP-OCR: A Practical Ultra Lightweight OCR System (https://arxiv.org/abs/2009.09941, accessed 2026-06-27)
- Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907, accessed 2026-06-27)
- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (https://arxiv.org/abs/1612.00593, accessed 2026-06-27)
- E(n) Equivariant Graph Neural Networks (https://arxiv.org/abs/2102.09844, accessed 2026-06-27)
- e3nn: Euclidean Neural Networks (https://arxiv.org/abs/2207.09453, accessed 2026-06-27)
- Open Catalyst Challenge (https://opencatalystproject.org/challenge.html, accessed 2026-06-27)

## Candidate findings
- cv-backbones-convnet-family: Modern ConvNet backbones should become a first-class validation track [needs-validation] - ResNet, ConvNeXt, and EfficientNet cover core non-generative image backbones with residual, depthwise, large-kernel, normalization, activation, pooling, and classifier-head semantics that the current generative/audio map does not exercise.
- dense-prediction-and-promptable-vision-gates: Detection, segmentation, OCR, depth, pose, and SAM-like masks need family-specific gates [needs-validation] - Mask R-CNN, DETR, DeepLabv3+, U-Net, CRNN/PP-OCR, SAM/SAM2, Depth Anything, and HRNet introduce postprocessing, prompt, dense map, coordinate, sequence-decoding, and task-metric gates that generic encoder or VLM coverage cannot prove.
- gnn-point-cloud-equivariant-science-gates: Graph, point-cloud, and scientific ports need sparse and symmetry tests [needs-validation] - GNNs, point clouds, equivariant molecular models, and protein/chemistry systems require message passing, neighbor search, scatter/segment reductions, ragged batching, permutation invariance, rotation/reflection equivariance, and domain metrics beyond ordinary tensor allclose.

## Decision notes
- Prioritize coverage by operator semantics, not model popularity.
- Implementation feasibility in MLX remains needs-validation unless a reproducible MLX path is proven.
- Validation gates should combine deterministic parity fixtures with task-level metrics.

## Open validation
- cv-backbones-convnet-family: Create synthetic parity fixtures for Conv2d, depthwise Conv2d, BatchNorm/LayerNorm, pooling, and residual add.
- dense-prediction-and-promptable-vision-gates: Start with one semantic segmentation fixture and one prompt-to-mask fixture before detection and OCR expansion.
- gnn-point-cloud-equivariant-science-gates: Inventory MLX scatter, segment, sparse, indexing, spherical-harmonic, tensor-product, and neighbor-list needs for one GCN-like and one EGNN-like target.
