# Coverage Skeptic Research Blog

## Assignment
Look for blind spots, unsupported architecture families, missing validation gates, and overclaimed optimizations.

## Planned sampling
- official_docs: MLX documentation index [official-doc] - https://ml-explore.github.io/mlx/build/html/index.html
- official_docs: MLX custom Metal kernels [official-doc] - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- official_docs: Apple Metal developer documentation [official-doc] - https://developer.apple.com/metal/
- papers: arXiv MLX framework search [paper-search] - https://arxiv.org/search/?query=MLX+framework&searchtype=all
- papers: arXiv on-device transformer search [paper-search] - https://arxiv.org/search/?query=on-device+transformer+inference&searchtype=all
- papers: Papers with Code MLX search [paper-index] - https://paperswithcode.com/search?q=MLX
- hugging_face: Hugging Face MLX model search [model-index] - https://huggingface.co/models?search=mlx
- hugging_face: mlx-community organization [model-index] - https://huggingface.co/mlx-community
- hugging_face: Hugging Face Transformers MLX search [docs-search] - https://huggingface.co/docs/transformers/search?query=mlx
- packages: PyPI mlx metadata [package-metadata] - https://pypi.org/pypi/mlx/json
- packages: PyPI mlx-lm metadata [package-metadata] - https://pypi.org/pypi/mlx-lm/json
- packages: PyPI mlx-vlm metadata [package-metadata] - https://pypi.org/pypi/mlx-vlm/json
- packages: PyPI mlx-audio metadata [package-metadata] - https://pypi.org/pypi/mlx-audio/json
- technical_blogs: Apple Machine Learning Research [technical-blog-index] - https://machinelearning.apple.com/
- technical_blogs: MLX examples discussions and notes [maintainer-notes] - https://ml-explore.github.io/mlx-examples/
- community_discussions: Apple Developer Forums MLX search [forum-search] - https://developer.apple.com/forums/search/?q=MLX
- community_discussions: LocalLLaMA MLX search [community-search] - https://www.reddit.com/r/LocalLLaMA/search/?q=MLX&restrict_sr=1
- repositories: ml-explore/mlx [repository] - https://github.com/ml-explore/mlx
- repositories: ml-explore/mlx-lm [repository] - https://github.com/ml-explore/mlx-lm
- repositories: ml-explore/mlx-examples [repository] - https://github.com/ml-explore/mlx-examples
- repo_local_audit: Research backlog [local-file] - assets/research_backlog.json
- repo_local_audit: Validation contract [local-file] - ../VALIDATION.md
- repo_local_audit: Tooling tests [local-file] - ../tests/test_tooling.py

## Sources sampled
- Local architectures registry (file:///Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/architectures.yaml, accessed 2026-06-27)
- Local golden scenario harness (file:///Users/amal/Downloads/mlx-porting-skill/tests/test_scenarios.py, accessed 2026-06-27)
- Local research backlog (file:///Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/research_backlog.json, accessed 2026-06-27)
- MLX examples: CIFAR and ResNets (https://github.com/ml-explore/mlx-examples/tree/main/cifar, accessed 2026-06-27)
- MLX examples: Segment Anything (https://github.com/ml-explore/mlx-examples/tree/main/segment_anything, accessed 2026-06-27)
- MLX examples: Graph Convolutional Network (https://github.com/ml-explore/mlx-examples/tree/main/gcn, accessed 2026-06-27)
- MLX examples: Fine-Tuning with LoRA or QLoRA (https://github.com/ml-explore/mlx-examples/tree/main/lora, accessed 2026-06-27)

## Candidate findings
- coverage-map-beyond-14-guardrail: Comprehensiveness claims must stay bounded until new families are validated [adopted] - The repo validates 14 declared families, while non-generative CV, structured/time-series/recsys, graph/scientific, and training remain backlog-level tracks that need architecture entries, fixtures, and gates before support language changes.
- official-examples-are-seeds-not-support: Official MLX examples seed future tracks but do not prove broad support [needs-validation] - MLX examples for CIFAR/ResNet, Segment Anything, GCN, and LoRA are valuable starting points for non-generative CV, graph/scientific, and training tracks, but each needs a porting-specific fixture and task-quality gate before the skill can claim support.

## Decision notes
- Local full coverage currently means full coverage of declared families, not every useful MLX porting domain.
- Official MLX examples show useful reference patterns for CIFAR/ResNet, SAM, GCN, and LoRA, but examples are not support claims.
- The four gap areas should become actionable backlog/reference updates until they have routing, fixtures, parity gates, task-quality gates, and rollback conditions.

## Open validation
- coverage-map-beyond-14-guardrail: After any family expansion, run unit discovery, skill audit, source validation, manifest check, and targeted scenario tests.
- official-examples-are-seeds-not-support: Create scoped tasks for ResNet/CIFAR, SAM mask generation, GCN message passing, and LoRA tiny-overfit fixtures.
