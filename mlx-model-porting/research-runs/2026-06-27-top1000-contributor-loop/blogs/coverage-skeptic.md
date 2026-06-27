# Coverage Skeptic Research Blog

## Assignment
Look for blind spots, unsupported architecture families, missing validation gates, and overclaimed optimizations.

## Sources sampled
- GitHub REST search API (https://docs.github.com/en/rest/search/search, accessed 2026-06-27)
- Top-1000 contributor learnings artifact (file:///Users/amal/Downloads/mlx-porting-skill/mlx-model-porting/assets/contributor_learnings.json, accessed 2026-06-27)
- TorchVision operators documentation (https://docs.pytorch.org/vision/0.9/ops.html, accessed 2026-06-27)
- PyTorch Geometric MessagePassing tutorial (https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/create_gnn.html, accessed 2026-06-27)
- Hugging Face Time Series Transformer docs (https://huggingface.co/docs/transformers/en/model_doc/time_series_transformer, accessed 2026-06-27)

## Candidate findings
- top1000-long-tail-rescreening-needed: Top-1000 contributor research needs repeatable long-tail rescreening [needs-validation] - The sweep found 71 repository-search matches and retained earlier code-search results, but GitHub code search rate limits interrupted several code queries. Search is lead generation, not proof that the long tail was exhausted, so preserve this as an explicit backlog item.
- coverage-gaps-need-family-specific-fixtures: Underserved model families need family-specific fixtures before support language [needs-validation] - Non-generative CV, graph/geometric/scientific ML, time-series, structured, and recsys ports have source-framework semantics that generic encoder or VLM parity checks do not cover.

## Decision notes
- The top-1000 sweep improved implementation evidence, but rate limits and long-tail drift still require a repeatable rescreening backlog item.

## Open validation
- top1000-long-tail-rescreening-needed: Automate or manually rerun the long-tail sweep when GitHub search budget is available.
- coverage-gaps-need-family-specific-fixtures: Keep existing P1 backlog items separate and add golden scenarios before extending architecture support.
