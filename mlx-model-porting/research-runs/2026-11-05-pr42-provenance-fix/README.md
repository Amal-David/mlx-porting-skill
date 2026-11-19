# PR #42 source-locator repair

This offline reconciliation applies the reviewed URL-normalization fix to the
current research graph. Article-identifying query parameters, including
`https://qwen.ai/blog?id=qwen3-next`, must survive normalization.

Command:

```bash
python3 mlx-model-porting/scripts/nightly_knowledge_curator.py \
  --run-id 2026-11-05-pr42-provenance-fix \
  --skip-live-collectors --no-research-loop
```

The run reuses the existing discovery and contributor receipts. Their original
retrieval dates remain authoritative; this is not a new network collection.
Earlier research runs and benchmark receipts remain unchanged.
