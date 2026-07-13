# T5-small port record

- Source: Hugging Face `t5-small`, cached revision
  `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`.
- Provenance: immutable Hub revision plus original config and deterministic
  overlay digests are recorded in `SOURCE_PROVENANCE.json`. The cached snapshot
  omitted its model card; the pinned upstream Apache-2.0 declaration is not
  misrepresented as locally byte-verified evidence.
- Source runtime: built-in Transformers T5, offline, `trust_remote_code=false`.
- Target: standalone eager MLX on Apple M4 Pro, FP32 converted weights.
- Fixture: encoder token IDs
  `[13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5, 1]`, decoder start ID `0`,
  eight greedy steps, no padding; a separate real two-prompt test covers padded
  batches.
- Correctness gate: exact input IDs and attention mask, every encoder and decoder
  semantic boundary, first-step logits, exact nontrivial greedy IDs, bucket
  edges, cache/full decode and cross-attention K/V reuse, T5 corruption
  negatives, complete schema-2 weight coverage, and explicit
  ignore provenance for the unused cross-attention relative-bias key.
- Rollback condition: any missing weight, non-finite tensor, cache/full-decode
  disagreement, floating rung below cosine `0.999999`, or greedy token mismatch.
- Deferred: beam search, task-quality evaluation, quantization, compilation,
  performance claims, and non-T5 encoder-decoder aliases.
