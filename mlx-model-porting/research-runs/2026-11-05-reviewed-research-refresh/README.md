# Reviewed research registry reconciliation

This is an offline reconciliation after the November 5 source review and
guidance update. It used the existing `update-candidates.json` and
`contributor-refresh.json`; their original retrieval dates remain authoritative.
The generated delta does not represent a fresh upstream or contributor scan.

Command:

```bash
python3 mlx-model-porting/scripts/nightly_knowledge_curator.py \
  --run-id 2026-11-05-reviewed-research-refresh \
  --skip-live-collectors --no-research-loop
```

New reviewed evidence is recorded in `assets/sources.yaml` and explained in
`references/research-updates-2026-11.md` and `references/runtime-compatibility.md`.
This run updates the current research graph and reconciles its backlog without
promoting measurements or changing any earlier run directory.
