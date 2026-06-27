# Test fixtures

All fixtures here are tiny, synthetic, and non-executable. They exist to exercise
static tooling (routing, risk flags, weight-map shape math, tensor parity, plan and
recommendation generation) without downloading real models or running model code.

## Regenerating the binary fixtures

The opaque/binary fixtures are produced deterministically by
[`generate_fixtures.py`](generate_fixtures.py) — the single source of truth, so no
committed blob exists without a spec that explains it:

```bash
python3 tests/fixtures/generate_fixtures.py
```

`tests/test_fixtures.py` asserts the committed files match a fresh generation, so an
edit that diverges from the spec (or smuggles in an unaudited blob) fails CI.

| Path | Produced by generator | Purpose |
|---|---|---|
| `models/decoder/` | yes (`model.safetensors`, `config.json`) | Llama-style dense decoder routing + weight signals |
| `models/moe/` | yes | Mixtral-style MoE routing (`block_sparse_moe`, experts) |
| `models/codec/` | yes | EnCodec-style neural-audio-codec routing; codebook tensor is a **size-reduced** stand-in (`[8, 32, 16]`, not the real `[8, 1024, 128]`) so the fixture stays KB-scale |
| `models/unsafe/` | yes | Remote-code/pickle lure: `auto_map`, a `.py` module, `setup.py`, and a non-pickle `pytorch_model.bin`; must trip the intake risk gates |
| `tensors/{source,close,bad}.npz` | yes | Tensor-oracle parity: `close` is within tolerance of `source`, `bad` diverges sharply |
| `models/decoder/README.md` | no (hand-authored) | static example doc |
| `manifests/*.json` | no (hand-authored) | source/target/map manifests for `validate_weight_map.py` |
| `updates/offline.json` | no (hand-authored) | offline fixture for the review-only daily update collector |
| `scenarios/*.json` | no (hand-authored) | declarative end-to-end cases for `tests/test_scenarios.py` |

Determinism: tensor values come from a SHA-256-seeded NumPy RNG keyed by `model/key`,
so raw `.safetensors` files are byte-stable across runs. `.npz` archives embed zip
timestamps, so they are compared by array content rather than raw bytes.
