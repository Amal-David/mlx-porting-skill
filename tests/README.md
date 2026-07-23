# Tests

The test suite uses tiny synthetic, non-executable model fixtures. It validates static architecture routing, security risk flags, architecture-aware plan generation, complete weight-map transforms, numerical parity reporting, benchmark capture, installation safety, and the skill audit.

```bash
python3 -m unittest discover -s tests -v
```

`unittest` is the canonical runner. `pytest tests/` is also supported for local convenience (a root `conftest.py` puts the repository root on `sys.path` so the shared `tests.*` helpers resolve); it is never required and is not a dependency.
