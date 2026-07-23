"""pytest collection support for the offline test suite.

The canonical runner is ``python3 -m unittest discover -s tests``; it resolves the
shared helpers that some modules import as ``tests.mlx_keystone`` /
``tests.test_tooling``. pytest does not set up ``sys.path`` the same way, so
collection otherwise fails with ``ModuleNotFoundError: No module named
'tests.mlx_keystone'``. Insert the repository root so ``pytest tests/`` can
collect the same modules. This adds no test dependency; unittest stays the
supported runner and never loads this file.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# tests/ is a namespace package (no __init__.py). An unrelated regular `tests`
# package elsewhere on sys.path would shadow it and break `from tests.* import`,
# because a regular package outranks a namespace portion regardless of order.
# Bind the checkout's tests/ directory as the `tests` package so the local suite
# always wins during collection. This is a no-op when nothing else shadows it.
_LOCAL_TESTS = ROOT / "tests"
_bound = sys.modules.get("tests")
if _LOCAL_TESTS.is_dir() and (
    _bound is None
    or _LOCAL_TESTS not in {Path(entry).resolve() for entry in getattr(_bound, "__path__", ())}
):
    _package = types.ModuleType("tests")
    _package.__path__ = [str(_LOCAL_TESTS)]
    sys.modules["tests"] = _package
