from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_fixtures", FIXTURES / "generate_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FixtureReproducibilityTests(unittest.TestCase):
    def test_committed_fixtures_match_generator(self) -> None:
        # Every committed binary/opaque fixture must be reproducible from the
        # generator, so a future edit cannot smuggle in an unaudited blob.
        gen = _load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            produced = gen.generate(tmp)
            self.assertTrue(produced, "generator produced no files")
            for path in produced:
                rel = path.relative_to(tmp)
                committed = FIXTURES / rel
                self.assertTrue(committed.exists(), f"committed fixture missing: {rel}")
                if path.suffix == ".npz":
                    # .npz is a zip with embedded timestamps; compare by array content.
                    fresh = np.load(path)
                    base = np.load(committed)
                    self.assertEqual(sorted(fresh.files), sorted(base.files), str(rel))
                    for key in fresh.files:
                        np.testing.assert_array_equal(fresh[key], base[key], err_msg=str(rel))
                else:
                    self.assertEqual(path.read_bytes(), committed.read_bytes(), f"byte mismatch: {rel}")

    def test_codec_fixture_is_kb_scale(self) -> None:
        size = (FIXTURES / "models" / "codec" / "model.safetensors").stat().st_size
        self.assertLess(size, 64 * 1024, f"codec fixture should be KB-scale, got {size} bytes")


if __name__ == "__main__":
    unittest.main()
