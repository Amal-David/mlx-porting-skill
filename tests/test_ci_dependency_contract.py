"""Keep the regenerable MLX CI inputs aligned with the hash-locked install."""
from __future__ import annotations

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / ".github" / "requirements-mlx-keystones.in"
LOCK = ROOT / ".github" / "requirements-mlx-keystones.txt"


def requirements(text: str) -> dict[str, tuple[str, str]]:
    """Read the deliberately restricted, exact-pin requirements format."""
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    result = {}
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)(.*)", line)
        if match is None:
            raise ValueError(f"expected an exact package pin: {line}")
        name, version, suffix = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in result:
            raise ValueError(f"duplicate package pin: {name}")
        result[name] = (version, suffix)
    return result


class CIDependencyContractTests(unittest.TestCase):
    def test_direct_inputs_match_the_hashed_keystone_lock(self) -> None:
        direct = requirements(INPUTS.read_text(encoding="utf-8"))
        locked = requirements(LOCK.read_text(encoding="utf-8"))
        self.assertTrue({"mlx", "mlx-lm", "torch", "transformers", "tokenizers", "safetensors", "numpy"} <= direct.keys())
        for name, (version, suffix) in direct.items():
            with self.subTest(package=name):
                self.assertEqual(suffix, "", "direct inputs must be plain exact pins")
                self.assertIn(name, locked)
                self.assertEqual(version, locked[name][0], "regenerate the lock after editing a direct dependency")
        for name, (_, hashes) in locked.items():
            with self.subTest(locked_package=name):
                self.assertRegex(hashes, r"^(?:\s+--hash=sha256:[0-9a-f]{64})+\s*$")

    def test_requirement_reader_rejects_unpinned_and_duplicate_inputs(self) -> None:
        for text in ("mlx>=0.32.0", "mlx==0.32.0\nmlx==0.33.0", "mlx_lm==0.31.3\nmlx-lm==0.31.3"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                requirements(text)


if __name__ == "__main__":
    unittest.main()
