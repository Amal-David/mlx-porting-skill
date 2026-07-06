from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import parse_frontmatter  # noqa: E402 - requires SCRIPTS on sys.path above


class DiscoveryTests(unittest.TestCase):
    def test_skill_discovery_symlinks_resolve_to_repo_skill(self) -> None:
        for rel in (".claude/skills/mlx-model-porting", ".agents/skills/mlx-model-porting"):
            with self.subTest(path=rel):
                path = ROOT / rel
                self.assertTrue(path.exists(), f"{rel} does not exist")
                self.assertTrue(path.is_symlink(), f"{rel} is not a symlink")
                self.assertEqual(path.resolve(strict=True), SKILL.resolve(strict=True))
                self.assertTrue((path / "SKILL.md").read_text(encoding="utf-8").startswith("---\n"))

    def test_skill_discovery_symlinks_are_not_gitignored(self) -> None:
        for rel in (".claude/skills/mlx-model-porting", ".agents/skills/mlx-model-porting"):
            with self.subTest(path=rel):
                result = subprocess.run(
                    ["git", "check-ignore", rel],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertNotEqual(
                    result.returncode,
                    0,
                    f"{rel} is git-ignored\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )

    def test_frontmatter_description_stays_discoverable(self) -> None:
        frontmatter, _body = parse_frontmatter((SKILL / "SKILL.md").read_text(encoding="utf-8"))
        description = frontmatter.get("description")
        self.assertIsInstance(description, str)
        assert isinstance(description, str)
        self.assertIn("Use when", description)
        self.assertIn("Do not use", description)
        self.assertLessEqual(len(description), 1024)


if __name__ == "__main__":
    unittest.main()
