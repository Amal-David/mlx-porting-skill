from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkerRetirementTests(unittest.TestCase):
    def test_public_advisor_worker_is_not_shipped(self) -> None:
        worker_path = ROOT / "apps" / ("model-advisor-" + "worker")
        self.assertFalse(
            worker_path.exists(),
            "The retired public advisor Worker must not be shipped",
        )

        forbidden_fragments = (
            "apps/model-advisor-" + "worker",
            ".wrangler/",
            "dist-pages/",
            "wrangler." + "toml",
            "workers_" + "dev",
            "npm run " + "smoke:local",
            "127.0.0.1:" + "8787",
            "mlx-model-advisor." + "pages.dev",
            "_worker." + "js",
        )
        surfaces = [
            ROOT / ".gitignore",
            ROOT / "AGENTS.md",
            ROOT / "CHANGELOG.md",
            ROOT / "CLAUDE.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "EVIDENCE_INDEX.md",
            ROOT / "README.md",
            ROOT / "RESEARCH_REPORT.md",
            ROOT / "VALIDATION.md",
            ROOT / ".github",
            ROOT / "adapters",
            ROOT / "mlx-model-porting" / "SKILL.md",
            ROOT / "mlx-model-porting" / "examples",
            ROOT / "mlx-model-porting" / "references",
            ROOT / "site",
            ROOT / "tasks",
        ]
        offenders: list[str] = []
        for surface in surfaces:
            paths = [surface] if surface.is_file() else surface.rglob("*")
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for fragment in forbidden_fragments:
                    if fragment in text:
                        offenders.append(f"{path.relative_to(ROOT)}: {fragment}")

        self.assertEqual(
            offenders,
            [],
            "Retired Worker references remain in shipped surfaces:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
