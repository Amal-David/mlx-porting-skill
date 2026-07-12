from __future__ import annotations

import os
import re
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
            ".wrangler" + "/",
            "dist-" + "pages/",
            "wrangler." + "toml",
            "npm run " + "smoke:local",
            "127.0.0.1:" + "8787",
            "_worker." + "js",
        )
        forbidden_host_patterns = (
            (
                re.compile(
                    r"\b(?:[a-z0-9-]+\.)*" + "workers" + r"\." + "dev" + r"\b",
                    re.IGNORECASE,
                ),
                "retired Worker host",
            ),
            (
                re.compile(
                    r"\b"
                    + "mlx-model-advisor"
                    + r"[a-z0-9-]*\."
                    + "pages"
                    + r"\."
                    + "dev"
                    + r"\b",
                    re.IGNORECASE,
                ),
                "retired advisor Pages host",
            ),
        )
        ignored_directories = {
            ".git",
            ".superplan",
            ".wrangler",
            "__pycache__",
            "node_modules",
            "research-runs",
        }
        offenders: list[str] = []
        for directory, directory_names, file_names in os.walk(ROOT):
            directory_names[:] = sorted(
                name for name in directory_names if name not in ignored_directories
            )
            for file_name in sorted(file_names):
                path = Path(directory) / file_name
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for fragment in forbidden_fragments:
                    if fragment in text:
                        offenders.append(f"{path.relative_to(ROOT)}: {fragment}")
                for pattern, label in forbidden_host_patterns:
                    for match in pattern.finditer(text):
                        offenders.append(
                            f"{path.relative_to(ROOT)}: {label} ({match.group(0)})"
                        )

        self.assertEqual(
            offenders,
            [],
            "Retired Worker references remain in shipped surfaces:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
