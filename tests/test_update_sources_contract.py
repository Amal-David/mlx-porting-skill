from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "mlx-model-porting" / "scripts" / "update_sources.py"
FIXTURE = ROOT / "tests" / "fixtures" / "updates" / "offline.json"


class UpdateSourcesContractTests(unittest.TestCase):
    def test_arxiv_candidate_preserves_identity_and_exact_revision(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            fixture = root / "fixture.json"
            sources = root / "sources.json"
            output = root / "candidates.json"
            fixture.write_text(json.dumps({
                "repositories": [],
                "papers": [{
                    "kind": "paper",
                    "query": "revision fixture",
                    "id": "http://arxiv.org/abs/2601.12345v2",
                    "title": "Versioned fixture paper",
                    "updated": "2026-07-10T00:00:00Z",
                    "published": "2026-07-09T00:00:00Z",
                    "authors": ["Fixture Author"],
                    "summary": "Exact arXiv revision fixture.",
                }],
            }), encoding="utf-8")
            sources.write_text(json.dumps({
                "sources": [{
                    "id": "fixture-paper",
                    "url": "https://arxiv.org/abs/2601.12345v1",
                    "kind": "paper",
                    "snapshot": "2601.12345v1",
                }],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--offline-fixture",
                    str(fixture),
                    "--sources",
                    str(sources),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            paper = json.loads(output.read_text(encoding="utf-8"))["papers"][0]
            self.assertEqual(paper["id"], "http://arxiv.org/abs/2601.12345v2")
            self.assertEqual(paper["canonical_url"], "https://arxiv.org/abs/2601.12345")
            self.assertEqual(paper["arxiv_id"], "2601.12345")
            self.assertEqual(paper["revision"], "v2")
            self.assertEqual(paper["immutable_url"], "https://arxiv.org/abs/2601.12345v2")
            self.assertTrue(paper["known_url"])
            self.assertEqual(paper["known_revision"], "v1")

    def test_unchanged_candidate_snapshot_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "candidates.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--offline-fixture",
                str(FIXTURE),
                "--output",
                str(output),
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            time.sleep(0.01)
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
