"""Offline source-provenance guards for the November 2026 MLX watchlist."""
from pathlib import Path
import unittest

REFS = Path(__file__).resolve().parents[1] / "mlx-model-porting" / "references"

class November2026WatchContractTests(unittest.TestCase):
    def test_immutable_upstream_commit_links(self):
        watch = (REFS / "research-updates-2026-11.md").read_text(encoding="utf-8")
        cases = (
            ("mlx-lm", 2076, "88df6e796caeaf59eb98af3f4564c8b119fac177"),
            ("mlx", 4910, "1c117492fd66047c0bfe5433f4a5a47f6119d214"),
            ("mlx", 4907, "bbc8c5d0c17b375bc4534e53ed8c740e573400ac"),
            ("mlx-lm", 2077, "77b5eb61800785a7ec219e75cccd8890e0e5b864"),
            ("mlx-lm", 1788, "e4c4af820905a2c660d7e4879a1cce5cc912b1f4"),
        )
        for project, pr, commit in cases:
            with self.subTest(pr=pr):
                self.assertRegex(commit, r"^[0-9a-f]{40}$")
                self.assertIn(f"https://github.com/ml-explore/{project}/pull/{pr}", watch)
                self.assertIn(f"https://github.com/ml-explore/{project}/commit/{commit}", watch)
        self.assertIn("not yet in a tagged MLX-LM release", watch)
        self.assertIn("No numeric result or runtime capability is promoted", watch)

    def test_distinct_cache_training_and_recurrence_gates(self):
        for name, markers in (
            ("attention-and-kv.md", ("/pull/2076", "left-padded", "multi-token")),
            ("compile-and-kernels.md", ("/pull/4910", "dQ/dK/dV", "bf16/fp16")),
            ("runbook-ssm-hybrid.md", ("/pull/4907", "per-step-input", "peak memory")),
        ):
            with self.subTest(reference=name):
                doc = (REFS / name).read_text(encoding="utf-8")
                for marker in markers:
                    self.assertIn(marker, doc)
                self.assertIn("fallback", doc.lower())

if __name__ == "__main__":
    unittest.main()
