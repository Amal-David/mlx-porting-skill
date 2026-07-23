"""Contract: published offline-test counts match the real discovered suite size.

README.md and VALIDATION.md advertise how many offline tests ship. Those numbers
drift whenever a test is added or removed. This test discovers the true count
in-process and fails loud -- printing the discovered count -- when a document is
stale, so a release pass can correct the prose from one authoritative number.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


def discovered_test_count() -> int:
    """Count TestCase methods across the suite exactly as the canonical runner does.

    Discovery imports the already-loaded test modules (cached, so no module-level
    code re-runs) and only counts test methods; it never executes them. It is
    called solely inside a test method, so importing this module cannot trigger
    recursive discovery.
    """
    suite = unittest.defaultTestLoader.discover(str(TESTS_DIR))
    return suite.countTestCases()


class CorpusCountContractTests(unittest.TestCase):
    def test_published_offline_test_counts_match_the_discovered_suite(self) -> None:
        actual = discovered_test_count()

        validation = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
        validation_match = re.search(r"\|\s*Offline tests\s*\|\s*(\d+)\s*\|", validation)
        self.assertIsNotNone(
            validation_match,
            "VALIDATION.md must state an offline-test count as '| Offline tests | N |'",
        )
        validation_count = int(validation_match.group(1))

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_match = re.search(r"(\d+)\s+offline tests", readme)
        self.assertIsNotNone(
            readme_match,
            "README.md must state an offline-test count as 'N offline tests'",
        )
        readme_count = int(readme_match.group(1))

        self.assertEqual(
            (validation_count, readme_count),
            (actual, actual),
            f"published offline-test counts are stale: discovered {actual} tests, but "
            f"VALIDATION.md says {validation_count} and README.md says {readme_count}; "
            f"update both to {actual}.",
        )


if __name__ == "__main__":
    unittest.main()
