from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "daily-research.yml"
REQUIREMENTS = ROOT / ".github" / "requirements-ci.txt"


class DailyResearchWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def job_block(self, name: str) -> str:
        match = re.search(
            rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
            self.workflow,
        )
        self.assertIsNotNone(match, f"missing {name!r} job")
        return match.group(0)

    def test_collection_fails_closed_before_publication(self) -> None:
        self.assertRegex(
            self.workflow,
            r"(?m)^permissions:\n  contents: read$",
        )
        self.assertRegex(
            self.workflow,
            r"(?m)^concurrency:\n  group: daily-mlx-research-\$\{\{ github\.repository \}\}\n  cancel-in-progress: false$",
        )
        collect = self.job_block("collect")
        self.assertIn("permissions:\n      contents: read", collect)
        self.assertIn("persist-credentials: false", collect)
        self.assertIn("--fail-on-network-error", collect)
        self.assertIn('candidate_path="$RUNNER_TEMP/update-candidates.json"', collect)
        self.assertIn('--output "$candidate_path"', collect)
        self.assertIn(
            '--previous "$previous_candidate_path"',
            collect,
        )
        self.assertIn("ref=automation%2Fdaily-mlx-research", collect)
        self.assertIn("--base mlx-model-porting/assets/update-candidates.json", collect)
        self.assertNotIn(
            "--output mlx-model-porting/assets/update-candidates.json",
            collect,
        )
        self.assertIn("actions/upload-artifact@", collect)
        self.assertLess(
            collect.index("--fail-on-network-error"),
            collect.index("actions/upload-artifact@"),
        )
        self.assertNotIn("git add", collect)
        self.assertNotIn("git push", collect)

    def test_publish_job_is_the_only_writer(self) -> None:
        publish = self.job_block("publish")
        self.assertIn("needs: collect", publish)
        self.assertIn("contents: write", publish)
        self.assertIn("pull-requests: write", publish)
        self.assertIn("actions/download-artifact@", publish)
        self.assertIn("persist-credentials: true", publish)
        self.assertIn(
            'install -m 0644 "$candidate_path" mlx-model-porting/assets/update-candidates.json',
            publish,
        )
        self.assertNotIn("update_sources.py", publish)
        self.assertIn("git add mlx-model-porting/assets/update-candidates.json", publish)
        self.assertIn("git add MANIFEST.json", publish)
        self.assertIn('gh pr close "$pr_number" --delete-branch', publish)
        self.assertIn('git push origin --delete "$branch"', publish)
        self.assertIn('cmp -s "$candidate_path" "$automation_snapshot_path"', publish)
        self.assertLess(
            publish.index('cmp -s "$candidate_path" "$automation_snapshot_path"'),
            publish.index('git checkout -B "$branch" "origin/${default_branch}"'),
        )
        self.assertIn('git checkout -B "$branch" "origin/${default_branch}"', publish)

    def test_all_actions_are_pinned_to_documented_release_commits(self) -> None:
        expected = {
            "actions/checkout": (
                "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
                "v7.0.0",
            ),
            "actions/setup-python": (
                "ece7cb06caefa5fff74198d8649806c4678c61a1",
                "v6.3.0",
            ),
            "actions/upload-artifact": (
                "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
                "v7.0.1",
            ),
            "actions/download-artifact": (
                "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
                "v8.0.1",
            ),
        }
        uses = re.findall(
            r"(?m)^\s+(?:- )?uses: ([^@\s]+)@([0-9a-f]{40})\s+#\s+(v\d+\.\d+\.\d+)\s*$",
            self.workflow,
        )
        self.assertTrue(uses, "workflow has no immutable action pins")
        observed: dict[str, set[tuple[str, str]]] = {}
        for action, sha, version in uses:
            observed.setdefault(action, set()).add((sha, version))
        self.assertEqual(
            observed,
            {action: {pin} for action, pin in expected.items()},
        )
        self.assertNotRegex(self.workflow, r"(?m)^\s+(?:- )?uses: [^\n]+@v\d+")

    def test_ci_dependency_is_version_and_hash_pinned(self) -> None:
        self.assertTrue(REQUIREMENTS.is_file(), "missing CI dependency lock")
        requirements = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("numpy==2.5.1", requirements)
        self.assertIn(
            "--hash=sha256:59fda5e192b570217ec2580c96f00e9a7e12ef6866a900eb089b62c1a32545ca",
            requirements,
        )
        self.assertIn("--require-hashes", self.workflow)
        self.assertIn("--only-binary=:all:", self.workflow)
        self.assertIn("-r .github/requirements-ci.txt", self.workflow)


if __name__ == "__main__":
    unittest.main()
