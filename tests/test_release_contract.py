"""Release metadata and CI supply-chain contracts.

These tests intentionally derive the release identity from ``VERSION``. A release
bump is incomplete until every public surface and the generated manifest agree.
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from datetime import datetime
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting" / "SKILL.md"
CHANGELOG = ROOT / "CHANGELOG.md"
RESEARCH_REPORT = ROOT / "RESEARCH_REPORT.md"
MANIFEST = ROOT / "MANIFEST.json"
WORKFLOW = ROOT / ".github" / "workflows" / "validate.yml"
SOURCE_HEALTH_WORKFLOW = ROOT / ".github" / "workflows" / "source-health.yml"
REQUIREMENTS = ROOT / ".github" / "requirements-ci.txt"
SITE_DATA_PREFIX = "window.MLX_PORTING_SITE_DATA = "
SEMVER_RE = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")


def canonical_version() -> str:
    return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise AssertionError(f"missing leading frontmatter in {path}")
    return parts[1]


def site_data() -> dict[str, object]:
    text = (ROOT / "site" / "data.js").read_text(encoding="utf-8")
    if not text.startswith(SITE_DATA_PREFIX) or not text.endswith(";\n"):
        raise AssertionError("site/data.js must be one deterministic window-global assignment")
    value = json.loads(text[len(SITE_DATA_PREFIX):-2])
    if not isinstance(value, dict):
        raise AssertionError("site/data.js payload must be a mapping")
    return value


def job_block(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
        workflow,
    )
    if match is None:
        raise AssertionError(f"missing {name!r} job")
    return match.group(0)


def workflow_run_scripts(workflow: str) -> tuple[str, ...]:
    """Extract only YAML ``run`` payloads, excluding names and comments."""
    lines = workflow.splitlines()
    scripts: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(?P<indent>\s*)run:\s*(?P<value>.*)$", lines[index])
        if match is None:
            index += 1
            continue
        indent = len(match.group("indent"))
        value = match.group("value").strip()
        if value not in {"|", "|-", ">", ">-"}:
            scripts.append(value.strip('"\''))
            index += 1
            continue
        index += 1
        payload: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            payload.append(line)
            index += 1
        nonblank_indents = [len(line) - len(line.lstrip()) for line in payload if line.strip()]
        trim = min(nonblank_indents) if nonblank_indents else 0
        scripts.append("\n".join(line[trim:] if len(line) >= trim else "" for line in payload))
    return tuple(scripts)


def executable_run_text(workflow: str) -> str:
    scripts: list[str] = []
    for script in workflow_run_scripts(workflow):
        uncommented = "\n".join(
            line for line in script.splitlines()
            if not line.lstrip().startswith("#")
        )
        scripts.append(re.sub(r"\\\s*\n\s*", " ", uncommented))
    return "\n".join(scripts)


def assert_executable_fragment(test: unittest.TestCase, workflow: str, fragment: str) -> None:
    normalized_scripts = re.sub(r"\s+", " ", executable_run_text(workflow))
    normalized_fragment = re.sub(r"\s+", " ", fragment)
    test.assertIn(normalized_fragment, normalized_scripts, f"missing executable workflow command: {fragment}")


class ReleaseContractTests(unittest.TestCase):
    def test_version_file_is_one_canonical_semver_line(self) -> None:
        raw = (ROOT / "VERSION").read_text(encoding="utf-8")
        self.assertRegex(raw, r"\A(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\n\Z")
        self.assertIsNotNone(SEMVER_RE.fullmatch(canonical_version()))

    def test_skill_frontmatter_matches_release_version(self) -> None:
        metadata = frontmatter(SKILL)
        version_match = re.search(r'(?m)^  version:\s*["\']?([^"\'\s]+)["\']?\s*$', metadata)
        reviewed_match = re.search(r'(?m)^  last-reviewed:\s*["\']?(\d{4}-\d{2}-\d{2})["\']?\s*$', metadata)
        self.assertIsNotNone(version_match, "skill frontmatter needs metadata.version")
        self.assertIsNotNone(reviewed_match, "skill frontmatter needs metadata.last-reviewed")
        self.assertEqual(version_match.group(1), canonical_version())
        datetime.strptime(reviewed_match.group(1), "%Y-%m-%d")

    def test_every_adapter_verification_uses_only_the_current_version(self) -> None:
        expected = canonical_version()
        version_pattern = re.compile(r"\bversion:?\s*`([^`]+)`", re.IGNORECASE)
        adapters = sorted((ROOT / "adapters").glob("*.md"))
        self.assertTrue(adapters, "no adapter documentation found")
        for path in adapters:
            with self.subTest(adapter=path.name):
                versions = version_pattern.findall(path.read_text(encoding="utf-8"))
                self.assertTrue(versions, f"{path} has no verification version")
                self.assertEqual(set(versions), {expected})

    def test_site_data_and_all_html_fallbacks_match_release_version(self) -> None:
        expected = canonical_version()
        self.assertEqual(site_data().get("version"), expected)
        pattern = re.compile(r'data-value="version"[^>]*>([^<]+)<')
        pages = (ROOT / "site" / "index.html", ROOT / "site" / "docs" / "index.html")
        for page in pages:
            with self.subTest(page=page.relative_to(ROOT)):
                fallbacks = pattern.findall(page.read_text(encoding="utf-8"))
                self.assertTrue(fallbacks, "page has no non-JavaScript release fallback")
                self.assertEqual(set(value.strip() for value in fallbacks), {expected})

    def test_changelog_research_and_review_metadata_describe_the_same_release(self) -> None:
        expected = canonical_version()
        changelog = CHANGELOG.read_text(encoding="utf-8")
        release_match = re.search(
            r"(?m)^## ((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\s+[—-]\s+(\d{4}-\d{2}-\d{2})\s*$",
            changelog,
        )
        self.assertIsNotNone(release_match, "changelog needs a dated release heading")
        self.assertEqual(release_match.group(1), expected, "latest changelog entry is stale")
        release_date = datetime.strptime(release_match.group(2), "%Y-%m-%d").date()

        report = RESEARCH_REPORT.read_text(encoding="utf-8")
        report_match = re.search(
            r"(?m)^\*\*Artifact version:\*\*\s+((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\s*$",
            report,
        )
        self.assertIsNotNone(report_match, "research report needs Artifact version metadata")
        self.assertEqual(report_match.group(1), expected, "research report release metadata is stale")

        reviewed_match = re.search(
            r'(?m)^  last-reviewed:\s*["\']?(\d{4}-\d{2}-\d{2})["\']?\s*$',
            frontmatter(SKILL),
        )
        self.assertEqual(
            datetime.strptime(reviewed_match.group(1), "%Y-%m-%d").date(),
            release_date,
            "skill review date and release date must move together",
        )

    def test_manifest_metadata_and_version_record_match_the_release(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("schema_version"), 1)
        self.assertEqual(manifest.get("artifact"), "mlx-porting-skill")
        self.assertEqual(manifest.get("version"), canonical_version())
        generated_at = datetime.fromisoformat(str(manifest.get("generated_at")))
        self.assertIsNotNone(generated_at.tzinfo, "manifest generation time must include a timezone")

        files = manifest.get("files")
        self.assertIsInstance(files, list)
        self.assertEqual(manifest.get("file_count"), len(files))
        paths: list[str] = []
        for index, record in enumerate(files):
            with self.subTest(manifest_record=index):
                self.assertIsInstance(record, dict)
                self.assertIsInstance(record.get("path"), str)
                self.assertTrue(record["path"])
                paths.append(record["path"])
        self.assertEqual(len(paths), len(set(paths)), "manifest contains duplicate paths")
        self.assertEqual(paths, sorted(paths), "manifest entries must be deterministic")
        for raw_path in paths:
            path = PurePosixPath(str(raw_path))
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)

        by_path = {record["path"]: record for record in files}
        for required in (
            ".github/workflows/source-health.yml",
            "VERSION",
            "mlx-model-porting/SKILL.md",
            "site/data.js",
            "site/index.html",
            "site/docs/index.html",
            "tests/test_public_claims_contract.py",
        ):
            self.assertIn(required, by_path, f"release surface missing from manifest: {required}")

        version_bytes = (ROOT / "VERSION").read_bytes()
        version_record = by_path["VERSION"]
        self.assertEqual(version_record.get("size_bytes"), len(version_bytes))
        self.assertEqual(version_record.get("sha256"), hashlib.sha256(version_bytes).hexdigest())

    def test_validate_workflow_uses_immutable_actions_and_hash_locked_dependencies(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        expected_actions = {
            "actions/checkout": ("9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0", "v7.0.0"),
            "actions/setup-python": ("ece7cb06caefa5fff74198d8649806c4678c61a1", "v6.3.0"),
        }
        uses = re.findall(
            r"(?m)^\s+(?:- )?uses: ([^@\s]+)@([0-9a-f]{40})\s+#\s+(v\d+\.\d+\.\d+)\s*$",
            workflow,
        )
        self.assertTrue(uses, "validation workflow has no immutable action pins")
        all_uses = re.findall(r"(?m)^\s+(?:- )?uses:\s+([^\n]+)\s*$", workflow)
        self.assertEqual(
            len(all_uses),
            len(uses),
            "every action use must have a full commit SHA and an audited release annotation",
        )
        observed: dict[str, set[tuple[str, str]]] = {}
        for action, sha, version in uses:
            observed.setdefault(action, set()).add((sha, version))
        self.assertEqual(observed, {action: {pin} for action, pin in expected_actions.items()})
        self.assertNotRegex(workflow, r"(?m)^\s+- uses: [^\n]+@v\d+")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        self.assertNotRegex(workflow, r"(?m)^\s+(?:contents|actions|pull-requests): write$")

        self.assertIn("--require-hashes", workflow)
        self.assertIn("--only-binary=:all:", workflow)
        self.assertIn("-r .github/requirements-ci.txt", workflow)
        requirements = re.sub(
            r"\\\s*\n\s*",
            " ",
            REQUIREMENTS.read_text(encoding="utf-8"),
        )
        requirement_lines = [
            line.strip() for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(requirement_lines, "CI dependency lock is empty")
        for line in requirement_lines:
            with self.subTest(requirement=line):
                self.assertRegex(
                    line,
                    r'^[A-Za-z0-9_.-]+==[^\s]+\s*;\s*python_version\s*(?:==|>=)\s*"[^"]+"'
                    r'(?:\s+--hash=sha256:[0-9a-f]{64})+$',
                )
        self.assertIn('numpy==2.2.6 ; python_version == "3.10"', requirements)
        self.assertIn('numpy==2.5.1 ; python_version == "3.12"', requirements)
        self.assertIn('numpy==2.5.1 ; python_version == "3.14"', requirements)
        self.assertEqual(
            set(re.findall(r"--hash=sha256:([0-9a-f]{64})", requirements)),
            {
                "fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915",
                "59fda5e192b570217ec2580c96f00e9a7e12ef6866a900eb089b62c1a32545ca",
                "54ad769f17bc2d833b620851989f62054fb9ab93c969d9e1dc3c8e3d56beea21",
            },
        )

    def test_validate_workflow_has_complete_offline_and_compatibility_lanes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        release = job_block(workflow, "release-validation")
        supported = job_block(workflow, "supported-python")
        for block in (release, supported):
            self.assertRegex(block, r"(?m)^    timeout-minutes: [1-9]\d*$")
            self.assertIn("persist-credentials: false", block)

        for command in (
            "audit_skill.py --strict",
            "validate_sources.py mlx-model-porting",
            "validate_benchmarks.py check",
            "generate_claim_catalog.py --check",
            "generate_evidence_index.py --check",
            "generate_site_data.py --check",
            "manifest.py check",
            "node --check site/app.js",
            "python3 -m unittest discover -s tests -v",
            "git diff --check",
            "install_skill.py --dest",
        ):
            assert_executable_fragment(self, release, command)

        scanner_digest = "d52e62fe0237bf0199cb79f94593051300b0989d9aeab8b431001d9f75e469c1"
        assert_executable_fragment(
            self,
            release,
            f"ghcr.io/trufflesecurity/trufflehog@sha256:{scanner_digest}",
        )
        assert_executable_fragment(self, release, "filesystem /repo --fail --no-update --github-actions")
        assert_executable_fragment(self, release, "--results=verified")
        self.assertNotIn("--results=verified,unknown", executable_run_text(release))
        self.assertNotRegex(
            executable_run_text(release),
            r"ghcr\.io/trufflesecurity/trufflehog:[^\s]+",
            "secret scanner image tags are mutable; use a registry digest",
        )
        assert_executable_fragment(self, release, "python3 -m unittest -v tests.test_distribution_portability")
        assert_executable_fragment(self, release, "tests.test_public_claims_contract")

        self.assertIn('python-version: ["3.10", "3.14"]', supported)
        assert_executable_fragment(self, supported, "python3 -m pip install")
        assert_executable_fragment(self, supported, "--require-hashes --only-binary=:all:")
        assert_executable_fragment(self, supported, "python3 -m compileall")
        assert_executable_fragment(self, supported, "python3 -m unittest discover -s tests -v")
        self.assertNotRegex(supported, r"\b(?:curl|wget|gh)\b")
        self.assertNotIn("--check-urls", workflow)
        self.assertNotIn("schedule:", workflow)

    def test_source_health_is_trusted_read_only_reported_and_failure_preserving(self) -> None:
        workflow = SOURCE_HEALTH_WORKFLOW.read_text(encoding="utf-8")
        event_block = re.search(r"(?ms)^on:\n(.*?)(?=^permissions:)", workflow)
        self.assertIsNotNone(event_block)
        self.assertEqual(
            set(re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):", event_block.group(1))),
            {"schedule", "workflow_dispatch"},
        )
        self.assertRegex(workflow, r'(?m)^  schedule:\n    - cron: "[^"]+"$')
        self.assertRegex(workflow, r"(?m)^  workflow_dispatch:$")
        self.assertNotRegex(workflow, r"(?m)^  (?:push|pull_request):")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        self.assertNotRegex(workflow, r"(?m)^\s+(?:contents|actions|pull-requests): write$")
        job = job_block(workflow, "check-source-health")
        self.assertRegex(job, r"(?m)^    timeout-minutes: [1-9]\d*$")
        self.assertIn("persist-credentials: false", job)
        assert_executable_fragment(
            self,
            job,
            'python3 mlx-model-porting/scripts/validate_sources.py mlx-model-porting --check-urls',
        )
        assert_executable_fragment(self, job, '--output "$RUNNER_TEMP/source-health-report.json"')
        self.assertIn("if: ${{ always() }}", job)
        self.assertIn("if-no-files-found: error", job)
        self.assertIn("${{ runner.temp }}/source-health-report.json", job)
        self.assertEqual(
            len(workflow_run_scripts(job)),
            1,
            "source-health may execute only the non-mutating URL validator",
        )
        self.assertNotRegex(job, r"(?i)\b(?:generate_claim_catalog|generate_evidence_index)\.py\b")
        self.assertNotRegex(job, r"(?i)\b(?:generate|regenerate)\b.*\bevidence\b")

        expected_actions = {
            "actions/checkout": ("9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0", "v7.0.0"),
            "actions/setup-python": ("ece7cb06caefa5fff74198d8649806c4678c61a1", "v6.3.0"),
            "actions/upload-artifact": ("043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", "v7.0.1"),
        }
        uses = re.findall(
            r"(?m)^\s+(?:- )?uses: ([^@\s]+)@([0-9a-f]{40})\s+#\s+(v\d+\.\d+\.\d+)\s*$",
            workflow,
        )
        all_uses = re.findall(r"(?m)^\s+(?:- )?uses:\s+([^\n]+)\s*$", workflow)
        self.assertEqual(len(all_uses), len(uses), "source-health actions must use immutable commit SHAs")
        observed: dict[str, set[tuple[str, str]]] = {}
        for action, sha, version in uses:
            observed.setdefault(action, set()).add((sha, version))
        self.assertEqual(observed, {action: {pin} for action, pin in expected_actions.items()})

    def test_validation_workflow_run_steps_cannot_mask_coverage_failures(self) -> None:
        validate_workflow = WORKFLOW.read_text(encoding="utf-8")
        source_health_workflow = SOURCE_HEALTH_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotRegex(validate_workflow, r"(?m)^\s*if:\s*")
        self.assertEqual(
            re.findall(r"(?m)^\s*if:\s*(.+?)\s*$", source_health_workflow),
            ["${{ always() }}"],
            "only the report upload may bypass the preceding source-health failure",
        )
        for path in (WORKFLOW, SOURCE_HEALTH_WORKFLOW):
            workflow = path.read_text(encoding="utf-8")
            with self.subTest(workflow=path.name):
                self.assertNotRegex(workflow, r"(?mi)^\s*continue-on-error:")
                self.assertNotRegex(workflow, r"(?mi)^\s*if:\s*(?:false|\$\{\{\s*false\s*\}\})\s*$")
                scripts = workflow_run_scripts(workflow)
                self.assertTrue(scripts, "workflow has no executable run steps")
                for script in scripts:
                    self.assertNotIn("||", script, "workflow commands must not mask a failed left-hand command")
                    self.assertNotRegex(script, r"(?m)\bset\s+\+e\b")
                    self.assertNotRegex(script, r"(?m)^\s*!\s*(?:python\d*|node|git)\b")
                    self.assertNotRegex(script, r"(?m)(?:^|[;\n])\s*exit\s+0\s*(?:$|[;#])")
                    self.assertNotRegex(script, r"(?m)(?:^|[;\n])\s*true\s*(?:$|[;#])")

        validate_runs = executable_run_text(validate_workflow)
        source_runs = executable_run_text(source_health_workflow)
        self.assertNotIn("--check-urls", validate_runs)
        self.assertIn("--check-urls", source_runs)
        self.assertNotIn("generate_evidence_index.py", source_runs)

    def test_ubuntu_jobs_explicitly_acknowledge_missing_mlx_keystone_coverage(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        acknowledgement = "Acknowledge Ubuntu MLX keystone coverage gap"
        self.assertEqual(workflow.count(acknowledgement), 2)
        self.assertEqual(workflow.count('run: test "$RUNNER_OS" = "Linux"'), 2)
        self.assertEqual(workflow.count("MLX_KEYSTONE_REQUIRED=1"), 2)
        for job_name in ("release-validation", "supported-python"):
            block = job_block(workflow, job_name)
            self.assertIn(acknowledgement, block)
            self.assertIn("MLX_KEYSTONE_REQUIRED=1", block)


if __name__ == "__main__":
    unittest.main()
