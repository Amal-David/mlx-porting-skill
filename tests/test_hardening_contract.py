from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _common as common  # noqa: E402
import collect_contributors  # noqa: E402
import inspect_mlx_project  # noqa: E402
import inspect_model  # noqa: E402
import manifest  # noqa: E402
import research_loop  # noqa: E402
import run_research_campaign  # noqa: E402
import validate_sources  # noqa: E402


class HardeningContractTests(unittest.TestCase):
    def test_campaign_ingest_requires_typed_allowlisted_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operation = {
                "type": "research_loop_ingest",
                "arguments": {
                    "run_id": "fixture-run",
                    "objective": "Review fixture findings",
                    "agent_count": 1,
                    "assignment_mode": "config-order",
                    "ingest_subagent_results": True,
                    "output_dir": "wave-01",
                },
            }
            command, output_dir = run_research_campaign.typed_research_loop_command(
                operation,
                root,
                expected_type="research_loop_ingest",
            )
            self.assertEqual(command[:2], [sys.executable, str(SCRIPTS / "research_loop.py")])
            self.assertIn("--ingest-subagent-results", command)
            self.assertEqual(output_dir, (root / "wave-01").resolve())

            malicious_wave = {
                "ingest": {
                    "command_args": [sys.executable, "-c", "raise SystemExit('receipt executed arbitrary code')"]
                }
            }
            with self.assertRaisesRegex(common.SkillError, "typed operation"):
                run_research_campaign.ingest_command_for_wave(root, malicious_wave)

    def test_research_loop_renders_in_skill_output_dirs_portably(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="hardening-portable-", dir=ROOT / "mlx-model-porting"))
        try:
            summary = {
                "run_id": "portable-run",
                "objective": "Keep receipts portable",
                "agent_count": 1,
                "gap_hints": [],
                "iteration": 1,
                "iteration_count": 1,
                "assignment_planner": {"mode": "config-order"},
                "subagent_dispatch": {"agent_count": 1},
                "review_gate": {"requirements": {}},
            }
            command = research_loop.ingest_command_args_for_wave(summary, output_dir)
            rendered = command[command.index("--output-dir") + 1]
            self.assertEqual(rendered, common.safe_relpath(ROOT / "mlx-model-porting", output_dir))
            self.assertFalse(os.path.isabs(rendered))

            operation = research_loop.ingest_operation_for_wave(summary, output_dir)
            self.assertEqual(operation["type"], "research_loop_ingest")
            self.assertEqual(operation["arguments"]["output_dir"], rendered)
        finally:
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)

    def test_static_inventories_ignore_noise_and_reject_escaping_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "project"
            root.mkdir()
            (root / "keep.py").write_text("import mlx\n", encoding="utf-8")
            ignored = root / "node_modules"
            ignored.mkdir()
            for index in range(20):
                (ignored / f"noise-{index}.js").write_text("noise", encoding="utf-8")

            for inventory in (inspect_model.inventory, inspect_mlx_project.inventory):
                records, truncated = inventory(root, 1, False)
                self.assertEqual([record["path"] for record in records], ["keep.py"])
                self.assertFalse(truncated)

            outside = base / "outside.py"
            outside.write_text("secret", encoding="utf-8")
            (root / "escape.py").symlink_to(outside)
            for inventory in (inspect_model.inventory, inspect_mlx_project.inventory):
                with self.assertRaisesRegex(common.SkillError, "symlink escapes"):
                    inventory(root, 10, False)

    def test_keras_archive_expansion_is_bounded_before_member_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "oversized.keras"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(inspect_model.MAX_ARCHIVE_MEMBERS + 1):
                    archive.writestr(f"empty-{index}.txt", b"")
            report = inspect_model.inspect_keras_archive(archive_path, archive_path)
            self.assertTrue(any("member limit" in error for error in report.get("errors", [])))
            self.assertEqual(report["entries"], [])

    def test_github_token_headers_are_same_origin_only(self) -> None:
        client = collect_contributors.GitHubClient(
            timeout=1,
            token="top-secret-token",
            token_origin="https://api.github.com",
        )
        same_origin = client.request_headers("https://api.github.com/repos/ml-explore/mlx/contributors")
        cross_origin = client.request_headers("https://example.invalid/collect")
        self.assertEqual(same_origin["Authorization"], "Bearer top-secret-token")
        self.assertNotIn("Authorization", cross_origin)

        request = urllib.request.Request(
            "https://api.github.com/repos/ml-explore/mlx/contributors",
            headers=same_origin,
        )
        with mock.patch.object(validate_sources, "require_public_https_url"):
            redirected = client.redirect_handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://example.invalid/collect",
            )
        self.assertIsNotNone(redirected)
        self.assertNotIn("Authorization", dict(redirected.header_items()))

    @unittest.skipIf(sys.platform == "win32", "POSIX process-group behavior")
    def test_bounded_process_timeout_kills_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "descendant-survived"
            child = (
                "import pathlib,time; "
                "time.sleep(0.5); "
                f"pathlib.Path({str(marker)!r}).write_text('leaked')"
            )
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                "time.sleep(10)"
            )
            completed, timed_out = common.run_process_capture(
                [sys.executable, "-c", parent],
                timeout=0.1,
            )
            self.assertTrue(timed_out)
            self.assertNotEqual(completed.returncode, 0)
            time.sleep(0.7)
            self.assertFalse(marker.exists(), "timed-out subprocess descendant survived")

    def test_process_capture_bounds_output_with_head_and_tail_marker(self) -> None:
        code = (
            "import sys; "
            "sys.stdout.write('HEAD' + ('x' * "
            f"{common.MAX_CAPTURE_BYTES + 256}"
            ") + 'TAIL')"
        )
        completed, timed_out = common.run_process_capture([sys.executable, "-c", code], timeout=5)
        self.assertFalse(timed_out)
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(completed.stdout.startswith("HEAD"))
        self.assertTrue(completed.stdout.endswith("TAIL"))
        self.assertIn("[truncated ", completed.stdout)
        self.assertLessEqual(len(completed.stdout.encode("utf-8")), common.MAX_CAPTURE_BYTES)

    def test_json_receipts_redact_secrets_and_leave_no_staging_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "receipt.json"
            output.write_text("stale", encoding="utf-8")
            common.dump_json(
                {
                    "token": "top-secret-token",
                    "authorization": "Bearer top-secret-token",
                    "command": "worker --api-key top-secret-token",
                    "command_args": ["worker", "--access-token", "top-secret-token"],
                    "quoted_command": "worker --password='top-secret-token'",
                    "tokenizer": "safe-tokenizer-name",
                },
                output,
            )
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("top-secret-token", text)
            self.assertIn("safe-tokenizer-name", text)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_copy_installer_is_idempotent_and_does_not_leave_staging_trees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "skills"
            command = [
                sys.executable,
                str(SCRIPTS / "install_skill.py"),
                "--dest",
                str(dest),
                "--mode",
                "copy",
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already installed", second.stdout)
            self.assertTrue((dest / "mlx-model-porting" / "SKILL.md").is_file())
            self.assertFalse(any(path.name.startswith(".mlx-model-porting.install-") for path in dest.iterdir()))

    def test_symlink_installer_is_idempotent_and_atomically_replaces_stale_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "skills"
            target = dest / "mlx-model-porting"
            command = [
                sys.executable,
                str(SCRIPTS / "install_skill.py"),
                "--dest",
                str(dest),
                "--mode",
                "symlink",
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already installed", second.stdout)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), (ROOT / "mlx-model-porting").resolve())

            target.unlink()
            target.mkdir()
            (target / "stale.txt").write_text("stale", encoding="utf-8")
            replaced = subprocess.run([*command, "--force"], capture_output=True, text=True, check=False)
            self.assertEqual(replaced.returncode, 0, replaced.stderr)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), (ROOT / "mlx-model-porting").resolve())
            self.assertFalse(any(".install-" in path.name or ".backup-" in path.name for path in dest.iterdir()))

    def test_manifest_records_shipped_symlink_target_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
            skill = root / "mlx-model-porting"
            skill.mkdir()
            (skill / "SKILL.md").write_text("fixture", encoding="utf-8")
            link = root / ".agents" / "skills" / "mlx-model-porting"
            link.parent.mkdir(parents=True)
            link.symlink_to("../../mlx-model-porting", target_is_directory=True)

            records = {record["path"]: record for record in manifest.build_files(root)}
            record = records[".agents/skills/mlx-model-porting"]
            self.assertEqual(record["type"], "symlink")
            self.assertEqual(record["target"], "../../mlx-model-porting")
            self.assertEqual(record["size_bytes"], len(record["target"].encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
