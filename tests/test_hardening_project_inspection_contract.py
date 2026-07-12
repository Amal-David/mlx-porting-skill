from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _common as common  # noqa: E402
import inspect_mlx_project  # noqa: E402


class ProjectInspectionHardeningTests(unittest.TestCase):
    def test_reports_use_portable_local_references_unless_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "portable-project"
            project.mkdir()
            (project / "model.py").write_text("import mlx.core as mx\n", encoding="utf-8")
            portable_output = Path(tmp) / "portable.json"
            explicit_output = Path(tmp) / "explicit.json"

            for output, extra in (
                (portable_output, []),
                (explicit_output, ["--include-local-paths"]),
            ):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "inspect_mlx_project.py"),
                        str(project),
                        "--output",
                        str(output),
                        *extra,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            portable = portable_output.read_text(encoding="utf-8")
            self.assertNotIn(str(project), portable)
            self.assertEqual(json.loads(portable)["project"]["path"], project.name)
            self.assertEqual(
                json.loads(explicit_output.read_text(encoding="utf-8"))["project"]["path"],
                str(project.resolve()),
            )

    def test_truncated_inventory_blocks_looks_good_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text(
                "import time\nimport mlx.core as mx\n"
                "assert mx.allclose(mx.array([1]), mx.array([1]))\n"
                "started = time.perf_counter()\nprint('tokens/s', started)\n",
                encoding="utf-8",
            )
            (root / "z.py").write_text("import mlx.core as mx\n", encoding="utf-8")
            output = root / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "inspect_mlx_project.py"),
                    str(root),
                    "--max-files",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["project"]["inventory_truncated"])
            self.assertEqual(report["health"]["status"], "inventory-truncated")
            self.assertTrue(report["recommendation_blockers"])
            self.assertIn(
                "complete-project-inventory",
                {item["id"] for item in report["improvement_opportunities"]},
            )

    def test_model_value_that_looks_like_an_option_is_rejected(self) -> None:
        with self.assertRaisesRegex(common.SkillError, "model path"):
            inspect_mlx_project.run_model_inspection("--help")

    def test_missing_child_output_is_a_skill_error(self) -> None:
        completed = subprocess.CompletedProcess(
            [sys.executable, "inspect_model.py"],
            0,
            stdout="",
            stderr="",
        )
        with (
            mock.patch.object(
                inspect_mlx_project,
                "run_process_capture",
                return_value=(completed, False),
            ),
            self.assertRaisesRegex(common.SkillError, "without writing"),
        ):
            inspect_mlx_project.run_model_inspection("safe-model")

    def test_cli_rejects_ambiguous_model_value_without_showing_child_help(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "inspect_mlx_project.py"),
                    tmp,
                    "--model",
                    "--help",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires a path value", completed.stderr)
            self.assertNotIn("Inspect a local MLX project", completed.stdout)


if __name__ == "__main__":
    unittest.main()
