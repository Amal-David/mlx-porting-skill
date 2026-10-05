"""Offline Doctor: preserve canonical decisions without asserting runtime proof."""
from __future__ import annotations
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import mlx_doctor as doctor
from _common import SkillError


class DoctorContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.model = self.work / "model"
        shutil.copytree(ROOT / "tests/fixtures/models/decoder", self.model)
        self.host = {"system": "Darwin", "machine": "arm64", "python": "3.12.0", "python_executable": "python3", "apple_silicon": True, "chip": "test-chip", "physical_memory_bytes": 16 * 1024**3, "packages": {p: "test-version" for p in doctor.PACKAGES}, "metal_execution": "not-probed", "check_scope": "metadata-only-no-ML-framework-import"}

    def run_doctor(self, *args, host=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(doctor, "host_metadata", return_value=copy.deepcopy(self.host if host is None else host)), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = doctor.main([str(arg) for arg in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def report(self, *args, expected=0, host=None):
        code, stdout, stderr = self.run_doctor(*args, host=host)
        self.assertEqual(code, expected, stderr + stdout)
        return json.loads(stdout)

    def test_host_only_is_metadata_not_device_execution(self):
        report = self.report()
        self.assertEqual(report["input"]["kind"], "host")
        self.assertEqual(report["host"]["metal_execution"], "not-probed")
        self.assertFalse(report["architecture"]["native_execution_verified"])

    def test_model_route_has_no_native_or_profile_claim(self):
        report = self.report(self.model)
        self.assertEqual(report["architecture"]["families"], ["dense-decoder-transformer"])
        self.assertEqual(report["architecture"]["scaffold"], "family-available-profile-unverified")
        self.assertIsNone(report["memory"]["estimated_runtime_bytes"])
        self.assertFalse(report["memory"]["fit_verified"])
        self.assertTrue(all(not a["executed"] for a in report["next_actions"]))

    def test_all_seventeen_canonical_routes_are_preserved(self):
        covered = set()
        for path in sorted((ROOT / "tests/fixtures/scenarios").glob("*.json")):
            case = json.loads(path.read_text())
            with self.subTest(case=case["name"]):
                code, stdout, stderr = self.run_doctor(ROOT / "tests/fixtures" / case["fixture"])
                self.assertIn(code, (0, 1), stderr)
                report = json.loads(stdout)
                self.assertIn(case["expected_family"], report["architecture"]["families"])
                self.assertIn(case["expected_runbook"], report["architecture"]["runbooks"])
                self.assertFalse(report["architecture"]["native_execution_verified"])
                covered.update(report["architecture"]["families"])
        self.assertEqual(len(covered), 17)

    def test_default_report_omits_absolute_paths_and_names_placeholders(self):
        code, stdout, stderr = self.run_doctor(self.model)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn(str(self.work), stdout)
        inspect = next(a for a in json.loads(stdout)["next_actions"] if a["id"] == "inspect")
        self.assertEqual(set(inspect["requires_values"]), {"<PYTHON>", "<SKILL_SCRIPTS>", "<MODEL_PATH>"})

    def test_opt_in_local_argv_is_explicit(self):
        report = self.report(self.model, "--include-local-paths")
        inspect = next(a for a in report["next_actions"] if a["id"] == "inspect")
        self.assertEqual(inspect["argv"][0], sys.executable)
        self.assertIn(str(self.model), inspect["argv"])
        self.assertFalse(inspect["requires_values"])

    def test_remote_code_blocker_is_preserved_without_execution(self):
        marker = self.work / "must-not-exist"
        (self.model / "custom.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
        config = json.loads((self.model / "config.json").read_text())
        config["auto_map"] = {"AutoModel": "custom.CustomModel"}
        (self.model / "config.json").write_text(json.dumps(config))
        report = self.report(self.model, expected=1)
        self.assertTrue(report["blockers"])
        self.assertFalse(marker.exists())
        self.assertNotIn("scaffold", [a["id"] for a in report["next_actions"]])

    def test_broken_weights_never_produce_a_clean_report(self):
        (self.model / "model.safetensors").write_bytes(b"broken")
        report = self.report(self.model, expected=1)
        self.assertTrue(report["blockers"])
        self.assertEqual(report["architecture"]["scaffold"], "blocked")

    def test_truncated_inventory_blocks_advice(self):
        report = self.report(self.model, "--max-files", "1", expected=1)
        self.assertTrue(any("truncat" in b or "identity" in b for b in report["blockers"]))

    def test_missing_runtime_only_fails_when_requested(self):
        host = copy.deepcopy(self.host)
        host["system"], host["machine"], host["apple_silicon"] = "Linux", "x86_64", False
        host["packages"] = {p: None for p in doctor.PACKAGES}
        report = self.report(self.model, host=host)
        self.assertEqual(report["status"], "runtime-prerequisites-missing")
        self.assertTrue(report["runtime_blockers"])
        self.report(self.model, "--require-runtime", expected=1, host=host)

    def test_backend_mismatch_stays_explicit(self):
        report = self.report(self.model, "--backend", "mlx-audio", "--require-runtime", expected=1)
        self.assertTrue(any("not a target hint" in b for b in report["runtime_blockers"]))

    def test_weight_budget_is_not_a_runtime_fit_estimate(self):
        args = doctor.parse_args([str(self.model), "--memory-budget-gib", "1", "--require-runtime"])
        raw = doctor.inspect_input(self.model, "model", args, None)
        raw["tensor_summary"]["estimated_bytes"] = 2 * 1024**3
        with mock.patch.object(doctor, "inspect_input", return_value=raw):
            report = self.report(self.model, "--memory-budget-gib", "1", "--require-runtime", expected=1)
        self.assertFalse(report["memory"]["fit_verified"])
        self.assertIsNone(report["memory"]["estimated_runtime_bytes"])

    def test_project_proof_filenames_are_not_runtime_validation(self):
        project = self.work / "project"
        project.mkdir()
        (project / "model.py").write_text("import mlx.core as mx\n# no execution\n")
        (project / "parity.json").write_text('{}')
        (project / "benchmark.json").write_text('{}')
        report = self.report(project, "--kind", "project")
        self.assertFalse(report["architecture"]["native_execution_verified"])
        self.assertIsNotNone(report["project_health"])

    def test_nested_project_model_blockers_are_propagated(self):
        project = self.work / "project"
        project.mkdir()
        (project / "model.py").write_text("import mlx.core as mx\n")
        (self.model / "model.safetensors").write_bytes(b"broken")
        report = self.report(project, "--kind", "project", "--model", self.model, expected=1)
        self.assertTrue(report["blockers"])

    def test_failed_inspector_reports_error_not_success(self):
        result = subprocess.CompletedProcess([], 2, "", "invalid source")
        with mock.patch.object(doctor, "run_process_capture", return_value=(result, False)):
            code, stdout, stderr = self.run_doctor(self.model)
        self.assertEqual(code, 2)
        self.assertFalse(stdout)
        self.assertEqual(json.loads(stderr)["status"], "inspection-error")

    def test_inspector_deadline_is_fail_closed(self):
        result = subprocess.CompletedProcess([], -9, "", "")
        with mock.patch.object(doctor, "run_process_capture", return_value=(result, True)):
            code, _, stderr = self.run_doctor(self.model)
        self.assertEqual(code, 2)
        self.assertIn("timeout", stderr)

    def test_network_identifiers_and_network_flags_are_rejected(self):
        for args in (("missing-owner/missing-repository",), (str(self.model), "--allow-network"), (str(self.model), "--execute")):
            with self.subTest(args=args):
                code, stdout, stderr = self.run_doctor(*args)
                self.assertEqual(code, 2)
                self.assertFalse(stdout)
                self.assertFalse(json.loads(stderr)["ok"])

    def test_invalid_bounds_are_json_errors(self):
        for flag, value in (("--max-files", "0"), ("--timeout-seconds", "301"), ("--memory-budget-gib", "-1"), ("--max-files", "bad")):
            with self.subTest(flag=flag, value=value):
                code, _, stderr = self.run_doctor(self.model, flag, value)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stderr)["status"], "inspection-error")

    def test_input_symlink_is_rejected(self):
        link = self.work / "model-link"
        link.symlink_to(self.model, target_is_directory=True)
        code, _, _ = self.run_doctor(link)
        self.assertEqual(code, 2)

    def test_json_and_markdown_are_new_complete_files(self):
        output, markdown = self.work / "report.json", self.work / "report.md"
        report = self.report(self.model, "--output", output, "--markdown", markdown)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertIn("# MLX Doctor", markdown.read_text())
        before = output.read_bytes()
        code, _, _ = self.run_doctor(self.model, "--output", output)
        self.assertEqual(code, 2)
        self.assertEqual(output.read_bytes(), before)
        self.assertFalse(list(self.work.glob(".mlx-doctor-*")))

    def test_output_cannot_change_input_or_alias_another_report(self):
        for args in (("--output", self.model / "doctor.json"), ("--output", self.work / "same", "--markdown", self.work / "same")):
            with self.subTest(args=args):
                code, _, _ = self.run_doctor(self.model, *args)
                self.assertEqual(code, 2)
        self.assertFalse((self.model / "doctor.json").exists())

    def test_output_symlink_and_linked_parent_are_rejected(self):
        link = self.work / "report.json"
        link.symlink_to(self.work / "missing")
        self.assertEqual(self.run_doctor(self.model, "--output", link)[0], 2)
        parent = self.work / "linked"
        parent.symlink_to(self.work, target_is_directory=True)
        self.assertEqual(self.run_doctor(self.model, "--output", parent / "new.json")[0], 2)
        self.assertFalse((self.work / "new.json").exists())

    def test_markdown_stdout_names_limits(self):
        code, stdout, stderr = self.run_doctor(self.model, "--format", "markdown")
        self.assertEqual(code, 0, stderr)
        self.assertIn("Native execution / parity | Not verified", stdout)
        self.assertIn("Runtime memory fit | Not verified", stdout)

    def test_metadata_and_doctor_imports_do_not_load_frameworks(self):
        code = f"import sys;sys.path.insert(0,{str(SCRIPTS)!r});import mlx_doctor;mlx_doctor.host_metadata();assert not any(k.split('.')[0] in ('mlx','torch','transformers') for k in sys.modules)"
        completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_and_installed_script_help_work(self):
        for path in (ROOT / "mlx-doctor", SCRIPTS / "mlx_doctor.py"):
            completed = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(b"--require-runtime", completed.stdout)


if __name__ == "__main__":
    unittest.main()
