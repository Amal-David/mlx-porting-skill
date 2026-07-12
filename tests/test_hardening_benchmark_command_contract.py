from __future__ import annotations

import io
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_command  # noqa: E402
import validate_benchmarks  # noqa: E402


class BenchmarkCommandHardeningTests(unittest.TestCase):
    def test_system_metadata_probe_uses_bounded_capture(self) -> None:
        with mock.patch.object(
            benchmark_command,
            "_run_bounded_probe",
            return_value=(b"bounded value\n", b""),
        ) as probe:
            self.assertEqual(benchmark_command.system_command(["fixture-metadata"]), "bounded value")
        kwargs = probe.call_args.kwargs
        self.assertEqual(kwargs["stdout_limit"], 64 * 1024)
        self.assertEqual(kwargs["stderr_limit"], 64 * 1024)
        self.assertEqual(kwargs["timeout"], 3)

    @unittest.skipIf(os.name == "nt", "symlink semantics differ on Windows")
    def test_artifacts_and_receipt_outputs_never_follow_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            outside = Path(outside_tmp)
            outside_file = outside / "runner.py"
            outside_file.write_text("# outside\n", encoding="utf-8")
            (root / "linked-runner.py").symlink_to(outside_file)
            raw = outside_file.read_bytes()
            descriptor = {
                "path": "linked-runner.py",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
            self.assertEqual(validate_benchmarks.check_artifact(root, descriptor), (False, "missing"))

            (root / "raw").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(benchmark_command.SkillError, "unsafe"):
                benchmark_command.prepare_receipt_output_path(
                    root,
                    "raw/fixture/benchmark-command.json",
                )
            self.assertFalse((outside / "fixture" / "benchmark-command.json").exists())

    def test_quality_contract_snapshot_detects_runner_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def artifact(path: str, text: str) -> dict[str, object]:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                raw = target.read_bytes()
                return {
                    "path": path,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }

            reference = artifact("quality/reference.txt", "stable output\n")
            candidate = artifact("quality/outputs/snapshot/observed.txt", "stable output\n")
            contract_path = root / "quality-contract.json"
            contract_path.write_text(json.dumps({
                "schema_version": 2,
                "validator": validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
                "metric": "exact-output-parity",
                "reference_artifact": reference,
                "candidate_artifact": candidate,
            }), encoding="utf-8")
            spec = {
                "label": "snapshot",
                "argv_template": [
                    {"source": ["variant_config", "quality_output_path"]},
                ],
                "variant_config": {"quality_output_path": candidate["path"]},
                "workload": {"artifacts": []},
            }

            _, _, control = benchmark_command.prepare_receipt_quality_output(
                root,
                spec,
                contract_path,
            )
            benchmark_command.verify_quality_control_snapshot(control)
            contract_path.write_text(contract_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark_command.SkillError, "changed during"):
                benchmark_command.verify_quality_control_snapshot(control)

    def test_invalid_revisions_and_nested_baselines_fail_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {
                "schema_version": 1,
                "label": "preflight",
                "argv_template": [],
                "models": {"target": {
                    "id": "fixture/model",
                    "revision": "main",
                    "lineage_id": "fixture-lineage",
                    "source_id": "fixture/source",
                    "source_revision": "b" * 40,
                }},
                "workload": {},
                "variant_config": {"mode": "candidate"},
                "enabled_methods": ["fixture-method"],
                "comparison_role": "candidate",
                "rollback_condition": "Rollback on regression.",
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(benchmark_command.SkillError, "pinned revisions"):
                benchmark_command.load_receipt_spec(str(spec_path))
            with self.assertRaisesRegex(benchmark_command.SkillError, "root-level"):
                benchmark_command.resolve_candidate_baseline(
                    root,
                    {"comparison_role": "candidate"},
                    "nested/baseline.json",
                )

            baseline_path = root / "baseline.json"
            baseline_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(benchmark_command.SkillError, "refusing to overwrite"):
                benchmark_command.validate_receipt_output_layout(
                    root,
                    baseline_path,
                    {"label": "baseline", "workload": {"artifacts": []}},
                    root / "quality/outputs/baseline/result.txt",
                    {"contract": {}, "path": root / "quality-contract.json"},
                    baseline_path,
                )

    def test_receipt_interpreter_isolated_metadata_is_bound_without_private_paths(self) -> None:
        resolved, descriptor = benchmark_command.resolve_receipt_interpreter(
            ["python3"],
            benchmark_command.controlled_receipt_environment(),
        )
        self.assertTrue(Path(resolved).is_file())
        self.assertEqual(descriptor["flags"], ["-I", "-B"])
        self.assertTrue(validate_benchmarks.external_interpreter_environment_valid(
            descriptor["environment"]
        ))
        serialized = json.dumps(descriptor["environment"])
        for private_prefix in ("/Users/", "/home/", "/private/", "/tmp/"):
            self.assertNotIn(private_prefix, serialized)

    def test_receipt_mode_rejects_unbounded_settings_hidden_env_and_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "runner.py"
            runner.write_text("# pinned runner\n", encoding="utf-8")
            input_path = root / "input.bin"
            input_path.write_text("input\n", encoding="utf-8")

            def descriptor(path: Path, role: str) -> dict[str, object]:
                raw = path.read_bytes()
                return {
                    "role": role,
                    "path": path.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }

            spec = {
                "schema_version": 1,
                "label": "bounded",
                "argv_template": [
                    {"literal": "python3"},
                    {"source": ["workload", "artifacts", 0, "path"]},
                    {"source": ["models", "target", "id"]},
                    {"source": ["workload", "artifacts", 1, "path"]},
                    {"source": ["variant_config", "mode"]},
                ],
                "models": {"target": {
                    "id": "fixture/model",
                    "revision": "a" * 40,
                    "lineage_id": "fixture-lineage",
                    "source_id": "fixture/source",
                    "source_revision": "b" * 40,
                }},
                "workload": {
                    "id": "bounded",
                    "artifacts": [descriptor(runner, "runner"), descriptor(input_path, "input")],
                    "parameters": {"steps": 1},
                },
                "variant_config": {"mode": "baseline"},
                "enabled_methods": [],
                "comparison_role": "baseline",
                "rollback_condition": "Rollback on regression.",
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            base = [
                "benchmark_command.py",
                "--receipt-spec", str(spec_path),
                "--quality-contract", str(root / "unused.json"),
                "--timeout", "10",
                "--output", str(root / "receipt.json"),
            ]
            timeout_index = base.index("--timeout")
            without_timeout = [
                *base[:timeout_index],
                *base[timeout_index + 2:],
            ]
            for timeout in (None, "0", "nan", "inf", str(benchmark_command.MAX_RECEIPT_TIMEOUT_SECONDS + 1)):
                argv = without_timeout if timeout is None else [*without_timeout, "--timeout", timeout]
                with self.subTest(timeout=timeout):
                    with (
                        mock.patch.object(sys, "argv", argv),
                        mock.patch.object(benchmark_command, "run_once") as run_once,
                    ):
                        self.assertEqual(benchmark_command.main(), 2)
                        run_once.assert_not_called()
            for extra in (
                ["--runs", str(benchmark_command.MAX_RECEIPT_RUNS + 1)],
                ["--warmup", str(benchmark_command.MAX_RECEIPT_WARMUPS + 1)],
                ["--stdout-tail", str(benchmark_command.MAX_RECEIPT_TAIL_BYTES + 1)],
                ["--stderr-tail", str(benchmark_command.MAX_RECEIPT_TAIL_BYTES + 1)],
                ["--env", "BENCHMARK_MODE=hidden-confounder"],
            ):
                with self.subTest(extra=extra), mock.patch.object(sys, "argv", [*base, *extra]):
                    self.assertEqual(benchmark_command.main(), 2)

            spec["workload"]["artifacts"][1]["sha256"] = "0" * 64
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with (
                mock.patch.object(sys, "argv", base),
                mock.patch.object(benchmark_command, "run_once") as run_once,
            ):
                self.assertEqual(benchmark_command.main(), 2)
                run_once.assert_not_called()

    def test_external_report_size_cap_fails_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "oversized.json"
            raw_path.write_text("{}", encoding="utf-8")
            raw_output = {
                "path": raw_path.name,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "size_bytes": validate_benchmarks.MAX_EXTERNAL_REPORT_BYTES + 1,
                "truncated": False,
            }
            run = {"run": 1, "wall_seconds": 1.0, "raw_output": raw_output}
            receipt = {"runs": [run]}

            self.assertFalse(validate_benchmarks._external_report_matches(root, receipt, run))

    def test_nested_secret_strings_and_option_lists_are_rejected(self) -> None:
        cases = (
            {"nested": ["AWS_SECRET_ACCESS_KEY=aws-secret"]},
            {"nested": [["PRIVATE_KEY=private-secret"]]},
            {"nested": ["DATABASE_URL=postgresql://user:password@db.example/app"]},
            {"nested": ["Authorization: Basic dXNlcjpwYXNz"]},
            {"nested": ["Authorization: Bearer bearer-secret"]},
            {"nested": ["X-API-Key: header-secret"]},
            {"nested": ["postgresql://user:url-password@db.example/app"]},
            {"nested": ["--private-key", "option-secret"]},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(validate_benchmarks.contains_serialized_secret(case))
        self.assertFalse(
            validate_benchmarks.contains_serialized_secret(
                {
                    "nested": [
                        "AWS_SECRET_ACCESS_KEY=[REDACTED]",
                        "tokenizer=safe-tokenizer",
                        "This is not instrumented first-token latency.",
                    ]
                }
            )
        )

    def test_assessment_adds_serialized_secret_blocker_for_nested_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = {
                "schema_version": 1,
                "label": "nested-secret-fixture",
                "runs": [],
                "metadata": {"headers": ["X-API-Key: nested-header-secret"]},
            }
            (root / "nested-secret-fixture.json").write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )

            report = validate_benchmarks.build_assessment_report(root)

            self.assertIn("serialized-secret", report["assessments"][0]["reasons"])

    def test_environment_override_receipt_never_serializes_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.json"
            argv = [
                "benchmark_command.py",
                "--warmup", "0",
                "--runs", "1",
                "--output", str(output),
                "--env", "BENCHMARK_MODE=ordinary-value-must-not-serialize",
                "--env", "AWS_SECRET_ACCESS_KEY=aws-value-must-not-serialize",
                "--env", "PRIVATE_KEY=private-value-must-not-serialize",
                "--",
                sys.executable,
                "-c",
                (
                    "import os,sys; "
                    "print(os.environ['BENCHMARK_MODE']); "
                    "print(os.environ['AWS_SECRET_ACCESS_KEY'], file=sys.stderr); "
                    "print(os.environ['PRIVATE_KEY'])"
                ),
            ]

            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(benchmark_command.main(), 0)

            serialized = output.read_text(encoding="utf-8")
            report = json.loads(serialized)
            self.assertEqual(
                report["environment_overrides"],
                {
                    "BENCHMARK_MODE": "[REDACTED]",
                    "AWS_SECRET_ACCESS_KEY": "[REDACTED]",
                    "PRIVATE_KEY": "[REDACTED]",
                },
            )
            self.assertNotIn("ordinary-value-must-not-serialize", serialized)
            self.assertNotIn("aws-value-must-not-serialize", serialized)
            self.assertNotIn("private-value-must-not-serialize", serialized)
            self.assertIn("[REDACTED]", report["runs"][0]["stdout_tail"])
            self.assertIn("[REDACTED]", report["runs"][0]["stderr_tail"])

    def test_monitor_setup_exception_terminates_process_tree_and_closes_pipes(self) -> None:
        process = mock.Mock()
        process.stdout = io.BytesIO(b"")
        process.stderr = io.BytesIO(b"")
        process.poll.return_value = None
        process.wait.return_value = -9

        with (
            mock.patch.object(benchmark_command.subprocess, "Popen", return_value=process),
            mock.patch.object(benchmark_command, "kill_process_tree") as kill_tree,
            mock.patch.object(benchmark_command.threading, "Thread", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                benchmark_command.run_once(
                    [sys.executable, "-c", "pass"],
                    cwd=None,
                    env={},
                    timeout=1,
                    stdout_tail=100,
                    stderr_tail=100,
                )

        kill_tree.assert_called_once_with(process)
        process.wait.assert_called()
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_capture_is_bounded_without_using_communicate(self) -> None:
        process = mock.Mock()
        process.stdout = io.BytesIO((b"x" * 100_000) + b"STDOUT-TAIL")
        process.stderr = io.BytesIO((b"y" * 100_000) + b"STDERR-TAIL")
        process.returncode = 0
        process.wait.return_value = 0
        process.poll.return_value = 0
        process.pid = 999_999
        process.communicate.side_effect = AssertionError("communicate() is an unbounded capture path")

        with mock.patch.object(benchmark_command.subprocess, "Popen", return_value=process):
            result = benchmark_command.run_once(
                [sys.executable, "-c", "pass"],
                cwd=None,
                env={},
                timeout=1,
                stdout_tail=32,
                stderr_tail=32,
            )

        process.communicate.assert_not_called()
        self.assertLessEqual(len(result["stdout_tail"].encode("utf-8")), 32)
        self.assertLessEqual(len(result["stderr_tail"].encode("utf-8")), 32)
        self.assertTrue(result["stdout_tail"].endswith("STDOUT-TAIL"))
        self.assertTrue(result["stderr_tail"].endswith("STDERR-TAIL"))

    def test_interpreter_probe_rejects_oversized_streams_and_cleans_up(self) -> None:
        for oversized_stream in ("stdout", "stderr"):
            with self.subTest(stream=oversized_stream):
                process = mock.Mock()
                process.stdout = io.BytesIO(
                    (b"x" * 100_000) if oversized_stream == "stdout" else b"ok"
                )
                process.stderr = io.BytesIO(
                    (b"y" * 100_000) if oversized_stream == "stderr" else b""
                )
                process.returncode = -9
                process.wait.return_value = -9
                process.pid = 999_999

                with (
                    mock.patch.object(benchmark_command.subprocess, "Popen", return_value=process),
                    mock.patch.object(benchmark_command, "kill_process_tree") as kill_tree,
                ):
                    with self.assertRaisesRegex(
                        benchmark_command.SkillError,
                        rf"fixture interpreter probe {oversized_stream} exceeded 32 bytes",
                    ):
                        benchmark_command._run_bounded_probe(
                            [sys.executable, "--version"],
                            env={},
                            timeout=1,
                            stdout_limit=32,
                            stderr_limit=32,
                            label="fixture interpreter probe",
                        )

                kill_tree.assert_called_once_with(process)
                process.wait.assert_called()
                self.assertTrue(process.stdout.closed)
                self.assertTrue(process.stderr.closed)

    def test_negative_tail_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(benchmark_command.SkillError, "tail sizes"):
            benchmark_command.run_once(
                [sys.executable, "-c", "pass"],
                cwd=None,
                env={},
                timeout=1,
                stdout_tail=-1,
                stderr_tail=10,
            )

    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    def test_exited_leader_cannot_leak_background_child(self) -> None:
        parent = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])"
        )
        started = time.monotonic()
        result = benchmark_command.run_once(
            [sys.executable, "-c", parent],
            cwd=None,
            env=os.environ.copy(),
            timeout=0.2,
            stdout_tail=100,
            stderr_tail=100,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
