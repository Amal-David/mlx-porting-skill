"""Filesystem safety contracts for distribution and static inspection tools."""
from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _common as common  # noqa: E402
import inspect_mlx_project  # noqa: E402
import inspect_model  # noqa: E402
import manifest  # noqa: E402


class HardeningFilesystemContractTests(unittest.TestCase):
    def run_installer(self, *args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "install_skill.py"), *(str(arg) for arg in args)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_installer_rejects_destination_or_target_inside_source(self) -> None:
        nested_destination = SKILL / "recursive-install-root"
        nested = self.run_installer(
            "--dest",
            nested_destination,
            "--mode",
            "copy",
            "--dry-run",
        )
        self.assertEqual(nested.returncode, 2, nested.stdout + nested.stderr)
        self.assertIn("inside the source skill directory", nested.stderr)
        self.assertFalse(nested_destination.exists())

        exact_target = self.run_installer(
            "--dest",
            SKILL.parent,
            "--name",
            SKILL.name,
            "--mode",
            "symlink",
            "--dry-run",
        )
        self.assertEqual(exact_target.returncode, 2, exact_target.stdout + exact_target.stderr)
        self.assertIn("source skill directory", exact_target.stderr)

    def test_installer_rejects_target_ancestor_that_contains_source(self) -> None:
        target_ancestor = SKILL.parents[1]
        self.assertTrue(SKILL.is_relative_to(target_ancestor))

        result = self.run_installer(
            "--dest",
            target_ancestor.parent,
            "--name",
            target_ancestor.name,
            "--mode",
            "copy",
            "--dry-run",
        )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("source skill directory", result.stderr)

    def test_installer_rejects_target_symlink_cycle_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            destination = Path(raw_tmp) / "skills"
            destination.mkdir()
            target = destination / SKILL.name
            target.symlink_to(target.name, target_is_directory=True)

            result = self.run_installer("--dest", destination, "--mode", "symlink", "--force")

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("symlink cycle", result.stderr.lower())
            self.assertNotIn("Traceback", result.stderr)
            self.assertTrue(target.is_symlink())

    def test_manifest_records_file_directory_and_dangling_symlinks_without_following(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
            (root / "target.txt").write_text("target", encoding="utf-8")
            directory = root / "target-dir"
            directory.mkdir()
            (directory / "inside.txt").write_text("inside", encoding="utf-8")
            (root / "file-link").symlink_to("target.txt")
            (root / "directory-link").symlink_to("target-dir", target_is_directory=True)
            (root / "dangling-link").symlink_to("missing-target")

            records = {record["path"]: record for record in manifest.build_files(root)}

            for name, target in (
                ("file-link", "target.txt"),
                ("directory-link", "target-dir"),
                ("dangling-link", "missing-target"),
            ):
                with self.subTest(name=name):
                    self.assertEqual(records[name]["type"], "symlink")
                    self.assertEqual(records[name]["target"], target)
            self.assertNotIn("directory-link/inside.txt", records)

    def test_manifest_rejects_unsafe_version_nodes_and_oversized_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "actual-version").write_text("1.0.0\n", encoding="utf-8")
            (root / "VERSION").symlink_to("actual-version")
            with self.assertRaisesRegex(common.SkillError, "VERSION.*symlink"):
                manifest.read_version(root)

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "VERSION").mkdir()
            with self.assertRaisesRegex(common.SkillError, "VERSION.*regular file"):
                manifest.read_version(root)

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "VERSION").write_bytes(b"1" * (manifest.MAX_VERSION_BYTES + 1))
            with self.assertRaisesRegex(common.SkillError, "VERSION.*size limit"):
                manifest.read_version(root)

    def test_manifest_prunes_excluded_directories_before_enumerating_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
            (root / "keep.txt").write_text("keep", encoding="utf-8")
            excluded = root / "node_modules"
            excluded.mkdir()
            (excluded / "must-not-be-scanned.js").write_text("noise", encoding="utf-8")
            real_listdir = os.listdir
            excluded_metadata = excluded.stat()

            def guarded_listdir(path: object):
                if isinstance(path, int):
                    metadata = os.fstat(path)
                    is_excluded = (metadata.st_dev, metadata.st_ino) == (
                        excluded_metadata.st_dev,
                        excluded_metadata.st_ino,
                    )
                else:
                    is_excluded = Path(path) == excluded
                if is_excluded:
                    raise AssertionError("excluded directory was enumerated")
                return real_listdir(path)

            with mock.patch.object(manifest.os, "listdir", side_effect=guarded_listdir):
                records = {record["path"] for record in manifest.build_files(root)}

            self.assertIn("keep.txt", records)
            self.assertFalse(any(path.startswith("node_modules/") for path in records))

    def test_inspect_model_bounds_json_and_readme_and_uses_bounded_root_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            config = root / "config.json"
            config.write_text(json.dumps({"model_type": "fixture"}), encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
            (root / "LICENSE-A").write_text("license", encoding="utf-8")
            (root / "LICENSE-Z").write_text("license", encoding="utf-8")
            bounded_records = [
                {"path": "README.md", "suffix": ".md", "size_bytes": readme.stat().st_size},
                {"path": "LICENSE-A", "suffix": "", "size_bytes": 7},
            ]

            with mock.patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
                parsed = inspect_model.read_json(config)
                license_report = inspect_model.extract_license(
                    root,
                    {"config.json": parsed or {}},
                    bounded_records,
                )

            self.assertEqual(parsed, {"model_type": "fixture"})
            self.assertEqual(
                license_report["declared"],
                [{"source": "README.md", "value": "apache-2.0"}],
            )
            self.assertEqual(license_report["license_files"], ["LICENSE-A"])

    def test_truncated_model_inventory_is_a_high_risk_recommendation_blocker(self) -> None:
        model = ROOT / "tests" / "fixtures" / "models" / "decoder"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "inspect_model.py"),
                str(model),
                "--max-files",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)

        self.assertTrue(report["file_summary"]["truncated"])
        self.assertIn("inventory-truncated", {risk["type"] for risk in report["risks"]})
        self.assertTrue(
            any("inventory" in blocker and "truncated" in blocker for blocker in report["recommendation_blockers"]),
            report["recommendation_blockers"],
        )
        self.assertIsNone(report["recommended_family"])
        self.assertEqual(report["recommended_families"], [])
        self.assertNotEqual(report["routing_decision"]["status"], "recommended")

    def test_keras_preflight_rejects_central_directory_bounds_before_zipfile(self) -> None:
        cases = (
            (
                "member-count",
                inspect_model.MAX_ARCHIVE_MEMBERS + 1,
                0,
                "member limit",
            ),
            (
                "central-directory-size",
                0,
                inspect_model.MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES + 1,
                "central directory size limit",
            ),
        )
        for name, member_count, central_directory_size, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw_tmp:
                archive_path = Path(raw_tmp) / "adversarial.keras"
                archive_path.write_bytes(struct.pack(
                    "<4s4H2IH",
                    b"PK\x05\x06",
                    0,
                    0,
                    member_count,
                    member_count,
                    central_directory_size,
                    0,
                    0,
                ))
                with mock.patch.object(
                    inspect_model.zipfile,
                    "ZipFile",
                    side_effect=AssertionError("ZipFile materialized an unbounded central directory"),
                ):
                    report = inspect_model.inspect_keras_archive(archive_path, archive_path)

                self.assertTrue(
                    any(expected_error in error for error in report.get("errors", [])),
                    report.get("errors", []),
                )

        with tempfile.TemporaryDirectory() as raw_tmp:
            archive_path = Path(raw_tmp) / "forged-member-count.keras"
            central_header = struct.pack(
                "<4s6H3I5H2I",
                b"PK\x01\x02",
                *([0] * 16),
            )
            central_directory = central_header * (inspect_model.MAX_ARCHIVE_MEMBERS + 1)
            eocd = struct.pack(
                "<4s4H2IH",
                b"PK\x05\x06",
                0,
                0,
                1,
                1,
                len(central_directory),
                0,
                0,
            )
            archive_path.write_bytes(central_directory + eocd)
            with mock.patch.object(
                inspect_model.zipfile,
                "ZipFile",
                side_effect=AssertionError("ZipFile trusted a forged member count"),
            ):
                report = inspect_model.inspect_keras_archive(archive_path, archive_path)
            self.assertTrue(
                any("member limit" in error for error in report.get("errors", [])),
                report.get("errors", []),
            )

    def test_bounded_files_sorts_before_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for name in ("z.txt", "a.txt", "m.txt"):
                (root / name).write_text(name, encoding="utf-8")

            paths, truncated = common.bounded_files(root, 2)

            self.assertTrue(truncated)
            self.assertEqual([path.name for path in paths], ["a.txt", "m.txt"])

    @unittest.skipIf(sys.platform == "win32", "POSIX permission mode contract")
    def test_install_tree_signature_detects_permission_mode_drift(self) -> None:
        import install_skill

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            left_file = left / "same.txt"
            right_file = right / "same.txt"
            left_file.write_text("same", encoding="utf-8")
            right_file.write_text("same", encoding="utf-8")
            left_file.chmod(0o644)
            right_file.chmod(0o600)

            self.assertNotEqual(
                install_skill.tree_signature(left),
                install_skill.tree_signature(right),
            )

    def test_inspect_mlx_project_quotes_next_action_paths_as_shell_tokens(self) -> None:
        project = Path("/tmp/project with spaces;echo injected")
        model = "/tmp/model $(touch injected)"
        args = SimpleNamespace(model=[model], include_local_paths=True)

        command = inspect_mlx_project.build_next_actions(args, project)[0]

        self.assertEqual(
            shlex.split(command),
            [
                "python3",
                "mlx-model-porting/scripts/inspect_mlx_project.py",
                str(project.resolve()),
                "--model",
                str(Path(model).resolve()),
                "--include-local-paths",
                "--markdown",
                "MLX_INSPECTION.md",
            ],
        )


if __name__ == "__main__":
    unittest.main()
