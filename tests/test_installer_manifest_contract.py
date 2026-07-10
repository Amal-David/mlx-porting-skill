"""Contracts for manifest-attested copy installation."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import install_skill  # noqa: E402
from _common import SkillError  # noqa: E402


OFFICIAL_APACHE_2_SHA256 = "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a"


class InstallerManifestContractTests(unittest.TestCase):
    def make_fixture(self, parent: Path) -> tuple[Path, Path]:
        repository = parent / "release"
        skill = repository / "mlx-model-porting"
        scripts = skill / "scripts"
        scripts.mkdir(parents=True)
        (repository / "VERSION").write_text("1.0.0\n", encoding="utf-8")
        (skill / "SKILL.md").write_text("# Fixture skill\n", encoding="utf-8")
        (skill / "LICENSE").write_text("fixture license\n", encoding="utf-8")
        tool = scripts / "tool.py"
        tool.write_text("#!/usr/bin/env python3\nprint('fixture')\n", encoding="utf-8")
        tool.chmod(0o755)
        return repository, skill

    def file_record(self, skill: Path, relative: str) -> dict[str, object]:
        path = skill / relative
        content = path.read_bytes()
        return {
            "path": f"{skill.name}/{relative}",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "executable": bool(path.stat().st_mode & 0o111),
        }

    def symlink_record(self, skill: Path, relative: str) -> dict[str, object]:
        target = os.readlink(skill / relative)
        encoded = target.encode("utf-8")
        return {
            "path": f"{skill.name}/{relative}",
            "type": "symlink",
            "target": target,
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def write_manifest(
        self,
        repository: Path,
        skill: Path,
        *,
        relative_paths: tuple[str, ...] = ("LICENSE", "SKILL.md", "scripts/tool.py"),
        symlink_paths: tuple[str, ...] = (),
    ) -> Path:
        records = [self.file_record(skill, relative) for relative in relative_paths]
        records.extend(self.symlink_record(skill, relative) for relative in symlink_paths)
        records.sort(key=lambda record: str(record["path"]))
        payload = {
            "schema_version": 1,
            "artifact": "mlx-porting-skill",
            "version": "1.0.0",
            "generated_at": "2026-07-10T00:00:00+00:00",
            "file_count": len(records),
            "files": records,
        }
        manifest = repository / "MANIFEST.json"
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return manifest

    def run_main(self, skill: Path, destination: Path, *extra: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        arguments = [
            "install_skill.py",
            "--dest",
            str(destination),
            "--mode",
            "copy",
            *extra,
        ]
        with (
            mock.patch.object(install_skill, "SOURCE", skill),
            mock.patch.object(sys, "argv", arguments),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = install_skill.main()
        return result, stdout.getvalue(), stderr.getvalue()

    def test_copy_install_is_exact_manifest_identity_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            self.write_manifest(repository, skill)
            cache = skill / "scripts" / "__pycache__"
            cache.mkdir()
            (cache / "tool.cpython-314.pyc").write_bytes(b"local cache")
            destination = Path(raw_tmp) / "agent-skills"

            first, first_stdout, first_stderr = self.run_main(skill, destination)
            second, second_stdout, second_stderr = self.run_main(skill, destination)

            self.assertEqual(first, 0, first_stdout + first_stderr)
            self.assertEqual(second, 0, second_stdout + second_stderr)
            self.assertIn("already installed", second_stdout)
            installed = destination / skill.name
            entries = install_skill.load_distribution_manifest(source=skill)
            self.assertEqual(
                install_skill.tree_signature(installed),
                install_skill.expected_tree_signature(entries),
            )
            self.assertFalse((installed / "scripts" / "__pycache__").exists())
            self.assertEqual((installed / "scripts" / "tool.py").stat().st_mode & 0o777, 0o755)
            self.assertEqual((installed / "SKILL.md").stat().st_mode & 0o777, 0o644)

    def test_copy_install_rejects_unlisted_dirty_source_before_idempotence_check(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            self.write_manifest(repository, skill)
            destination = Path(raw_tmp) / "agent-skills"
            first, _, first_stderr = self.run_main(skill, destination)
            self.assertEqual(first, 0, first_stderr)
            (skill / "unlisted.txt").write_text("must not be installed\n", encoding="utf-8")

            second, second_stdout, second_stderr = self.run_main(skill, destination)

            self.assertEqual(second, 2, second_stdout + second_stderr)
            self.assertIn("unlisted source content: unlisted.txt", second_stderr)
            self.assertNotIn("already installed", second_stdout)
            self.assertFalse((destination / skill.name / "unlisted.txt").exists())

    def test_copy_install_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            self.write_manifest(repository, skill)
            (skill / "SKILL.md").write_text("tampered after manifest\n", encoding="utf-8")
            entries = install_skill.load_distribution_manifest(source=skill)

            with self.assertRaisesRegex(SkillError, "changed manifest content: SKILL.md"):
                install_skill.verify_source_tree(skill, entries)

    def test_manifest_requires_a_regular_in_payload_license(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            manifest = self.write_manifest(
                repository,
                skill,
                relative_paths=("SKILL.md", "scripts/tool.py"),
            )

            with self.assertRaisesRegex(SkillError, "missing regular mlx-model-porting/LICENSE"):
                install_skill.load_distribution_manifest(manifest, skill)

    def test_copy_install_preserves_only_safe_manifest_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            (skill / "tool-link.py").symlink_to("scripts/tool.py")
            self.write_manifest(repository, skill, symlink_paths=("tool-link.py",))
            destination = Path(raw_tmp) / "agent-skills"

            result, stdout, stderr = self.run_main(skill, destination)

            self.assertEqual(result, 0, stdout + stderr)
            installed_link = destination / skill.name / "tool-link.py"
            self.assertTrue(installed_link.is_symlink())
            self.assertEqual(os.readlink(installed_link), "scripts/tool.py")
            self.assertTrue(installed_link.resolve().is_file())

    def test_manifest_rejects_escaping_symlink_before_source_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            link = skill / "escape"
            link.symlink_to("../outside")
            manifest = self.write_manifest(repository, skill, symlink_paths=("escape",))

            with self.assertRaisesRegex(SkillError, "target escapes the skill root"):
                install_skill.load_distribution_manifest(manifest, skill)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO nodes require POSIX")
    def test_source_and_installed_signatures_reject_special_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            repository, skill = self.make_fixture(Path(raw_tmp))
            self.write_manifest(repository, skill)
            fifo = skill / "unexpected.fifo"
            os.mkfifo(fifo)
            entries = install_skill.load_distribution_manifest(source=skill)

            with self.assertRaisesRegex(SkillError, "unsupported FIFO at unexpected.fifo"):
                install_skill.verify_source_tree(skill, entries)
            with self.assertRaisesRegex(SkillError, "unsupported FIFO in installed tree"):
                install_skill.tree_signature(skill)

    def test_distributed_license_is_the_complete_official_apache_2_text(self) -> None:
        license_path = ROOT / "mlx-model-porting" / "LICENSE"
        content = license_path.read_bytes()

        self.assertEqual(hashlib.sha256(content).hexdigest(), OFFICIAL_APACHE_2_SHA256)
        text = content.decode("utf-8")
        self.assertIn("3. Grant of Patent License.", text)
        self.assertIn("APPENDIX: How to apply the Apache License to your work.", text)


if __name__ == "__main__":
    unittest.main()
