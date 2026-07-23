"""Schema and mutation contracts for the distribution manifest."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
MANIFEST_SCRIPT = SCRIPTS / "manifest.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import manifest  # noqa: E402


Mutation = Callable[[dict[str, Any]], None]


class ManifestContractTests(unittest.TestCase):
    def run_manifest(
        self,
        action: str,
        root: Path,
        *,
        expected: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, str(MANIFEST_SCRIPT), action, "--root", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def create_root(self, parent: str) -> Path:
        root = Path(parent)
        (root / "VERSION").write_text("1.2.3\n", encoding="utf-8")
        (root / "payload.txt").write_text("payload\n", encoding="utf-8")
        return root

    def read_payload(self, root: Path) -> dict[str, Any]:
        return json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))

    def write_payload(self, root: Path, payload: Any) -> None:
        (root / "MANIFEST.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def assert_invalid_mutation(
        self,
        root: Path,
        canonical: dict[str, Any],
        mutation: Mutation,
        message: str,
    ) -> None:
        payload = copy.deepcopy(canonical)
        mutation(payload)
        self.write_payload(root, payload)
        result = self.run_manifest("check", root, expected=2)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def assert_swap_to_external_symlink_rejected(
        self,
        victim: Path,
        external: Path,
        operation: Callable[[], Any],
    ) -> None:
        real_open = manifest.os.open
        real_read = manifest.os.read
        parked = victim.with_name(f".{victim.name}.original")
        external_metadata = external.stat()
        parent_metadata = victim.parent.stat()
        swapped = False
        external_reads = 0

        def swapping_open(path: object, flags: int, *args: Any, **kwargs: Any) -> int:
            nonlocal swapped
            parent_descriptor = kwargs.get("dir_fd")
            parent_matches = False
            if isinstance(parent_descriptor, int):
                descriptor_metadata = os.fstat(parent_descriptor)
                parent_matches = (descriptor_metadata.st_dev, descriptor_metadata.st_ino) == (
                    parent_metadata.st_dev,
                    parent_metadata.st_ino,
                )
            if os.fspath(path) == victim.name and parent_matches and not swapped:
                self.assertIsInstance(manifest.NOFOLLOW_FLAG, int)
                self.assertTrue(flags & int(manifest.NOFOLLOW_FLAG))
                victim.replace(parked)
                victim.symlink_to(external)
                swapped = True
            return real_open(path, flags, *args, **kwargs)

        def guarded_read(descriptor: int, size: int) -> bytes:
            nonlocal external_reads
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) == (
                external_metadata.st_dev,
                external_metadata.st_ino,
            ):
                external_reads += 1
                raise AssertionError("external file content must not be read")
            return real_read(descriptor, size)

        with (
            mock.patch.object(manifest.os, "open", side_effect=swapping_open),
            mock.patch.object(manifest.os, "read", side_effect=guarded_read),
            self.assertRaisesRegex(manifest.SkillError, "must not be a symlink"),
        ):
            operation()

        self.assertTrue(swapped)
        self.assertEqual(external_reads, 0)

    def test_generation_is_stable_except_for_its_utc_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            first = manifest.generate(root)
            second = manifest.generate(root)

        self.assertEqual(first["schema_version"], manifest.SCHEMA_VERSION)
        self.assertEqual(first["artifact"], manifest.ARTIFACT)
        self.assertEqual(first["file_count"], len(first["files"]))
        self.assertEqual(
            [record["path"] for record in first["files"]],
            sorted(record["path"] for record in first["files"]),
        )
        first.pop("generated_at")
        second.pop("generated_at")
        self.assertEqual(first, second)

    def test_build_files_excludes_convert_atomic_write_staging(self) -> None:
        # convert_checkpoint.py stages converted weights as ".converted-*.safetensors"
        # before an atomic link into place. build_files must skip these transient
        # files so a check that races with an in-progress conversion cannot raise a
        # spurious "changed while it was being hashed" error on a file that is never
        # part of the distribution, while a real .safetensors weight is still covered.
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            converted = root / "converted"
            converted.mkdir()
            (converted / "model.safetensors").write_bytes(b"real distributed weights\n")
            staging = converted / ".converted-abc123.safetensors"
            staging.write_bytes(b"in-flight staging bytes\n")

            paths = {record["path"] for record in manifest.build_files(root)}
            self.assertIn("converted/model.safetensors", paths)
            self.assertNotIn("converted/.converted-abc123.safetensors", paths)

            # The staging file present at rest must not register as manifest drift.
            self.run_manifest("generate", root)
            self.run_manifest("check", root)

    def test_check_rejects_missing_and_unexpected_top_level_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            self.run_manifest("generate", root)
            canonical = self.read_payload(root)

            for field in sorted(manifest.MANIFEST_FIELDS):
                with self.subTest(missing=field):
                    self.assert_invalid_mutation(
                        root,
                        canonical,
                        lambda payload, field=field: payload.pop(field),
                        f"missing {field}",
                    )

            self.assert_invalid_mutation(
                root,
                canonical,
                lambda payload: payload.__setitem__("extension", {}),
                "unexpected extension",
            )

    def test_check_rejects_wrong_values_and_types_for_every_top_level_field(self) -> None:
        def replace(field: str, value: Any) -> Mutation:
            return lambda payload: payload.__setitem__(field, value)

        cases = [
            ("schema-version-value", replace("schema_version", 2), "schema_version"),
            ("schema-version-string", replace("schema_version", "1"), "schema_version"),
            ("schema-version-bool", replace("schema_version", True), "schema_version"),
            ("artifact-value", replace("artifact", "other-artifact"), "artifact"),
            ("artifact-type", replace("artifact", 7), "artifact"),
            ("version-value", replace("version", "9.9.9"), "version must match"),
            ("version-type", replace("version", ["1.2.3"]), "version must match"),
            ("generated-at-value", replace("generated_at", "not-a-timestamp"), "generated_at"),
            ("generated-at-type", replace("generated_at", 0), "generated_at"),
            (
                "generated-at-naive",
                replace("generated_at", "2026-07-10T00:00:00"),
                "generated_at must use UTC",
            ),
            (
                "generated-at-non-utc",
                replace("generated_at", "2026-07-10T05:30:00+05:30"),
                "generated_at must use UTC",
            ),
            ("file-count-value", replace("file_count", 99), "file_count does not match"),
            ("file-count-string", replace("file_count", "2"), "file_count"),
            ("file-count-bool", replace("file_count", True), "file_count"),
            ("file-count-negative", replace("file_count", -1), "file_count"),
            ("files-type", replace("files", {}), "files must be an array"),
        ]
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            self.run_manifest("generate", root)
            canonical = self.read_payload(root)
            for name, mutation, message in cases:
                with self.subTest(name=name):
                    self.assert_invalid_mutation(root, canonical, mutation, message)

            self.write_payload(root, [])
            result = self.run_manifest("check", root, expected=2)
            self.assertIn("manifest must be a JSON object", result.stderr)

    def test_check_rejects_malformed_file_records_and_ambiguous_path_sets(self) -> None:
        def mutate_record(field: str, value: Any) -> Mutation:
            return lambda payload: payload["files"][0].__setitem__(field, value)

        def remove_record_field(field: str) -> Mutation:
            return lambda payload: payload["files"][0].pop(field)

        def append_duplicate(payload: dict[str, Any]) -> None:
            payload["files"].append(copy.deepcopy(payload["files"][-1]))
            payload["file_count"] = len(payload["files"])

        cases = [
            ("record-type", lambda payload: payload["files"].__setitem__(0, []), "must be an object"),
            ("missing-path", remove_record_field("path"), "missing path"),
            ("extra-field", mutate_record("extra", True), "unexpected extra"),
            ("path-type", mutate_record("path", 3), ".path must be a non-empty string"),
            ("path-absolute", mutate_record("path", "/tmp/value"), "normalized relative POSIX path"),
            ("path-parent", mutate_record("path", "a/../value"), "normalized relative POSIX path"),
            ("size-string", mutate_record("size_bytes", "1"), "size_bytes"),
            ("size-bool", mutate_record("size_bytes", True), "size_bytes"),
            ("size-negative", mutate_record("size_bytes", -1), "size_bytes"),
            ("digest-type", mutate_record("sha256", 7), "sha256"),
            ("digest-uppercase", mutate_record("sha256", "A" * 64), "sha256"),
            ("digest-length", mutate_record("sha256", "a" * 63), "sha256"),
            ("missing-executable", remove_record_field("executable"), "missing executable"),
            ("executable-type", mutate_record("executable", 1), ".executable must be a boolean"),
            ("file-type", mutate_record("type", "file"), "unexpected type"),
            ("duplicate", append_duplicate, "unique paths"),
            ("unsorted", lambda payload: payload["files"].reverse(), "sorted by path"),
        ]
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            self.run_manifest("generate", root)
            canonical = self.read_payload(root)
            for name, mutation, message in cases:
                with self.subTest(name=name):
                    self.assert_invalid_mutation(root, canonical, mutation, message)

    def test_check_validates_symlink_record_shape(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            (root / "payload-link").symlink_to("payload.txt")
            self.run_manifest("generate", root)
            canonical = self.read_payload(root)
            symlink_index = next(
                index for index, record in enumerate(canonical["files"])
                if record.get("type") == "symlink"
            )

            def mutate_symlink(field: str, value: Any) -> Mutation:
                return lambda payload: payload["files"][symlink_index].__setitem__(field, value)

            def remove_symlink_field(field: str) -> Mutation:
                return lambda payload: payload["files"][symlink_index].pop(field)

            cases = [
                ("missing-target", remove_symlink_field("target"), "missing target"),
                ("target-type", mutate_symlink("target", 4), ".target must be a non-empty string"),
                ("target-empty", mutate_symlink("target", ""), ".target must be a non-empty string"),
                ("wrong-type", mutate_symlink("type", "link"), "fields are invalid"),
                ("missing-type", remove_symlink_field("type"), "unexpected target"),
            ]
            for name, mutation, message in cases:
                with self.subTest(name=name):
                    self.assert_invalid_mutation(root, canonical, mutation, message)

    def test_generation_rejects_absolute_and_escaping_symlink_targets(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            parent = Path(raw_tmp)
            root = parent / "distribution"
            root.mkdir()
            self.create_root(str(root))
            outside = parent / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")

            link = root / "unsafe-link"
            link.symlink_to(outside)
            with self.assertRaisesRegex(manifest.SkillError, "relative path inside"):
                manifest.build_files(root)
            link.unlink()

            nested = root / "nested"
            nested.mkdir()
            escaping = nested / "escaping-link"
            escaping.symlink_to("../../outside.txt")
            with self.assertRaisesRegex(manifest.SkillError, "escapes the distribution root"):
                manifest.build_files(root)
            escaping.unlink()

            internal = nested / "internal-link"
            internal.symlink_to("../payload.txt")
            dangling = nested / "dangling-link"
            dangling.symlink_to("missing/target.txt")
            real_readlink = manifest.os.readlink
            readlink_parent_descriptors: list[int] = []

            def pinned_readlink(path: object, *args: Any, **kwargs: Any) -> str:
                if os.fspath(path) in {"internal-link", "dangling-link"}:
                    parent_descriptor = kwargs.get("dir_fd")
                    self.assertIsInstance(parent_descriptor, int)
                    readlink_parent_descriptors.append(parent_descriptor)
                return real_readlink(path, *args, **kwargs)

            with mock.patch.object(manifest.os, "readlink", side_effect=pinned_readlink):
                records = {record["path"]: record for record in manifest.build_files(root)}

            self.assertEqual(records["nested/internal-link"]["target"], "../payload.txt")
            self.assertEqual(records["nested/dangling-link"]["target"], "missing/target.txt")
            self.assertEqual(len(readlink_parent_descriptors), 2)

    def test_check_rejects_forged_symlink_destinations_even_with_matching_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            nested = root / "nested"
            nested.mkdir()
            (nested / "payload-link").symlink_to("../payload.txt")
            self.run_manifest("generate", root)
            canonical = self.read_payload(root)
            symlink_index = next(
                index for index, record in enumerate(canonical["files"])
                if record.get("type") == "symlink"
            )

            def forge_target(target: str) -> Mutation:
                def mutate(payload: dict[str, Any]) -> None:
                    encoded = target.encode("utf-8")
                    record = payload["files"][symlink_index]
                    record["target"] = target
                    record["size_bytes"] = len(encoded)
                    record["sha256"] = hashlib.sha256(encoded).hexdigest()

                return mutate

            for name, target, message in (
                ("posix-absolute", "/outside.txt", "relative path inside"),
                ("windows-absolute", r"C:\outside.txt", "relative path inside"),
                ("relative-escape", "../../outside.txt", "escapes the distribution root"),
            ):
                with self.subTest(name=name):
                    self.assert_invalid_mutation(
                        root,
                        canonical,
                        forge_target(target),
                        message,
                    )

    def test_final_component_swaps_never_read_external_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            parent = Path(raw_tmp)
            external = parent / "external-secret.txt"
            external.write_text("must-not-be-read\n", encoding="utf-8")

            with self.subTest(surface="VERSION"):
                root = parent / "version-root"
                root.mkdir()
                self.create_root(str(root))
                self.assert_swap_to_external_symlink_rejected(
                    root / "VERSION",
                    external,
                    lambda: manifest.read_version(root),
                )

            with self.subTest(surface="manifest-json"):
                root = parent / "manifest-root"
                root.mkdir()
                self.create_root(str(root))
                manifest_path = root / "MANIFEST.json"
                manifest_path.write_text("{}\n", encoding="utf-8")
                self.assert_swap_to_external_symlink_rejected(
                    manifest_path,
                    external,
                    lambda: manifest.read_manifest(manifest_path),
                )

            with self.subTest(surface="distributed-file"):
                root = parent / "file-root"
                root.mkdir()
                self.create_root(str(root))
                self.assert_swap_to_external_symlink_rejected(
                    root / "payload.txt",
                    external,
                    lambda: manifest.build_files(root),
                )

    def test_regular_file_reads_fail_closed_without_nofollow_support(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            with (
                mock.patch.object(manifest, "NOFOLLOW_FLAG", None),
                self.assertRaisesRegex(manifest.SkillError, "does not provide O_NOFOLLOW"),
            ):
                manifest.read_version(root)

    @unittest.skipIf(sys.platform == "win32", "POSIX executable-bit contract")
    def test_check_detects_executable_permission_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            payload_path = root / "payload.txt"
            payload_path.chmod(0o644)
            self.run_manifest("generate", root)
            generated = self.read_payload(root)
            payload_record = next(
                record for record in generated["files"] if record["path"] == "payload.txt"
            )
            self.assertIs(payload_record["executable"], False)

            payload_path.chmod(0o755)
            changed = self.run_manifest("check", root, expected=1)

            self.assertIn("payload.txt", changed.stdout)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation requires POSIX")
    def test_manifest_and_installer_reject_special_nodes(self) -> None:
        import install_skill

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            fifo = root / ".dev.vars"
            os.mkfifo(fifo)

            with self.assertRaisesRegex(manifest.SkillError, "unsupported FIFO"):
                manifest.build_files(root)
            with self.assertRaisesRegex(install_skill.SkillError, "unsupported"):
                install_skill.tree_signature(root)

            fifo.unlink()
            ignored_fifo = root / "ignored.pyc"
            os.mkfifo(ignored_fifo)
            with self.assertRaisesRegex(manifest.SkillError, "unsupported FIFO"):
                manifest.build_files(root)
            with self.assertRaisesRegex(install_skill.SkillError, "unsupported"):
                install_skill.tree_signature(root)

            ignored_fifo.unlink()
            socket_path = root / "unsupported.sock"
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                try:
                    unix_socket.bind(str(socket_path))
                except PermissionError:
                    self.skipTest("sandbox forbids creating Unix-domain socket fixtures")
                with self.assertRaisesRegex(manifest.SkillError, "unsupported socket"):
                    manifest.build_files(root)
                with self.assertRaisesRegex(install_skill.SkillError, "unsupported"):
                    install_skill.tree_signature(root)
            finally:
                unix_socket.close()

    def test_ancestor_directory_swap_never_reads_external_tree_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            parent = Path(raw_tmp)
            root = parent / "distribution"
            root.mkdir()
            self.create_root(str(root))
            nested = root / "nested"
            nested.mkdir()
            (nested / "inside.txt").write_text("inside\n", encoding="utf-8")
            parked = root / ".nested.original"

            external = parent / "external"
            external.mkdir()
            external_secret = external / "external-secret.txt"
            external_secret.write_text("must-not-be-read\n", encoding="utf-8")
            external_metadata = external_secret.stat()
            root_metadata = root.stat()
            real_open = manifest.os.open
            real_read = manifest.os.read
            swapped = False
            external_reads = 0

            def swapping_open(path: object, flags: int, *args: Any, **kwargs: Any) -> int:
                nonlocal swapped
                parent_descriptor = kwargs.get("dir_fd")
                parent_matches = False
                if isinstance(parent_descriptor, int):
                    descriptor_metadata = os.fstat(parent_descriptor)
                    parent_matches = (descriptor_metadata.st_dev, descriptor_metadata.st_ino) == (
                        root_metadata.st_dev,
                        root_metadata.st_ino,
                    )
                if os.fspath(path) == "nested" and parent_matches and not swapped:
                    self.assertTrue(flags & int(manifest.NOFOLLOW_FLAG))
                    self.assertTrue(flags & int(manifest.DIRECTORY_FLAG))
                    nested.replace(parked)
                    nested.symlink_to(external, target_is_directory=True)
                    swapped = True
                return real_open(path, flags, *args, **kwargs)

            def guarded_read(descriptor: int, size: int) -> bytes:
                nonlocal external_reads
                metadata = os.fstat(descriptor)
                if (metadata.st_dev, metadata.st_ino) == (
                    external_metadata.st_dev,
                    external_metadata.st_ino,
                ):
                    external_reads += 1
                    raise AssertionError("external tree content must not be read")
                return real_read(descriptor, size)

            with (
                mock.patch.object(manifest.os, "open", side_effect=swapping_open),
                mock.patch.object(manifest.os, "read", side_effect=guarded_read),
                self.assertRaisesRegex(manifest.SkillError, "without following symlinks"),
            ):
                manifest.build_files(root)

            self.assertTrue(swapped)
            self.assertEqual(external_reads, 0)

    def test_check_never_rewrites_clean_or_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            self.run_manifest("generate", root)
            path = root / "MANIFEST.json"

            clean_bytes = path.read_bytes()
            self.run_manifest("check", root)
            self.assertEqual(path.read_bytes(), clean_bytes)

            payload = self.read_payload(root)
            payload["artifact"] = "forged-artifact"
            self.write_payload(root, payload)
            invalid_bytes = path.read_bytes()
            self.run_manifest("check", root, expected=2)
            self.assertEqual(path.read_bytes(), invalid_bytes)

    def test_check_rejects_manifest_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = self.create_root(raw_tmp)
            self.run_manifest("generate", root)
            path = root / "MANIFEST.json"
            target = root / "actual-manifest.json"
            path.replace(target)
            path.symlink_to(target.name)

            result = self.run_manifest("check", root, expected=2)

            self.assertIn("manifest must not be a symlink", result.stderr)
            self.assertTrue(path.is_symlink())
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
