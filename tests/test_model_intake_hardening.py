"""Integrity and resource-boundary regressions for static model intake."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import inspect_model  # noqa: E402
from _common import SkillError  # noqa: E402


def safetensors_bytes(records: dict[str, dict[str, object]], payload_size: int) -> bytes:
    header = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + (b"\0" * payload_size)


def proto_varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def proto_field_varint(field: int, value: int) -> bytes:
    return proto_varint(field << 3) + proto_varint(value)


def proto_field_bytes(field: int, value: bytes | str) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return proto_varint((field << 3) | 2) + proto_varint(len(raw)) + raw


class ModelIntakeHardeningTests(unittest.TestCase):
    def run_inspector(self, model: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "inspect_model.py"), str(model), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_plus_corrupt_shard_is_never_treated_as_an_intact_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            root.mkdir()
            (root / "config.json").write_text(
                json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
                encoding="utf-8",
            )
            first = "model-00001-of-00002.safetensors"
            second = "model-00002-of-00002.safetensors"
            (root / first).write_bytes(safetensors_bytes({
                "good.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }, 4))
            (root / second).write_bytes(safetensors_bytes({
                "bad.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }, 2))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 8},
                "weight_map": {"good.weight": first, "bad.weight": second},
            }), encoding="utf-8")

            completed = self.run_inspector(root)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)

            self.assertEqual([tensor["key"] for tensor in report["tensors"]], ["good.weight"])
            self.assertFalse(report["tensor_summary"]["integrity_ok"])
            self.assertTrue(
                any("bad.weight" in error or second in error for error in report["tensor_summary"]["errors"]),
                report["tensor_summary"]["errors"],
            )
            self.assertTrue(
                any("integrity" in blocker for blocker in report["recommendation_blockers"]),
                report["recommendation_blockers"],
            )
            self.assertIsNone(report["recommended_family"])

    def test_safetensors_rejects_duplicate_invalid_and_partial_records(self) -> None:
        cases: dict[str, bytes] = {}
        duplicate = (
            b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            b'"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        )
        cases["duplicate JSON key"] = struct.pack("<Q", len(duplicate)) + duplicate + b"\0" * 4
        cases["invalid dtype"] = safetensors_bytes({
            "x": {"dtype": "NOT_A_DTYPE", "shape": [1], "data_offsets": [0, 4]},
        }, 4)
        cases["invalid shape"] = safetensors_bytes({
            "x": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 0]},
        }, 0)
        cases["out-of-range data_offsets"] = safetensors_bytes({
            "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }, 2)
        cases["truncated safetensors header"] = struct.pack("<Q", 50) + b"{}"

        for expected, payload in cases.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as raw_tmp:
                path = Path(raw_tmp) / "model.safetensors"
                path.write_bytes(payload)
                with self.assertRaisesRegex(SkillError, expected):
                    inspect_model.read_safetensors_header(path)

    def test_shard_index_must_match_every_tensor_and_shard(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            first = root / "a.safetensors"
            second = root / "b.safetensors"
            first.write_bytes(safetensors_bytes({
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }, 4))
            second.write_bytes(safetensors_bytes({
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }, 4))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 8},
                "weight_map": {"a": "b.safetensors", "b": "a.safetensors"},
            }), encoding="utf-8")

            tensors, _, errors = inspect_model.inspect_tensors(root)

            self.assertEqual(len(tensors), 2)
            self.assertTrue(any("does not contain" in error or "wrong shard" in error for error in errors), errors)

    def test_onnx_and_gguf_structural_counts_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            onnx = root / "many-fields.onnx"
            onnx.write_bytes(proto_field_varint(1, 8) + proto_field_varint(5, 1))
            with mock.patch.object(inspect_model, "MAX_ONNX_FIELDS_PER_MESSAGE", 1):
                manifest = inspect_model.inspect_onnx_file(onnx, root)
            self.assertTrue(any("field-count limit" in error for error in manifest.get("errors", [])), manifest)

            gguf = root / "too-many.gguf"
            gguf.write_bytes(
                b"GGUF"
                + (3).to_bytes(4, "little")
                + (inspect_model.MAX_GGUF_TENSORS + 1).to_bytes(8, "little")
                + (0).to_bytes(8, "little")
            )
            manifest = inspect_model.inspect_gguf_file(gguf, root)
            self.assertTrue(any("tensor-count limit" in error for error in manifest.get("errors", [])), manifest)

    def test_onnx_external_data_is_reported_but_never_opened(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            outside = root / "outside.bin"
            outside.write_bytes(b"must-not-be-read")
            external_entry = proto_field_bytes(1, "location") + proto_field_bytes(2, "../outside.bin")
            tensor = (
                proto_field_bytes(1, proto_varint(1))
                + proto_field_varint(2, 1)
                + proto_field_bytes(8, "weight")
                + proto_field_bytes(13, external_entry)
                + proto_field_varint(14, 1)
            )
            graph = proto_field_bytes(2, "external_graph") + proto_field_bytes(5, tensor)
            model = proto_field_varint(1, 8) + proto_field_bytes(7, graph)
            onnx = root / "model.onnx"
            onnx.write_bytes(model)
            opened: list[Path] = []
            original_read = inspect_model.read_regular_bytes

            def recording_read(path: Path, limit: int, *, label: str | None = None) -> bytes:
                opened.append(path)
                return original_read(path, limit, label=label)

            with mock.patch.object(inspect_model, "read_regular_bytes", side_effect=recording_read):
                manifest = inspect_model.inspect_onnx_file(onnx, root)

            self.assertEqual(opened, [onnx])
            self.assertTrue(
                any("unsafe external data location" in hold for hold in manifest["hold_conditions"]),
                manifest["hold_conditions"],
            )

    def test_source_format_aggregate_artifact_and_byte_budgets_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "a.onnx").write_bytes(proto_field_varint(1, 8))
            (root / "b.onnx").write_bytes(proto_field_varint(1, 8))
            with mock.patch.object(inspect_model, "MAX_SOURCE_FORMAT_ARTIFACTS", 1):
                manifests, errors = inspect_model.inspect_source_formats(root)
            self.assertEqual(manifests, [])
            self.assertTrue(any("artifact-count limit" in error for error in errors), errors)

            with mock.patch.object(inspect_model, "MAX_SOURCE_FORMAT_TOTAL_BYTES", 1):
                manifests, errors = inspect_model.inspect_source_formats(root)
            self.assertEqual(manifests, [])
            self.assertTrue(any("aggregate byte budget" in error for error in errors), errors)

    @unittest.skipIf(os.name == "nt", "POSIX no-follow and FIFO contract")
    def test_static_reads_reject_external_symlinks_special_files_and_path_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            outside = root / "outside.onnx"
            outside.write_bytes(b"outside-secret")
            link = root / "link.onnx"
            link.symlink_to(outside)
            with self.assertRaisesRegex(SkillError, "symlink"):
                inspect_model.read_regular_bytes(link, 100, label="link.onnx")

            fifo = root / "special.onnx"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(SkillError, "regular file"):
                inspect_model.read_regular_bytes(fifo, 100, label="special.onnx")

            target = root / "target.onnx"
            replacement = root / "replacement.onnx"
            target.write_bytes(b"trusted")
            replacement.write_bytes(b"external")
            original_read_exact = inspect_model._read_exact

            def swap_after_read(descriptor: int, size: int, label: str) -> bytes:
                data = original_read_exact(descriptor, size, label)
                os.replace(replacement, target)
                return data

            with mock.patch.object(inspect_model, "_read_exact", side_effect=swap_after_read):
                with self.assertRaisesRegex(SkillError, "changed while"):
                    inspect_model.read_regular_bytes(target, 100, label="target.onnx")

    def test_huggingface_tree_quotas_run_before_snapshot_download(self) -> None:
        api = SimpleNamespace(
            model_info=lambda **_: SimpleNamespace(sha="a" * 40),
            list_repo_tree=lambda **_: [SimpleNamespace(path="config.json", type="file", size=6)],
        )
        snapshot_called = False

        def snapshot_download(**_: object) -> str:
            nonlocal snapshot_called
            snapshot_called = True
            raise AssertionError("snapshot download must not run after failed preflight")

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.HfApi = lambda: api  # type: ignore[attr-defined]
        fake_module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
            with mock.patch.object(inspect_model, "MAX_HF_TOTAL_DOWNLOAD_BYTES", 5):
                with self.assertRaisesRegex(SkillError, "aggregate download quota"):
                    inspect_model.resolve_model("org/quota-fixture", True, "main", False)
        self.assertFalse(snapshot_called)

    def test_huggingface_download_uses_resolved_commit_and_reports_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            local = Path(raw_tmp) / "snapshot"
            local.mkdir()
            api = SimpleNamespace(
                model_info=lambda **_: SimpleNamespace(sha="b" * 40),
                list_repo_tree=lambda **_: [SimpleNamespace(path="config.json", type="file", size=12)],
            )
            calls: list[dict[str, object]] = []

            def snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs)
                return str(local)

            fake_module = types.ModuleType("huggingface_hub")
            fake_module.HfApi = lambda: api  # type: ignore[attr-defined]
            fake_module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                resolved, source = inspect_model.resolve_model(
                    "org/success-fixture",
                    True,
                    "main",
                    False,
                )

            self.assertEqual(resolved, local.resolve())
            self.assertEqual(calls[0]["revision"], "b" * 40)
            self.assertEqual(source["revision"], "b" * 40)
            self.assertEqual(source["requested_revision"], "main")
            self.assertEqual(source["preflight"]["selected_bytes"], 12)
            self.assertNotIn("local_cache_path", source)

    def test_local_absolute_paths_require_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            model = Path(raw_tmp) / "private-model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "unknown"}), encoding="utf-8")

            portable = self.run_inspector(model)
            self.assertEqual(portable.returncode, 0, portable.stderr)
            self.assertNotIn(str(model), portable.stdout)
            report = json.loads(portable.stdout)
            self.assertEqual(report["local_path"], model.name)
            self.assertEqual(report["source"]["input"], model.name)

            explicit = self.run_inspector(model, "--include-local-paths")
            self.assertEqual(explicit.returncode, 0, explicit.stderr)
            self.assertIn(str(model), explicit.stdout)


if __name__ == "__main__":
    unittest.main()
