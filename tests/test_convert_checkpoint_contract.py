"""Contracts for explicit safetensors-to-MLX checkpoint conversion."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
SCRIPT = SCRIPTS / "convert_checkpoint.py"
VALIDATOR = SCRIPTS / "validate_weight_map.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_checkpoint as converter


def mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core as mx; value=mx.array([1], dtype=mx.int32); mx.eval(value)",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return probe.returncode == 0


HAS_MLX = mlx_runtime_available()


def resolved_temp() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory()


def root_path(raw: str) -> Path:
    # macOS exposes /var as a symlink. Resolve the test-owned temporary root so
    # no-follow component checks exercise the fixture, not the OS compatibility link.
    return Path(raw).resolve()


def source_arrays() -> dict[str, np.ndarray]:
    return {
        "fused_qkv": np.arange(12, dtype=np.float32).reshape(6, 2),
        "merge_a": np.array([[10.0, 11.0]], dtype=np.float32),
        "merge_b": np.array([[12.0, 13.0], [14.0, 15.0]], dtype=np.float32),
        "rename_source": np.array([1.0, 2.0], dtype=np.float32),
        "transpose_source": np.arange(6, dtype=np.float32).reshape(2, 3),
        "unused_source": np.array([99.0], dtype=np.float32),
    }


def complete_mapping(*, draft: bool = False, wrong_shape: bool = False) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schema_version": 2,
        "draft": draft,
        "dtype_policy": "keep",
        "entries": [
            {
                "source": "rename_source",
                "source_shape": [2],
                "target": "renamed",
                "target_shape": [2],
                "transforms": [{"op": "rename"}],
            },
            {
                "source": "transpose_source",
                "source_shape": [2, 3],
                "target": "transposed",
                "target_shape": [2, 3] if wrong_shape else [3, 2],
                "transforms": [{"op": "transpose", "axes": [1, 0]}],
            },
            {
                "source": "fused_qkv",
                "source_shape": [6, 2],
                "targets": [
                    {"target": "q_proj", "shape": [2, 2]},
                    {"target": "k_proj", "shape": [2, 2]},
                    {"target": "v_proj", "shape": [2, 2]},
                ],
                "transforms": [{"op": "split", "axis": 0, "sizes": [2, 2, 2]}],
            },
            {
                "sources": [
                    {"source": "merge_a", "shape": [1, 2]},
                    {"source": "merge_b", "shape": [2, 2]},
                ],
                "target": "merged",
                "target_shape": [3, 2],
                "transforms": [
                    {"op": "merge", "axis": 0},
                    {"op": "cast", "dtype": "f16"},
                ],
            },
        ],
        "ignore": [
            {"source": "unused_source", "reason": "fixture-only source not used by the target graph"}
        ],
        "unresolved": [],
    }
    return mapping


def write_checkpoint(path: Path, arrays: dict[str, np.ndarray] | None = None) -> None:
    path.mkdir()
    converter.write_safetensors_pure(path / "model.safetensors", arrays or source_arrays())


def write_mapping(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_converter(
    *args: object,
    pure_only: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if pure_only:
        environment[converter.OPTIONAL_LIBS_ENV] = "1"
    else:
        environment.pop(converter.OPTIONAL_LIBS_ENV, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(value) for value in args)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def run_validator(
    source: Path,
    target: Path,
    mapping: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--source", str(source),
            "--target", str(target),
            "--mapping", str(mapping),
            "--output", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def output_arrays(path: Path) -> dict[str, np.ndarray]:
    info = converter.read_safetensors_header(path)
    return {
        key: converter.read_safetensors_tensor_pure(info, key)
        for key in sorted(info["tensors"])
    }


class ConvertCheckpointDependencyFreeContractTests(unittest.TestCase):
    def test_happy_path_rename_transpose_split_merge_and_cast(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            write_checkpoint(source)
            write_mapping(mapping, complete_mapping())

            completed = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", output,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(completed.stderr, "")
            arrays = output_arrays(output / "model.safetensors")
            np.testing.assert_array_equal(arrays["renamed"], np.array([1.0, 2.0], dtype=np.float32))
            np.testing.assert_array_equal(
                arrays["transposed"],
                np.arange(6, dtype=np.float32).reshape(2, 3).T,
            )
            np.testing.assert_array_equal(
                arrays["q_proj"],
                np.arange(12, dtype=np.float32).reshape(6, 2)[:2],
            )
            np.testing.assert_array_equal(
                arrays["k_proj"],
                np.arange(12, dtype=np.float32).reshape(6, 2)[2:4],
            )
            np.testing.assert_array_equal(
                arrays["v_proj"],
                np.arange(12, dtype=np.float32).reshape(6, 2)[4:],
            )
            np.testing.assert_array_equal(
                arrays["merged"],
                np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], dtype=np.float16),
            )
            report = json.loads((output / "conversion-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["counts"]["source_tensors"], 6)
            self.assertEqual(report["counts"]["target_tensors"], 6)
            self.assertEqual(
                report["transforms_applied"],
                {"cast": 1, "merge": 1, "rename": 1, "split": 1, "transpose": 1},
            )
            self.assertEqual(report["outputs"]["weights"]["writer"], "pure-python-safetensors")
            self.assertRegex(report["inputs"]["mapping"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(report["outputs"]["weights"]["sha256"], r"^[0-9a-f]{64}$")

    def test_unmapped_source_tensor_fails_closed(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            write_checkpoint(source)
            value = complete_mapping()
            value["ignore"] = []
            write_mapping(mapping, value)

            completed = run_converter("--source", source, "--mapping", mapping, "--output", output)

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("source tensors are unmapped and not ignored", completed.stderr)
            self.assertFalse(output.exists())

    def test_global_and_per_tensor_dtype_policies_are_explicit(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            write_checkpoint(source, {
                "global": np.array([1.25], dtype=np.float64),
                "half": np.array([2.5], dtype=np.float32),
                "single": np.array([3.75], dtype=np.float16),
            })
            mapping = root / "WEIGHT_MAP.json"
            write_mapping(mapping, {
                "schema_version": 2,
                "dtype_policy": "bf16",
                "entries": [
                    {
                        "source": "global",
                        "source_shape": [1],
                        "target": "global",
                        "target_shape": [1],
                        "transforms": [{"op": "rename"}],
                    },
                    {
                        "source": "half",
                        "source_shape": [1],
                        "target": "half",
                        "target_shape": [1],
                        "dtype_policy": "f16",
                        "transforms": [{"op": "rename"}],
                    },
                    {
                        "source": "single",
                        "source_shape": [1],
                        "target": "single",
                        "target_shape": [1],
                        "dtype_policy": "f32",
                        "transforms": [{"op": "rename"}],
                    },
                ],
                "ignore": [],
                "unresolved": [],
            })
            output = root / "converted"

            completed = run_converter("--source", source, "--mapping", mapping, "--output", output)

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads((output / "target-manifest.json").read_text(encoding="utf-8"))
            dtypes = {item["key"]: item["dtype"] for item in manifest["tensors"]}
            self.assertEqual(dtypes, {"global": "BF16", "half": "F16", "single": "F32"})
            report = json.loads((output / "conversion-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["dtype_policy"]["global"], "bf16")
            self.assertEqual(
                report["dtype_policy"]["targets"],
                {"global": "bf16", "half": "f16", "single": "f32"},
            )

    def test_missing_index_shard_fails_closed(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            source.mkdir()
            converter.write_safetensors_pure(
                source / "model-00001-of-00002.safetensors",
                {"present": np.array([1.0], dtype=np.float32)},
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps({
                    "metadata": {"total_size": 8},
                    "weight_map": {
                        "present": "model-00001-of-00002.safetensors",
                        "missing": "model-00002-of-00002.safetensors",
                    },
                }),
                encoding="utf-8",
            )
            mapping = root / "WEIGHT_MAP.json"
            write_mapping(mapping, {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "present",
                    "source_shape": [1],
                    "target": "present",
                    "target_shape": [1],
                    "transforms": [{"op": "rename"}],
                }],
                "ignore": [],
                "unresolved": [],
            })

            completed = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", root / "converted",
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("references missing shards", completed.stderr)

    def test_post_transform_shape_mismatch_fails_closed(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            write_checkpoint(source)
            write_mapping(mapping, complete_mapping(wrong_shape=True))

            completed = run_converter("--source", source, "--mapping", mapping, "--output", output)

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("post-transform shape mismatch for transposed", completed.stderr)
            self.assertFalse(output.exists())

    def test_draft_map_is_refused(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            write_checkpoint(source)
            write_mapping(mapping, complete_mapping(draft=True))

            completed = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", root / "converted",
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("refusing to convert with a draft WEIGHT_MAP", completed.stderr)

    def test_hostile_safetensors_headers_fail_closed(self) -> None:
        cases: dict[str, bytes] = {}
        cases["oversized"] = struct.pack("<Q", converter.MAX_HEADER_BYTES + 1)
        overlapping_header = {
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        raw_overlap = json.dumps(overlapping_header, separators=(",", ":")).encode("utf-8")
        cases["overlapping"] = struct.pack("<Q", len(raw_overlap)) + raw_overlap + b"\x00" * 8
        unknown_header = {
            "a": {"dtype": "F13", "shape": [1], "data_offsets": [0, 4]},
        }
        raw_unknown = json.dumps(unknown_header, separators=(",", ":")).encode("utf-8")
        cases["unknown"] = struct.pack("<Q", len(raw_unknown)) + raw_unknown + b"\x00" * 4

        for name, payload in cases.items():
            with self.subTest(name=name), resolved_temp() as raw:
                root = root_path(raw)
                source = root / "source"
                source.mkdir()
                (source / "model.safetensors").write_bytes(payload)
                mapping = root / "WEIGHT_MAP.json"
                write_mapping(mapping, {
                    "schema_version": 2,
                    "dtype_policy": "keep",
                    "entries": [{
                        "source": "a",
                        "source_shape": [1],
                        "target": "a",
                        "target_shape": [1],
                        "transforms": [{"op": "rename"}],
                    }],
                    "ignore": [],
                    "unresolved": [],
                })

                completed = run_converter(
                    "--source", source,
                    "--mapping", mapping,
                    "--output", root / "converted",
                )

                self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
                if name == "oversized":
                    self.assertIn("suspicious safetensors header length", completed.stderr)
                elif name == "overlapping":
                    self.assertIn("overlapping or non-contiguous offsets", completed.stderr)
                else:
                    self.assertIn("unknown dtype", completed.stderr)

    def test_emitted_target_manifest_passes_weight_map_validator(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            source_manifest = root / "source-manifest.json"
            arrays = source_arrays()
            write_checkpoint(source, arrays)
            write_mapping(mapping, complete_mapping())
            source_manifest.write_text(
                json.dumps({
                    "tensors": [
                        {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in sorted(arrays.items())
                    ]
                }),
                encoding="utf-8",
            )
            converted = run_converter("--source", source, "--mapping", mapping, "--output", output)
            self.assertEqual(converted.returncode, 0, converted.stdout + converted.stderr)
            report_path = root / "validation.json"

            validated = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--source", str(source_manifest),
                    "--target", str(output / "target-manifest.json"),
                    "--mapping", str(mapping),
                    "--output", str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["source_tensors"], 6)
            self.assertEqual(report["target_tensors"], 6)

    def test_all_transform_shapes_validate_and_convert_equivalently(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            arrays = {
                "cast": np.arange(2, dtype=np.float32),
                "merge_a": np.arange(2, dtype=np.float32).reshape(1, 2),
                "merge_b": np.arange(4, dtype=np.float32).reshape(2, 2),
                "permute": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "rename": np.arange(2, dtype=np.float32),
                "reshape": np.arange(6, dtype=np.float32).reshape(2, 3),
                "slice": np.arange(12, dtype=np.float32).reshape(2, 6),
                "split": np.arange(8, dtype=np.float32).reshape(4, 2),
                "squeeze": np.arange(3, dtype=np.float32).reshape(1, 3),
                "transpose": np.arange(6, dtype=np.float32).reshape(2, 3),
                "unsqueeze": np.arange(3, dtype=np.float32),
            }
            write_checkpoint(source, arrays)
            mapping_value = {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [
                    {"source": "rename", "source_shape": [2], "target": "renamed", "target_shape": [2], "transforms": [{"op": "rename"}]},
                    {"source": "transpose", "source_shape": [2, 3], "target": "transposed", "target_shape": [3, 2], "transforms": [{"op": "transpose", "axes": [1, 0]}]},
                    {"source": "permute", "source_shape": [2, 3, 4], "target": "permuted", "target_shape": [4, 2, 3], "transforms": [{"op": "permute", "axes": [2, 0, 1]}]},
                    {"source": "reshape", "source_shape": [2, 3], "target": "reshaped", "target_shape": [3, 2], "transforms": [{"op": "reshape", "shape": [-1, 2]}]},
                    {"source": "slice", "source_shape": [2, 6], "target": "sliced", "target_shape": [2, 2], "transforms": [{"op": "slice", "axis": -1, "start": -5, "end": -1, "step": 2}]},
                    {"source": "squeeze", "source_shape": [1, 3], "target": "squeezed", "target_shape": [3], "transforms": [{"op": "squeeze", "axis": -2}]},
                    {"source": "unsqueeze", "source_shape": [3], "target": "unsqueezed", "target_shape": [1, 3], "transforms": [{"op": "unsqueeze", "axis": -2}]},
                    {"source": "cast", "source_shape": [2], "target": "casted", "target_shape": [2], "transforms": [{"op": "cast", "dtype": "f16"}]},
                    {
                        "source": "split",
                        "source_shape": [4, 2],
                        "targets": [
                            {"target": "split_a", "shape": [1, 2]},
                            {"target": "split_b", "shape": [3, 2]},
                        ],
                        "transforms": [{"op": "split", "axis": -2, "sizes": [1, 3]}],
                    },
                    {
                        "sources": [
                            {"source": "merge_a", "shape": [1, 2]},
                            {"source": "merge_b", "shape": [2, 2]},
                        ],
                        "target": "merged",
                        "target_shape": [3, 2],
                        "transforms": [{"op": "merge", "axis": -2}],
                    },
                ],
                "ignore": [],
                "unresolved": [],
            }
            mapping = root / "WEIGHT_MAP.json"
            write_mapping(mapping, mapping_value)
            source_manifest = root / "source-manifest.json"
            source_manifest.write_text(
                json.dumps({"tensors": [
                    {"key": key, "shape": list(value.shape)}
                    for key, value in sorted(arrays.items())
                ]}),
                encoding="utf-8",
            )
            expected_shapes = {
                "casted": [2],
                "merged": [3, 2],
                "permuted": [4, 2, 3],
                "renamed": [2],
                "reshaped": [3, 2],
                "sliced": [2, 2],
                "split_a": [1, 2],
                "split_b": [3, 2],
                "squeezed": [3],
                "transposed": [3, 2],
                "unsqueezed": [1, 3],
            }
            target_manifest = root / "target-manifest.json"
            target_manifest.write_text(
                json.dumps({"tensors": [
                    {"key": key, "shape": value}
                    for key, value in sorted(expected_shapes.items())
                ]}),
                encoding="utf-8",
            )
            validation_report = root / "validation.json"
            validated = run_validator(
                source_manifest, target_manifest, mapping, validation_report
            )
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)

            output = root / "converted"
            converted = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", output,
            )
            self.assertEqual(converted.returncode, 0, converted.stdout + converted.stderr)
            validation = json.loads(validation_report.read_text(encoding="utf-8"))
            validated_shapes = {
                check["target"]: check["transformed_shape"]
                for check in validation["checks"]
            }
            emitted = json.loads(
                (output / "target-manifest.json").read_text(encoding="utf-8")
            )
            converted_shapes = {
                item["key"]: item["shape"] for item in emitted["tensors"]
            }
            self.assertEqual(validated_shapes, expected_shapes)
            self.assertEqual(converted_shapes, expected_shapes)

    def test_out_of_range_split_axis_and_missing_transpose_axes_fail_both_tools(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            arrays = {"x": np.arange(8, dtype=np.float32).reshape(4, 2)}
            write_checkpoint(source, arrays)
            source_manifest = root / "source-manifest.json"
            source_manifest.write_text(
                json.dumps({"tensors": [{"key": "x", "shape": [4, 2]}]}),
                encoding="utf-8",
            )
            target_manifest = root / "target-manifest.json"
            target_manifest.write_text(
                json.dumps({"tensors": [
                    {"key": "a", "shape": [2, 2]},
                    {"key": "b", "shape": [2, 2]},
                ]}),
                encoding="utf-8",
            )
            mapping = root / "WEIGHT_MAP.json"
            invalid_axis = {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "x",
                    "source_shape": [4, 2],
                    "targets": [
                        {"target": "a", "shape": [2, 2]},
                        {"target": "b", "shape": [2, 2]},
                    ],
                    "transforms": [{"op": "split", "axis": 2, "sizes": [2, 2]}],
                }],
                "ignore": [],
                "unresolved": [],
            }
            write_mapping(mapping, invalid_axis)
            report_path = root / "validation.json"
            validated = run_validator(source_manifest, target_manifest, mapping, report_path)
            converted = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", root / "converted-axis",
            )
            self.assertEqual(validated.returncode, 1, validated.stdout + validated.stderr)
            self.assertEqual(converted.returncode, 2, converted.stdout + converted.stderr)
            validation_errors = "\n".join(
                json.loads(report_path.read_text(encoding="utf-8"))["errors"]
            )
            self.assertIn("axis 2 is out of range for rank 2", validation_errors)
            self.assertIn("axis 2 is out of range for rank 2", converted.stderr)

            missing_axes = {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "x",
                    "source_shape": [4, 2],
                    "target": "a",
                    "target_shape": [2, 4],
                    "transforms": [{"op": "transpose"}],
                }],
                "ignore": [],
                "unresolved": [],
            }
            write_mapping(mapping, missing_axes)
            missing_target = root / "missing-target.json"
            missing_target.write_text(
                json.dumps({"tensors": [{"key": "a", "shape": [2, 4]}]}),
                encoding="utf-8",
            )
            validated = run_validator(
                source_manifest, missing_target, mapping, root / "missing-validation.json"
            )
            converted = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", root / "converted-transpose",
            )
            self.assertEqual(validated.returncode, 2, validated.stdout + validated.stderr)
            self.assertEqual(converted.returncode, 2, converted.stdout + converted.stderr)
            self.assertIn("missing=['axes']", validated.stderr)
            self.assertIn("missing=['axes']", converted.stderr)

    def test_mapping_digest_attests_the_exact_bytes_used_after_path_swap(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            write_checkpoint(source)
            write_mapping(mapping, complete_mapping())
            used_bytes = mapping.read_bytes()
            replacement = {**complete_mapping(), "draft": False}
            replacement_bytes = (json.dumps(replacement, separators=(",", ":")) + "\n").encode()
            original_parser = converter.parse_conversion_map

            def swap_after_parse(value: object) -> dict[str, object]:
                parsed = original_parser(value)
                mapping.write_bytes(replacement_bytes)
                return parsed

            with mock.patch.dict(
                os.environ,
                {converter.OPTIONAL_LIBS_ENV: "1"},
            ), mock.patch.object(
                converter,
                "parse_conversion_map",
                side_effect=swap_after_parse,
            ):
                report = converter.run_conversion(source, mapping, output)

            attested = report["inputs"]["mapping"]
            self.assertEqual(attested["size_bytes"], len(used_bytes))
            self.assertEqual(attested["sha256"], hashlib.sha256(used_bytes).hexdigest())
            self.assertNotEqual(
                attested["sha256"], hashlib.sha256(replacement_bytes).hexdigest()
            )

    def test_shard_index_digest_attests_the_exact_bytes_used_after_path_swap(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            source.mkdir()
            shard = source / "model-00001-of-00001.safetensors"
            converter.write_safetensors_pure(
                shard,
                {"x": np.arange(2, dtype=np.float32)},
            )
            index = source / "model.safetensors.index.json"
            index_value = {
                "metadata": {"total_size": 8},
                "weight_map": {"x": shard.name},
            }
            index.write_text(
                json.dumps(index_value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            used_bytes = index.read_bytes()
            replacement_bytes = (
                json.dumps(index_value, separators=(",", ":")) + "\n"
            ).encode()
            original_parser = converter.parse_strict_json

            def swap_index_after_parse(raw_bytes: bytes, *, label: str) -> object:
                parsed = original_parser(raw_bytes, label=label)
                if label == "safetensors shard index":
                    index.write_bytes(replacement_bytes)
                return parsed

            with mock.patch.object(
                converter,
                "parse_strict_json",
                side_effect=swap_index_after_parse,
            ):
                checkpoint = converter.discover_checkpoint(source)

            index_digest = next(
                item
                for item in checkpoint["input_digests"]
                if item["name"] == index.name
            )
            self.assertEqual(
                index_digest["sha256"], hashlib.sha256(used_bytes).hexdigest()
            )
            self.assertNotEqual(
                index_digest["sha256"], hashlib.sha256(replacement_bytes).hexdigest()
            )

    def test_aggregate_memory_bound_fails_before_load_and_normal_limit_succeeds(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            write_checkpoint(source, {"x": np.arange(8, dtype=np.float32)})
            mapping = root / "WEIGHT_MAP.json"
            write_mapping(mapping, {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "x",
                    "source_shape": [8],
                    "target": "x",
                    "target_shape": [8],
                    "transforms": [{"op": "rename"}],
                }],
                "ignore": [],
                "unresolved": [],
            })
            with mock.patch.object(
                converter,
                "load_tensor",
                side_effect=AssertionError("tensor allocation was attempted"),
            ):
                with self.assertRaisesRegex(
                    converter.SkillError,
                    "aggregate materialization memory bound exceeded before allocation",
                ):
                    converter.run_conversion(
                        source,
                        mapping,
                        root / "too-small",
                        max_working_set_bytes=63,
                    )
            self.assertFalse((root / "too-small").exists())

            with mock.patch.dict(
                os.environ,
                {converter.OPTIONAL_LIBS_ENV: "1"},
            ):
                report = converter.run_conversion(
                    source,
                    mapping,
                    root / "under-bound",
                    max_working_set_bytes=128,
                )
            self.assertEqual(report["memory_budget"]["aggregate_materialized_bytes"], 64)
            self.assertEqual(report["memory_budget"]["estimated_peak_bytes"], 128)

    def test_checked_in_worked_map_validates_and_passes_converter_preflight(self) -> None:
        example = (
            ROOT
            / "mlx-model-porting"
            / "examples"
            / "worked-port-qwen2.5-0.5b-instruct"
        )
        inspection_path = example / "inspection.json"
        mapping_path = example / "WEIGHT_MAP.json"
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        targets = [
            {
                "key": entry["target"],
                "shape": entry["target_shape"],
            }
            for entry in raw_mapping["entries"]
        ]
        with resolved_temp() as raw:
            root = root_path(raw)
            target_path = root / "worked-target.json"
            report_path = root / "worked-validation.json"
            target_path.write_text(
                json.dumps({"tensors": targets}),
                encoding="utf-8",
            )
            validated = run_validator(
                inspection_path, target_path, mapping_path, report_path
            )
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            validation = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(validation["checks"]), 290)

        normalized = converter.parse_conversion_map(raw_mapping)
        source_tensors = {
            item["key"]: item for item in inspection["tensors"]
        }
        converter._validate_mapping_against_source(normalized, source_tensors)
        budget = converter.estimate_memory_budget(
            normalized,
            {"tensors": source_tensors},
            converter.DEFAULT_MAX_WORKING_SET_BYTES,
        )
        self.assertEqual(len(normalized["mapped_targets"]), 290)
        self.assertLessEqual(
            budget["estimated_peak_bytes"], budget["limit_bytes"]
        )

    def test_draft_map_pairs_only_exact_names_and_leaves_unresolved(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            source_manifest = root / "source.json"
            target_manifest = root / "target.json"
            draft = root / "WEIGHT_MAP.draft.json"
            source_manifest.write_text(json.dumps({"tensors": [
                {"key": "exact", "shape": [2]},
                {"key": "source_only", "shape": [1]},
                {"key": "different_shape", "shape": [2]},
            ]}), encoding="utf-8")
            target_manifest.write_text(json.dumps({"tensors": [
                {"key": "exact", "shape": [2]},
                {"key": "target_only", "shape": [1]},
                {"key": "different_shape", "shape": [3]},
            ]}), encoding="utf-8")

            completed = run_converter(
                "--source", source_manifest,
                "--scaffold-manifest", target_manifest,
                "--emit-draft-map", draft,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            value = json.loads(draft.read_text(encoding="utf-8"))
            self.assertTrue(value["draft"])
            self.assertEqual([entry["source"] for entry in value["entries"]], ["exact"])
            self.assertEqual(len(value["unresolved"]), 3)

    @unittest.skipIf(os.name == "nt", "symlink component contract is POSIX-specific")
    def test_input_and_output_symlink_components_are_rejected(self) -> None:
        with resolved_temp() as raw:
            root = root_path(raw)
            real_source = root / "real-source"
            write_checkpoint(real_source)
            linked_source = root / "linked-source"
            linked_source.symlink_to(real_source, target_is_directory=True)
            mapping = root / "WEIGHT_MAP.json"
            write_mapping(mapping, complete_mapping())

            source_result = run_converter(
                "--source", linked_source,
                "--mapping", mapping,
                "--output", root / "converted",
            )
            self.assertEqual(source_result.returncode, 2, source_result.stdout + source_result.stderr)
            self.assertIn("source path contains symlink component", source_result.stderr)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            output_result = run_converter(
                "--source", real_source,
                "--mapping", mapping,
                "--output", linked_parent / "converted",
            )
            self.assertEqual(output_result.returncode, 2, output_result.stdout + output_result.stderr)
            self.assertIn("output path contains symlink component", output_result.stderr)


@unittest.skipUnless(HAS_MLX, "a usable MLX runtime is required")
class ConvertCheckpointMLXContractTests(unittest.TestCase):
    def test_mlx_core_load_reads_converted_safetensors(self) -> None:
        import mlx.core as mx

        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            write_checkpoint(source)
            write_mapping(mapping, complete_mapping())

            completed = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", output,
                pure_only=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            weights = mx.load(str(output / "model.safetensors"))
            mx.eval(weights)
            self.assertEqual(set(weights), {"k_proj", "merged", "q_proj", "renamed", "transposed", "v_proj"})
            self.assertEqual(weights["transposed"].shape, (3, 2))
            self.assertEqual(weights["merged"].dtype, mx.float16)
            report = json.loads((output / "conversion-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["outputs"]["weights"]["writer"], "mlx.core.save_safetensors")

    def test_mlx_core_load_reads_pure_python_fallback_safetensors(self) -> None:
        import mlx.core as mx

        with resolved_temp() as raw:
            root = root_path(raw)
            source = root / "source"
            mapping = root / "WEIGHT_MAP.json"
            output = root / "converted"
            write_checkpoint(source)
            write_mapping(mapping, complete_mapping())

            completed = run_converter(
                "--source", source,
                "--mapping", mapping,
                "--output", output,
                pure_only=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            weights = mx.load(str(output / "model.safetensors"))
            mx.eval(weights)
            self.assertEqual(weights["merged"].shape, (3, 2))
            self.assertEqual(weights["merged"].dtype, mx.float16)
            report = json.loads((output / "conversion-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["outputs"]["weights"]["writer"], "pure-python-safetensors")


if __name__ == "__main__":
    unittest.main()
