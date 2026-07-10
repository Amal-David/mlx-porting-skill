from __future__ import annotations

import contextlib
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_tensors  # noqa: E402


class CompareTensorsBoundednessTests(unittest.TestCase):
    def test_adjacent_uint64_values_cannot_collapse_through_float64(self) -> None:
        source = np.array([2**63], dtype=np.uint64)
        target = np.array([2**63 + 1], dtype=np.uint64)

        metrics = compare_tensors.compare_arrays(source, target, np, atol=0.0, rtol=0.0)

        self.assertFalse(metrics["allclose"])
        self.assertGreaterEqual(metrics["max_abs"], 1.0)

    def test_mixed_uint64_and_float64_cannot_collapse_to_equal(self) -> None:
        source = np.array([2**63 + 1], dtype=np.uint64)
        target = np.array([float(2**63)], dtype=np.float64)

        metrics = compare_tensors.compare_arrays(source, target, np, atol=0.0, rtol=0.0)

        self.assertFalse(metrics["dtype_compatible"])
        self.assertFalse(metrics["allclose"])

    def test_non_finite_or_negative_thresholds_cannot_bypass_parity(self) -> None:
        cases = (
            ("--atol", "inf", "--atol"),
            ("--atol", "nan", "--atol"),
            ("--atol", "-1", "--atol"),
            ("--rtol", "inf", "--rtol"),
            ("--rtol", "-1", "--rtol"),
            ("--cosine-min", "nan", "--cosine-min"),
            ("--cosine-min", "1.01", "--cosine-min"),
            ("--cosine-min", "-1.01", "--cosine-min"),
        )
        for flag, value, expected in cases:
            with self.subTest(flag=flag, value=value):
                stderr = io.StringIO()
                argv = ["compare_tensors.py", "unopened-a.npy", "unopened-b.npy", flag, value]
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
                    self.assertEqual(compare_tensors.main(), 2)
                self.assertIn(expected, stderr.getvalue())

    def test_preflight_rejects_oversized_file_before_format_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.npy"
            path.write_bytes(b"not-a-tensor" * 16)
            with (
                mock.patch.object(compare_tensors, "MAX_TENSOR_FILE_BYTES", path.stat().st_size - 1),
                mock.patch.object(
                    compare_tensors,
                    "_parse_npy_header",
                    side_effect=AssertionError("oversized input reached NPY parsing"),
                ),
                self.assertRaisesRegex(compare_tensors.SkillError, "file size limit exceeded"),
                compare_tensors.TensorArchive(path, np),
            ):
                pass

    def test_preflight_rejects_compressed_member_bomb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bomb.npz"
            np.savez_compressed(path, zeros=np.zeros(4_096, dtype=np.uint8))
            with (
                mock.patch.object(compare_tensors, "MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES", 1_024),
                mock.patch.object(
                    compare_tensors.zipfile,
                    "ZipFile",
                    side_effect=AssertionError("compressed bomb reached ZipFile"),
                ),
                self.assertRaisesRegex(compare_tensors.SkillError, "member expansion limit exceeded"),
                compare_tensors.TensorArchive(path, np),
            ):
                pass

    def test_preflight_rejects_total_uncompressed_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "total-bomb.npz"
            np.savez_compressed(
                path,
                first=np.zeros(600, dtype=np.uint8),
                second=np.zeros(600, dtype=np.uint8),
            )
            with (
                mock.patch.object(compare_tensors, "MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES", 2_048),
                mock.patch.object(compare_tensors, "MAX_TENSOR_TOTAL_UNCOMPRESSED_BYTES", 1_000),
                mock.patch.object(
                    compare_tensors.zipfile,
                    "ZipFile",
                    side_effect=AssertionError("expansion bomb reached ZipFile"),
                ),
                self.assertRaisesRegex(compare_tensors.SkillError, "total expansion limit exceeded"),
                compare_tensors.TensorArchive(path, np),
            ):
                pass

    def test_preflight_rejects_npz_member_count_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "members.npz"
            np.savez(
                path,
                first=np.array([1], dtype=np.int8),
                second=np.array([2], dtype=np.int8),
                third=np.array([3], dtype=np.int8),
            )
            with (
                mock.patch.object(compare_tensors, "MAX_NPZ_MEMBERS", 2),
                mock.patch.object(
                    compare_tensors.zipfile,
                    "ZipFile",
                    side_effect=AssertionError("member bomb reached ZipFile"),
                ),
                self.assertRaisesRegex(compare_tensors.SkillError, "member limit exceeded"),
                compare_tensors.TensorArchive(path, np),
            ):
                pass

    def test_direct_npy_expansion_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.npy"
            np.save(path, np.zeros(2_048, dtype=np.uint8))
            with (
                mock.patch.object(compare_tensors, "MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES", 1_024),
                self.assertRaisesRegex(compare_tensors.SkillError, "NPY member expansion limit exceeded"),
                compare_tensors.TensorArchive(path, np),
            ):
                pass

    def test_chunked_metrics_match_whole_array_reference_for_npy_inputs(self) -> None:
        source = np.array([1.0, -2.0, 0.0, 3.0, 8.0, -4.0, 2.0], dtype=np.float32)
        target = np.array([1.5, -1.0, 0.5, 1.0, 7.5, -3.0, 2.25], dtype=np.float16)
        aa = source.astype(np.float64)
        bb = target.astype(np.float64)
        diff = np.abs(aa - bb)
        expected_cosine = float(
            np.real(np.sum(np.conjugate(aa) * bb))
            / (np.sqrt(np.sum(np.abs(aa) ** 2)) * np.sqrt(np.sum(np.abs(bb) ** 2)))
        )

        with mock.patch.object(compare_tensors, "COMPARE_CHUNK_BYTES", 16):
            metrics = compare_tensors.compare_arrays(source, target, np, atol=0.0, rtol=0.0)
        self.assertGreater(metrics["chunks"], 1)
        self.assertEqual(metrics["max_abs"], float(diff.max()))
        self.assertAlmostEqual(metrics["mean_abs"], float(diff.mean()), places=14)
        self.assertEqual(
            metrics["max_rel"],
            float((diff / np.maximum(np.abs(aa), 1e-30)).max()),
        )
        self.assertAlmostEqual(metrics["cosine"], expected_cosine, places=14)
        self.assertFalse(metrics["allclose"])
        self.assertTrue(metrics["finite"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.npy"
            target_path = root / "target.npy"
            output = root / "report.json"
            np.save(source_path, source)
            np.save(target_path, target)
            argv = [
                "compare_tensors.py",
                str(source_path),
                str(target_path),
                "--atol", "0",
                "--rtol", "0",
                "--no-fail",
                "--output", str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(compare_tensors, "COMPARE_CHUNK_BYTES", 16),
            ):
                self.assertEqual(compare_tensors.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["compared"], 1)
            self.assertEqual(report["rows"][0]["source"], compare_tensors.NPY_SINGLE_KEY)
            self.assertAlmostEqual(report["rows"][0]["mean_abs"], float(diff.mean()), places=14)

    def test_non_finite_inputs_produce_strict_json_null_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.npy"
            target_path = root / "target.npy"
            output = root / "report.json"
            np.save(source_path, np.array([1.0, np.nan], dtype=np.float64))
            np.save(target_path, np.array([1.0, np.nan], dtype=np.float64))
            argv = [
                "compare_tensors.py",
                str(source_path),
                str(target_path),
                "--output", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(compare_tensors.main(), 1)
            raw = output.read_text(encoding="utf-8")

            def reject_constant(value: str) -> None:
                raise AssertionError(f"non-standard JSON constant: {value}")

            report = json.loads(raw, parse_constant=reject_constant)
            self.assertFalse(report["ok"])
            row = report["rows"][0]
            self.assertFalse(row["finite"])
            self.assertIsNone(row["max_abs"])
            self.assertIsNone(row["mean_abs"])
            self.assertIsNone(row["max_rel"])
            self.assertIsNone(row["cosine"])


if __name__ == "__main__":
    unittest.main()
