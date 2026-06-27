from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"

# Bundled scripts import `_common` as a sibling module.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import applies_to_family  # noqa: E402 - requires SCRIPTS on sys.path above


def run_script(name: str, *args: object, expected: int = 0) -> SimpleNamespace:
    old_argv = sys.argv[:]
    old_env = os.environ.copy()
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    try:
        os.environ.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        sys.argv = [str(SCRIPTS / name), *(str(x) for x in args)]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(str(SCRIPTS / name), run_name="__main__")
            except SystemExit as exc:
                code = int(exc.code or 0)
    finally:
        sys.argv = old_argv
        os.environ.clear()
        os.environ.update(old_env)
    if code != expected:
        raise AssertionError(
            f"{name} returned {code}, expected {expected}\n"
            f"stdout:\n{stdout.getvalue()}\nstderr:\n{stderr.getvalue()}"
        )
    return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def proto_varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def proto_field_varint(field: int, value: int) -> bytes:
    return proto_varint((field << 3) | 0) + proto_varint(value)


def proto_field_bytes(field: int, value: bytes | str) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return proto_varint((field << 3) | 2) + proto_varint(len(raw)) + raw


def onnx_value_info(name: str, dtype: int, shape: list[int]) -> bytes:
    dims = b"".join(proto_field_bytes(1, proto_field_varint(1, dim)) for dim in shape)
    tensor_type = proto_field_varint(1, dtype) + proto_field_bytes(2, dims)
    type_proto = proto_field_bytes(1, tensor_type)
    return proto_field_bytes(1, name) + proto_field_bytes(2, type_proto)


def tiny_onnx_model() -> bytes:
    node = (
        proto_field_bytes(1, "input")
        + proto_field_bytes(1, "weight")
        + proto_field_bytes(2, "output")
        + proto_field_bytes(3, "matmul_0")
        + proto_field_bytes(4, "MatMul")
    )
    tensor = (
        proto_field_bytes(1, proto_varint(2) + proto_varint(3))
        + proto_field_varint(2, 1)
        + proto_field_bytes(8, "weight")
        + proto_field_bytes(9, b"\x00" * 24)
    )
    graph = (
        proto_field_bytes(1, node)
        + proto_field_bytes(2, "tiny_graph")
        + proto_field_bytes(5, tensor)
        + proto_field_bytes(11, onnx_value_info("input", 1, [1, 2]))
        + proto_field_bytes(12, onnx_value_info("output", 1, [1, 3]))
    )
    opset = proto_field_bytes(1, "") + proto_field_varint(2, 17)
    return proto_field_varint(1, 8) + proto_field_bytes(2, "fixture") + proto_field_bytes(7, graph) + proto_field_bytes(8, opset)


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return len(raw).to_bytes(8, "little") + raw


def gguf_kv_string(key: str, value: str) -> bytes:
    return gguf_string(key) + (8).to_bytes(4, "little") + gguf_string(value)


def gguf_kv_u32(key: str, value: int) -> bytes:
    return gguf_string(key) + (4).to_bytes(4, "little") + value.to_bytes(4, "little")


def tiny_gguf_model(include_provenance: bool = True) -> bytes:
    metadata = [
        gguf_kv_string("general.architecture", "llama"),
        gguf_kv_string("general.name", "tiny-gguf"),
        gguf_kv_string("tokenizer.ggml.model", "llama"),
        gguf_kv_u32("general.file_type", 0),
        gguf_kv_u32("general.quantization_version", 2),
    ]
    if include_provenance:
        metadata.insert(2, gguf_kv_string("general.source.url", "https://example.invalid/source"))
    tensor = (
        gguf_string("blk.0.attn_q.weight")
        + (2).to_bytes(4, "little")
        + (2).to_bytes(8, "little")
        + (3).to_bytes(8, "little")
        + (0).to_bytes(4, "little")
        + (0).to_bytes(8, "little")
    )
    return (
        b"GGUF"
        + (3).to_bytes(4, "little")
        + (1).to_bytes(8, "little")
        + len(metadata).to_bytes(8, "little")
        + b"".join(metadata)
        + tensor
    )


class ToolingTests(unittest.TestCase):
    def test_skill_audit_passes(self) -> None:
        result = run_script("audit_skill.py", SKILL)
        self.assertIn("Audit passed.", result.stdout)
        self.assertIn("Architecture families: 14", result.stdout)

    def test_source_provenance_validation_passes(self) -> None:
        result = run_script("validate_sources.py", SKILL)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["sources"], 348)
        self.assertEqual(report["optimization_methods"], 27)
        self.assertTrue(report["recommendation_taxonomy"])

    def test_static_inspection_routes_dense_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            markdown = Path(tmp) / "inspection.md"
            run_script(
                "inspect_model.py",
                FIXTURES / "models" / "decoder",
                "--output", output,
                "--markdown", markdown,
            )
            report = json.loads(output.read_text())
            self.assertEqual(report["architecture_candidates"][0]["family"], "dense-decoder-transformer")
            self.assertEqual(report["tensor_summary"]["count"], 4)
            self.assertEqual(report["tensor_summary"]["parameters"], 704)
            self.assertFalse(report["license"]["requires_review"])
            self.assertIn("dense-decoder-transformer", markdown.read_text())

    def test_static_inspection_routes_moe_and_codec(self) -> None:
        for fixture, family in (("moe", "moe-decoder-transformer"), ("codec", "neural-audio-codec")):
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "inspection.json"
                run_script("inspect_model.py", FIXTURES / "models" / fixture, "--output", output)
                report = json.loads(output.read_text())
                self.assertEqual(report["architecture_candidates"][0]["family"], family)

    def test_static_inspection_flags_remote_code_and_pickle_capable_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "models" / "unsafe", "--output", output)
            report = json.loads(output.read_text())
            risk_types = {risk["type"] for risk in report["risks"]}
            self.assertTrue({"remote-code", "unsafe-serialization", "custom-code", "dependency-install", "no-safe-weights"}.issubset(risk_types))
            self.assertIsNone(report["recommended_family"])
            self.assertTrue(report["recommendation_blockers"])

    def test_static_inspection_reports_onnx_source_format_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "onnx"
            model_dir.mkdir()
            (model_dir / "model.onnx").write_bytes(tiny_onnx_model())
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", model_dir, "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["onnx"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "onnx")
            self.assertEqual(manifest["ir_version"], 8)
            self.assertEqual(manifest["opsets"][0]["version"], 17)
            self.assertEqual(manifest["graph"]["name"], "tiny_graph")
            self.assertEqual(manifest["graph"]["op_types"], {"MatMul": 1})
            self.assertEqual(manifest["graph"]["inputs"][0]["shape"], [1, 2])
            self.assertEqual(manifest["graph"]["outputs"][0]["dtype"], "FLOAT")
            self.assertEqual(manifest["graph"]["initializers"][0]["shape"], [2, 3])
            self.assertEqual(manifest["graph"]["initializers"][0]["parameters"], 6)
            self.assertEqual(manifest["graph"]["initializers"][0]["raw_data_bytes"], 24)
            self.assertIn("source-format static intake is triage-only", report["recommendation_blockers"][0])

    def test_static_inspection_reports_gguf_source_format_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "gguf"
            model_dir.mkdir()
            (model_dir / "model.gguf").write_bytes(tiny_gguf_model())
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", model_dir, "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["gguf"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "gguf")
            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["header"]["magic"], "GGUF")
            self.assertEqual(manifest["header"]["alignment"], 32)
            self.assertEqual(manifest["architecture"], "llama")
            self.assertEqual(manifest["tokenizer_model"], "llama")
            self.assertEqual(manifest["quantization_version"], 2)
            self.assertIn("general.source.url", manifest["source_provenance_keys"])
            self.assertEqual(manifest["quantization_summary"]["quantized_tensor_count"], 0)
            self.assertEqual(manifest["tensors"][0]["name"], "blk.0.attn_q.weight")
            self.assertEqual(manifest["tensors"][0]["shape"], [2, 3])
            self.assertEqual(manifest["tensors"][0]["type"], "F32")
            self.assertEqual(manifest["tensors"][0]["relative_data_offset"], 0)
            self.assertGreaterEqual(manifest["tensors"][0]["absolute_data_offset"], manifest["tensor_data_start_offset"])
            self.assertIn("source-format static intake is triage-only", report["recommendation_blockers"][0])

    def test_static_inspection_blocks_gguf_without_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "gguf"
            model_dir.mkdir()
            (model_dir / "model.gguf").write_bytes(tiny_gguf_model(include_provenance=False))
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", model_dir, "--output", output)
            report = json.loads(output.read_text())
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertIn("Missing source/base-model provenance metadata.", manifest["hold_conditions"])
            self.assertIn("GGUF source/base-model provenance is missing or incomplete", report["recommendation_blockers"])

    def test_port_plan_is_architecture_and_evidence_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            plan = Path(tmp) / "PORT_PLAN.md"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", inspection)
            run_script("make_port_plan.py", inspection, "--output", plan)
            text = plan.read_text()
            self.assertIn("dense-decoder-transformer", text)
            self.assertIn("runbook-decoder-transformer.md", text)
            self.assertIn("Source oracle", text)
            self.assertIn("Optimization shortlist", text)
            self.assertIn("Expected effect", text)
            self.assertIn("Research candidates", text)
            self.assertIn("Stop conditions", text)

    def test_optimization_recommender_splits_ready_and_research_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inspection = tmp_path / "inspection.json"
            output = tmp_path / "recommendations.json"
            markdown = tmp_path / "recommendations.md"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", inspection)
            run_script(
                "recommend_optimizations.py",
                inspection,
                "--output", output,
                "--markdown", markdown,
                "--limit", 5,
            )
            report = json.loads(output.read_text())
            ready_ids = {item["id"] for item in report["ready_candidates"]}
            research_ids = {item["id"] for item in report["research_candidates"]}
            self.assertIn("fast-sdpa", ready_ids)
            self.assertNotIn("moe-expert-dispatch-and-quantization", ready_ids)
            self.assertIn("prompt-lookup-ngram-speculation", research_ids)
            text = markdown.read_text()
            self.assertIn("Ready candidates", text)
            self.assertIn("Research experiments", text)

    def test_block_weight_streaming_is_scoped_to_repeated_block_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps({"recommended_family": "diffusion-flow"}))
            diffusion = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "peak-memory", "--limit", 50,
            ).stdout)
            diffusion_ready = {m["id"] for m in diffusion["ready_candidates"]}
            self.assertIn("block-weight-streaming", diffusion_ready)

            inspection.write_text(json.dumps({"recommended_family": "dense-decoder-transformer"}))
            decoder = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "peak-memory", "--limit", 50,
            ).stdout)
            decoder_ids = {m["id"] for m in decoder["ready_candidates"] + decoder["research_candidates"]}
            self.assertNotIn("block-weight-streaming", decoder_ids)

    def test_grid_sample_kernel_is_scoped_to_spatial_ports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps({"recommended_family": "diffusion-flow"}))
            diffusion = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "latency", "--limit", 50,
            ).stdout)
            diffusion_ready = {m["id"] for m in diffusion["ready_candidates"]}
            self.assertIn("spatial-grid-sample-kernel", diffusion_ready)

            inspection.write_text(json.dumps({"recommended_family": "dense-decoder-transformer"}))
            decoder = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "latency", "--limit", 50,
            ).stdout)
            decoder_ids = {m["id"] for m in decoder["ready_candidates"] + decoder["research_candidates"]}
            self.assertNotIn("spatial-grid-sample-kernel", decoder_ids)

    def test_optimization_guidance_has_no_unreachable_methods(self) -> None:
        # Every method must be reachable by at least one architecture family via the
        # SAME matcher production uses (imported, not re-implemented) so a divergence
        # between test and production cannot mask a real reachability bug.
        families = [f["id"] for f in json.loads((SKILL / "assets" / "architectures.yaml").read_text())["families"]]
        guidance = json.loads((SKILL / "assets" / "optimization_guidance.yaml").read_text())
        unreachable = [
            m["id"]
            for m in guidance["methods"]
            if not any(applies_to_family(m.get("applies_to", []), fam) for fam in families)
        ]
        self.assertEqual(unreachable, [], f"methods unreachable by any architecture family: {unreachable}")

    def test_recommender_rejects_cuda_only_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps({"recommended_family": "dense-decoder-transformer"}))
            result = run_script("recommend_optimizations.py", inspection, "--limit", 50)
            report = json.loads(result.stdout)
            ready = {m["id"] for m in report["ready_candidates"]}
            research = {m["id"] for m in report["research_candidates"]}
            excluded = {m["id"] for m in report["notable_exclusions"]}
            self.assertNotIn("cuda-graphs-decode-capture", ready | research)
            self.assertIn("cuda-graphs-decode-capture", excluded)
            # Reachability fix: KV-cache quantization must reach a dense decoder.
            self.assertIn("uniform-kv-quantization", ready | research)

    def test_recommender_holds_candidates_when_intake_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps({
                "recommended_family": "dense-decoder-transformer",
                "recommendation_blockers": ["high: remote-code auto_map present"],
            }))
            blocked = json.loads(run_script("recommend_optimizations.py", inspection).stdout)
            self.assertTrue(blocked["blocked"])
            self.assertEqual(blocked["ready_candidates"], [])
            self.assertEqual(blocked["research_candidates"], [])
            self.assertTrue(blocked["held_candidates"])
            allowed = json.loads(run_script("recommend_optimizations.py", inspection, "--allow-blocked").stdout)
            self.assertFalse(allowed["blocked"])
            self.assertTrue(allowed["ready_candidates"])

    def test_recommender_requires_family_and_respects_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps({"recommendation_blockers": []}))
            # No family anywhere -> hard error, not a silent empty shortlist.
            run_script("recommend_optimizations.py", inspection, expected=2)
            # --family override + schema_version + gated stdout (no print when --output is set).
            output = Path(tmp) / "rec.json"
            result = run_script(
                "recommend_optimizations.py", inspection,
                "--family", "moe-decoder-transformer", "--output", output,
            )
            self.assertEqual(result.stdout.strip(), "")
            report = json.loads(output.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["family"], "moe-decoder-transformer")
            # --objective filter narrows to methods that carry that objective.
            filtered = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--family", "dense-decoder-transformer", "--objective", "peak-memory", "--limit", 50,
            ).stdout)
            surfaced = filtered["ready_candidates"] + filtered["research_candidates"]
            self.assertTrue(surfaced)
            guidance = {m["id"]: m for m in json.loads((SKILL / "assets" / "optimization_guidance.yaml").read_text())["methods"]}
            for item in surfaced:
                self.assertIn("peak-memory", guidance[item["id"]]["objectives"])

    def test_port_plan_excludes_rejected_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            plan = Path(tmp) / "PORT_PLAN.md"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", inspection)
            run_script("make_port_plan.py", inspection, "--output", plan)
            self.assertNotIn("cuda-graphs-decode-capture", plan.read_text())

    def test_pipeline_chains_inspect_to_plan_recommend_and_validate(self) -> None:
        # Prove inspect_model.py output feeds plan, recommend, and validate_weight_map
        # --source with no manual reshaping (guards JSON-schema cohesion across the CLIs).
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inspection = tmp / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", inspection)
            data = json.loads(inspection.read_text())
            self.assertTrue(data["tensors"])

            run_script("make_port_plan.py", inspection, "--output", tmp / "PORT_PLAN.md")

            rec = tmp / "rec.json"
            run_script("recommend_optimizations.py", inspection, "--output", rec)
            self.assertEqual(json.loads(rec.read_text())["schema_version"], 1)

            # inspection.json is itself a valid --source manifest for validate_weight_map.
            target = {"tensors": [{"key": t["key"], "shape": t["shape"]} for t in data["tensors"]]}
            mapping = {
                "entries": [{"source": t["key"], "target": t["key"], "transforms": []} for t in data["tensors"]],
                "ignored_source": [],
                "generated_target": [],
            }
            target_path, map_path, report_path = tmp / "target.json", tmp / "map.json", tmp / "wm.json"
            target_path.write_text(json.dumps(target))
            map_path.write_text(json.dumps(mapping))
            run_script(
                "validate_weight_map.py",
                "--source", inspection, "--target", target_path, "--mapping", map_path, "--output", report_path,
            )
            report = json.loads(report_path.read_text())
            self.assertTrue(report["ok"], report)
            self.assertFalse(report["unexplained_source"])
            self.assertFalse(report["unexplained_target"])

    def test_weight_mapping_transforms_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "weight-report.json"
            run_script(
                "validate_weight_map.py",
                "--source", FIXTURES / "manifests" / "source.json",
                "--target", FIXTURES / "manifests" / "target.json",
                "--mapping", FIXTURES / "manifests" / "map.json",
                "--output", output,
            )
            report = json.loads(output.read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(report["checks"][0]["transformed_shape"], [2, 3])
            self.assertFalse(report["unexplained_source"])
            self.assertFalse(report["unexplained_target"])

    def test_weight_mapping_rejects_duplicate_and_unknown_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mapping = Path(tmp) / "bad-map.json"
            mapping.write_text(json.dumps({
                "entries": [
                    {"source": "linear.weight", "target": "layer.weight", "transforms": [{"op": "transpose", "axes": [1, 0]}]},
                    {"source": "linear.weight", "target": "layer.bias", "transforms": []},
                ],
                "ignored_source": ["missing.source"],
                "generated_target": ["missing.target"],
            }))
            output = Path(tmp) / "bad-weight-report.json"
            run_script(
                "validate_weight_map.py",
                "--source", FIXTURES / "manifests" / "source.json",
                "--target", FIXTURES / "manifests" / "target.json",
                "--mapping", mapping,
                "--output", output,
                expected=1,
            )
            report = json.loads(output.read_text())
            self.assertFalse(report["ok"])
            self.assertTrue(any("source mapped more than once" in error for error in report["errors"]))
            self.assertTrue(any("ignored_source keys not found" in error for error in report["errors"]))

    def test_tensor_comparison_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = FIXTURES / "tensors" / "source.npz"
            close = FIXTURES / "tensors" / "close.npz"
            bad = FIXTURES / "tensors" / "bad.npz"
            good_out = tmp_path / "good.json"
            bad_out = tmp_path / "bad.json"
            run_script("compare_tensors.py", src, close, "--output", good_out)
            run_script("compare_tensors.py", src, bad, "--output", bad_out, expected=1)
            self.assertTrue(json.loads(good_out.read_text())["ok"])
            self.assertFalse(json.loads(bad_out.read_text())["ok"])

    def test_tensor_comparison_rejects_empty_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "empty.json"
            run_script(
                "compare_tensors.py",
                FIXTURES / "tensors" / "source.npz",
                FIXTURES / "tensors" / "close.npz",
                "--include", "does-not-match-anything",
                "--output", output,
                expected=1,
            )
            report = json.loads(output.read_text())
            self.assertFalse(report["ok"])
            self.assertEqual(report["compared"], 0)
            self.assertTrue(any("no source tensors selected" in failure for failure in report["failures"]))

    def test_compare_tensors_fails_loud_on_nan_and_inf(self) -> None:
        import numpy as np

        # A regression that treated NaN==NaN (or inf) as a match would keep every other
        # test green; this guards the finite + equal_nan=False fail-loud contract.
        for bad_value in (np.nan, np.inf):
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "a.npz"
                dst = Path(tmp) / "b.npz"
                out = Path(tmp) / "r.json"
                arr = np.array([[1.0, bad_value], [3.0, 4.0]], dtype=np.float64)
                np.savez(src, w=arr)
                np.savez(dst, w=arr.copy())
                run_script("compare_tensors.py", src, dst, "--output", out, expected=1)
                report = json.loads(out.read_text())
                self.assertFalse(report["ok"])
                self.assertFalse(report["rows"][0]["finite"])

    def test_weight_map_transform_ops_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = {"tensors": [
                {"key": "reshape.w", "shape": [4, 3]},
                {"key": "squeeze.w", "shape": [1, 5]},
                {"key": "unsqueeze.w", "shape": [5]},
                {"key": "slice.w", "shape": [10]},
                {"key": "permute.w", "shape": [2, 3, 4]},
            ]}
            target = {"tensors": [
                {"key": "reshape.t", "shape": [2, 6]},
                {"key": "squeeze.t", "shape": [5]},
                {"key": "unsqueeze.t", "shape": [1, 5]},
                {"key": "slice.t", "shape": [3]},
                {"key": "permute.t", "shape": [4, 2, 3]},
            ]}
            mapping = {"entries": [
                {"source": "reshape.w", "target": "reshape.t", "transforms": [{"op": "reshape", "shape": [-1, 6]}]},
                {"source": "squeeze.w", "target": "squeeze.t", "transforms": [{"op": "squeeze", "axis": 0}]},
                {"source": "unsqueeze.w", "target": "unsqueeze.t", "transforms": [{"op": "unsqueeze", "axis": 0}]},
                {"source": "slice.w", "target": "slice.t", "transforms": [{"op": "slice", "axis": 0, "start": 2, "end": 8, "step": 2}]},
                {"source": "permute.w", "target": "permute.t", "transforms": [{"op": "permute", "axes": [2, 0, 1]}]},
            ], "ignored_source": [], "generated_target": []}
            sp, tp, mp, out = tmp / "s.json", tmp / "t.json", tmp / "m.json", tmp / "r.json"
            sp.write_text(json.dumps(source))
            tp.write_text(json.dumps(target))
            mp.write_text(json.dumps(mapping))
            run_script("validate_weight_map.py", "--source", sp, "--target", tp, "--mapping", mp, "--output", out)
            report = json.loads(out.read_text())
            self.assertTrue(report["ok"], report)
            shapes = {c["source"]: c["transformed_shape"] for c in report["checks"]}
            self.assertEqual(shapes["reshape.w"], [2, 6])
            self.assertEqual(shapes["squeeze.w"], [5])
            self.assertEqual(shapes["unsqueeze.w"], [1, 5])
            self.assertEqual(shapes["slice.w"], [3])
            self.assertEqual(shapes["permute.w"], [4, 2, 3])

    def test_weight_map_transform_ops_reject_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = {"tensors": [{"key": "x", "shape": [2, 3]}]}
            target = {"tensors": [{"key": "y", "shape": [3]}]}
            mapping = {"entries": [
                {"source": "x", "target": "y", "transforms": [{"op": "squeeze", "axis": 0}]},
            ], "ignored_source": [], "generated_target": []}
            sp, tp, mp, out = tmp / "s.json", tmp / "t.json", tmp / "m.json", tmp / "r.json"
            sp.write_text(json.dumps(source))
            tp.write_text(json.dumps(target))
            mp.write_text(json.dumps(mapping))
            run_script("validate_weight_map.py", "--source", sp, "--target", tp, "--mapping", mp, "--output", out, expected=1)
            report = json.loads(out.read_text())
            self.assertFalse(report["ok"])
            self.assertTrue(any("squeeze" in error for error in report["errors"]))

    def test_compare_tensors_cosine_gate_complex_and_no_fail(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "a.npz", tmp / "b.npz"
            np.savez(a, v=np.array([1.0, 2.0, 3.0]))
            np.savez(b, v=np.array([-1.0, -2.0, -3.0]))
            cos_out = tmp / "cos.json"
            run_script("compare_tensors.py", a, b, "--cosine-min", "0.99", "--output", cos_out, expected=1)
            cos = json.loads(cos_out.read_text())
            self.assertFalse(cos["ok"])
            self.assertLess(cos["rows"][0]["cosine"], 0.99)
            # Identical vectors clear the same cosine floor.
            c = tmp / "c.npz"
            np.savez(c, v=np.array([1.0, 2.0, 3.0]))
            run_script("compare_tensors.py", a, c, "--cosine-min", "0.99", "--output", tmp / "cos2.json")
            # Complex dtype path compares cleanly when identical.
            ca, cb = tmp / "ca.npz", tmp / "cb.npz"
            cz = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
            np.savez(ca, z=cz)
            np.savez(cb, z=cz.copy())
            cx_out = tmp / "cx.json"
            run_script("compare_tensors.py", ca, cb, "--output", cx_out)
            self.assertTrue(json.loads(cx_out.read_text())["ok"])
            # --no-fail reports the mismatch but exits 0 (report-only).
            d = tmp / "d.npz"
            np.savez(d, v=np.array([9.0, 9.0, 9.0]))
            nf_out = tmp / "nf.json"
            run_script("compare_tensors.py", a, d, "--no-fail", "--output", nf_out)
            nf = json.loads(nf_out.read_text())
            self.assertFalse(nf["ok"])
            self.assertTrue(nf["report_only"])

    def test_benchmark_reports_failure_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench.json"
            run_script(
                "benchmark_command.py", "--warmup", "0", "--runs", "1", "--output", out,
                "--", sys.executable, "-c", "import sys; sys.exit(3)", expected=1,
            )
            report = json.loads(out.read_text())
            self.assertFalse(report["ok"])
            self.assertEqual(report["summary"]["successful_runs"], 0)

    def test_manifest_generate_and_check_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "a.txt").write_text("hello")
            (tmp / "sub").mkdir()
            (tmp / "sub" / "b.txt").write_text("world")
            run_script("manifest.py", "generate", "--root", tmp)
            self.assertTrue((tmp / "MANIFEST.json").exists())
            run_script("manifest.py", "check", "--root", tmp)  # clean tree verifies
            # A changed file is caught.
            (tmp / "a.txt").write_text("HELLO-CHANGED")
            changed = run_script("manifest.py", "check", "--root", tmp, expected=1)
            self.assertIn("a.txt", changed.stdout)
            # Regenerating restores a clean check.
            run_script("manifest.py", "generate", "--root", tmp)
            run_script("manifest.py", "check", "--root", tmp)
            # A newly added file is caught as drift too.
            (tmp / "c.txt").write_text("new")
            run_script("manifest.py", "check", "--root", tmp, expected=1)

    def test_daily_update_collector_is_review_only_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidates.json"
            result = run_script(
                "update_sources.py",
                "--offline-fixture", FIXTURES / "updates" / "offline.json",
                "--output", output,
            )
            report = json.loads(output.read_text())
            self.assertEqual(len(report["repositories"]), 1)
            self.assertEqual(len(report["papers"]), 1)
            self.assertEqual(report["papers"][0]["review_status"], "candidate-unreviewed")
            self.assertTrue(any("Do not execute code" in line for line in report["instructions"]))
            self.assertIn("1 repositories, 1 paper candidates", result.stdout)

    def test_research_loop_generates_review_blogs_and_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "research-run"
            run_script(
                "research_loop.py",
                "--run-id", "fixture-loop",
                "--objective", "Broaden MLX porting evidence beyond GitHub",
                "--offline-fixture", FIXTURES / "research_loop" / "offline_findings.json",
                "--output-dir", output_dir,
            )
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            self.assertTrue(synthesis["review_only"])
            self.assertEqual(synthesis["finding_count"], 3)
            self.assertEqual(synthesis["execution_counts"]["fixture_ingested"], 3)
            self.assertEqual(synthesis["execution_counts"]["scaffolded_not_run"], 3)
            self.assertEqual(synthesis["decision_counts"]["adopted"], 1)
            self.assertIn("official_docs", synthesis["non_github_lanes_covered"])
            self.assertIn("hugging_face", synthesis["non_github_lanes_covered"])
            self.assertIn("technical_blogs", synthesis["non_github_lanes_covered"])
            self.assertIn("tests/fixtures/research_loop/offline_findings.json", synthesis["offline_fixture"])
            self.assertEqual(len(assignments["assignments"]), 6)
            self.assertEqual(assignments["assignments"][0]["execution"]["state"], "fixture_ingested")
            self.assertEqual(assignments["assignments"][-1]["execution"]["state"], "scaffolded_not_run")
            prompt = assignments["assignments"][0]["prompt"]
            self.assertIn("Do not execute remote model code", prompt)
            blog = output_dir / "blogs" / "official-docs-cartographer.md"
            self.assertTrue(blog.exists())
            blog_text = blog.read_text()
            self.assertIn("Candidate findings", blog_text)
            self.assertIn("official-custom-metal-validation", blog_text)

    def test_research_loop_rejects_malformed_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "bad_findings.json"
            fixture.write_text(json.dumps({
                "agents": [
                    {
                        "persona_id": "official-docs-cartographer",
                        "findings": [
                            {
                                "id": "missing-source",
                                "title": "Missing source should fail",
                                "summary": "No source provenance.",
                                "source_lane": "official_docs",
                                "decision": "held",
                                "evidence_level": "lead-only",
                                "validation_gate": "None",
                                "affects": ["references/benchmarking.md"],
                            }
                        ],
                    }
                ]
            }))
            result = run_script(
                "research_loop.py",
                "--offline-fixture", fixture,
                "--output-dir", Path(tmp) / "out",
                expected=2,
            )
            self.assertIn("missing required fields: sources", result.stderr)

    def test_research_loop_executor_records_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "executor-run"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            run_script(
                "research_loop.py",
                "--run-id", "executor-loop",
                "--objective", "Prove local executor receipt flow",
                "--agent-count", 2,
                "--executor-command", f"{sys.executable} {fake_executor}",
                "--output-dir", output_dir,
            )
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            self.assertEqual(synthesis["finding_count"], 2)
            self.assertEqual(synthesis["execution_counts"]["executor_completed"], 2)
            self.assertEqual(synthesis["execution_counts"]["executor_failed"], 0)
            self.assertEqual(synthesis["execution_counts"]["scaffolded_not_run"], 0)
            self.assertIn("executor_command", synthesis)
            for assignment in assignments["assignments"]:
                execution = assignment["execution"]
                self.assertEqual(execution["kind"], "local-executor")
                self.assertEqual(execution["state"], "executor_completed")
                self.assertEqual(execution["exit_code"], 0)
                for key in ("prompt_path", "stdout_path", "stderr_path", "result_path"):
                    self.assertTrue((output_dir / execution[key]).exists(), key)
                result = json.loads((output_dir / execution["result_path"]).read_text())
                self.assertEqual(result["persona_id"], assignment["persona_id"])
            blog = output_dir / "blogs" / "official-docs-cartographer.md"
            self.assertIn("fake-executor-official-docs-cartographer", blog.read_text())

    def test_research_loop_rejects_fixture_and_executor_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            result = run_script(
                "research_loop.py",
                "--offline-fixture", FIXTURES / "research_loop" / "offline_findings.json",
                "--executor-command", f"{sys.executable} {fake_executor}",
                "--output-dir", Path(tmp) / "out",
                expected=2,
            )
            self.assertIn("mutually exclusive", result.stderr)

    def test_benchmark_harness_and_installer_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "bench.json"
            run_script(
                "benchmark_command.py",
                "--warmup", 1,
                "--runs", 2,
                "--output", output,
                "--", sys.executable, "-c", "print('ok')",
            )
            report = json.loads(output.read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(report["summary"]["successful_runs"], 2)
            install = run_script("install_skill.py", "--dest", tmp_path / "skills", "--dry-run")
            self.assertIn("target:", install.stdout)

    @unittest.skipIf(sys.platform == "win32", "POSIX process-group timeout behavior only")
    def test_benchmark_timeout_kills_child_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child-leaked"
            report_path = Path(tmp) / "benchmark.json"
            code = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', \"import pathlib, time; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('leaked')\"]); "
                "time.sleep(5)"
            )
            run_script(
                "benchmark_command.py",
                "--warmup", "0",
                "--runs", "1",
                "--timeout", "0.2",
                "--output", report_path,
                "--",
                sys.executable, "-c", code,
                expected=1,
            )
            time.sleep(1.2)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["runs"][0]["timed_out"])
            self.assertFalse(marker.exists(), "timed-out child process was left running")


if __name__ == "__main__":
    unittest.main()
