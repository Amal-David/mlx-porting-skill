from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import runpy
import shutil
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

from _common import applies_to_family, compose_stack_band, load_structured, parse_band  # noqa: E402 - requires SCRIPTS on sys.path above


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


def trusted_inspection_fixture(
    families: list[str],
    *,
    model_type: str = "synthetic",
    traits: list[str] | None = None,
    blockers: list[str] | None = None,
    runbooks: list[str] | None = None,
) -> dict[str, object]:
    """Build a complete synthetic inspect_model.py report for downstream contract tests."""
    resolved_blockers = list(blockers or [])
    primary = families[0] if families else None
    resolved_runbooks = runbooks or [f"references/runbook-{family}.md" for family in families]
    recommended = not resolved_blockers and primary is not None
    manifest = [
        {"path": "config.json", "size_bytes": 128, "sha256": "1" * 64},
        {"path": "model.safetensors", "size_bytes": 12, "sha256": "2" * 64},
    ]
    canonical_identity = json.dumps(
        {"schema_version": 1, "files": manifest},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "generated_at": "2026-07-10T00:00:00+00:00",
        "inspection_mode": "static-no-model-import",
        "source": {
            "kind": "local",
            "input": "synthetic-model",
            "revision": "a" * 40,
        },
        "artifact_identity": {
            "schema_version": 1,
            "algorithm": "sha256-tree-v1",
            "status": "verified",
            "immutable": True,
            "fingerprint": "sha256:" + hashlib.sha256(canonical_identity).hexdigest(),
            "file_count": len(manifest),
            "total_bytes": sum(record["size_bytes"] for record in manifest),
            "manifest": manifest,
            "errors": [],
        },
        "local_path": "synthetic-model",
        "configs": {"config.json": {"model_type": model_type, "license": "apache-2.0"}},
        "file_summary": {
            "count": 2,
            "truncated": False,
            "extensions": {".json": 1, ".safetensors": 1},
            "files": [
                {"path": "config.json", "size_bytes": 128, "suffix": ".json"},
                {"path": "model.safetensors", "size_bytes": 12, "suffix": ".safetensors"},
            ],
        },
        "tensor_summary": {
            "count": 1,
            "parameters": 1,
            "estimated_bytes": 4,
            "dtypes": {"F32": 1},
            "files": {"model.safetensors": 1},
            "metadata": {},
            "errors": [],
            "integrity_ok": True,
        },
        "tensors": [],
        "source_format_summary": {
            "count": 0,
            "formats": [],
            "errors": [],
            "integrity_ok": True,
            "aggregate": {
                "artifact_count": 0,
                "inspected_bytes": 0,
                "artifact_limit": 256,
                "byte_limit": 536870912,
            },
            "manifests": [],
        },
        "architecture_candidates": [
            {
                "family": family,
                "score": 20.0 - index,
                "confidence": 1.0,
                "runbook": resolved_runbooks[index] if index < len(resolved_runbooks) else None,
                "targets": [],
                "state": "must be documented",
                "evidence": ["synthetic exact identity"],
                "notes": "",
            }
            for index, family in enumerate(families)
        ],
        "architecture_profile": None,
        "architecture_traits": list(traits or []),
        "routing_decision": {
            "status": "recommended",
            "minimum_score": 8.0,
            "minimum_margin": 2.0,
            "winner_family": primary,
            "winner_score": 20.0 if primary else 0.0,
            "runner_up_family": families[1] if len(families) > 1 else None,
            "runner_up_score": 15.0 if len(families) > 1 else None,
            "winner_margin": 5.0 if len(families) > 1 else None,
            "reasons": [],
        },
        "recommended_family": primary if recommended else None,
        "recommended_runbook": resolved_runbooks[0] if recommended and resolved_runbooks else None,
        "recommended_families": list(families) if recommended else [],
        "recommended_runbooks": list(resolved_runbooks) if recommended else [],
        "recommendation_blockers": resolved_blockers,
        "license": {
            "declared": [{"source": "config.json", "value": "apache-2.0"}],
            "license_files": [],
            "file_evidence": [],
            "accepted_evidence": [{
                "kind": "declaration",
                "source": "config.json",
                "value": "apache-2.0",
            }],
            "status": "acceptable-evidence",
            "requires_review": False,
            "compatibility_assessed": False,
            "reasons": [],
        },
        "risks": [],
        "limitations": [],
    }


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
    matmul_node = (
        proto_field_bytes(1, "input")
        + proto_field_bytes(1, "weight")
        + proto_field_bytes(2, "matmul_output")
        + proto_field_bytes(3, "matmul_0")
        + proto_field_bytes(4, "MatMul")
    )
    unsupported_node = (
        proto_field_bytes(1, "boxes")
        + proto_field_bytes(1, "scores")
        + proto_field_bytes(2, "output")
        + proto_field_bytes(3, "nms_0")
        + proto_field_bytes(4, "NonMaxSuppression")
    )
    tensor = (
        proto_field_bytes(1, proto_varint(2) + proto_varint(3))
        + proto_field_varint(2, 1)
        + proto_field_bytes(8, "weight")
        + proto_field_bytes(9, b"\x00" * 24)
    )
    graph = (
        proto_field_bytes(1, matmul_node)
        + proto_field_bytes(1, unsupported_node)
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
        self.assertIn("Architecture families: 17", result.stdout)

    def test_source_provenance_validation_passes(self) -> None:
        result = run_script("validate_sources.py", SKILL)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"], report)
        source_count = len(json.loads((SKILL / "assets" / "sources.yaml").read_text())["sources"])
        self.assertEqual(report["sources"], source_count)
        self.assertEqual(report["optimization_methods"], 28)
        self.assertEqual(report["optimization_stacks"], 4)
        self.assertTrue(report["recommendation_taxonomy"])

    def test_adapter_verification_versions_match_version_file(self) -> None:
        version = (ROOT / "VERSION").read_text().strip()
        version_pattern = re.compile(r"\bversion:? `([^`]+)`", re.IGNORECASE)

        for path in sorted((ROOT / "adapters").glob("*.md")):
            with self.subTest(adapter=path.name):
                versions = version_pattern.findall(path.read_text())
                self.assertTrue(versions, f"{path} has no verification version")
                self.assertEqual(set(versions), {version})

    def test_optimization_stacks_schema_is_machine_readable(self) -> None:
        stacks_path = SKILL / "assets" / "optimization_stacks.yaml"
        stacks = load_structured(stacks_path)
        guidance = load_structured(SKILL / "assets" / "optimization_guidance.yaml")
        method_ids = {method["id"] for method in guidance["methods"]}
        numeric_range = re.compile(r"\b~?\d+(?:\.\d+)?x(?:\s*-\s*\d+(?:\.\d+)?x)?\b")

        raw = stacks_path.read_text()
        self.assertNotIn('"compound_range"', raw)
        self.assertEqual(len(stacks["stacks"]), 4)
        for stack in stacks["stacks"]:
            with self.subTest(stack=stack["id"]):
                step_ids = {step["method"] for step in stack["steps"]}
                self.assertTrue(step_ids)
                self.assertTrue(step_ids.issubset(method_ids))
                compound = stack["compound"]
                self.assertIs(compound["measured_together"], False)
                self.assertEqual(compound["receipts"], [])
                if stack["id"] == "dense-decoder-inference":
                    rejected = stack["observed_configurations"][0]
                    self.assertEqual(rejected["decision"], "rejected")
                    self.assertEqual(rejected["measured_ratio"], "0.33x")
                    self.assertIn("not a measurement", rejected["basis"])
                compound_shape = {key: value for key, value in compound.items() if key != "receipts"}
                self.assertIsNone(numeric_range.search(json.dumps(compound_shape)))
                for note in stack["composition_notes"]:
                    self.assertEqual(len(note["pair"]), 2)
                    self.assertTrue(set(note["pair"]).issubset(step_ids))

    def test_source_validation_fails_loudly_on_invalid_optimization_stack(self) -> None:
        for mutation, expected in (
            ("unknown-step", "optimization stack dense-decoder-inference step references missing method missing-method"),
            ("compound-range", "optimization stack dense-decoder-inference compound stores a numeric range"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                tmp_skill = Path(tmp) / "skill"
                shutil.copytree(SKILL / "assets", tmp_skill / "assets")
                stacks_path = tmp_skill / "assets" / "optimization_stacks.yaml"
                stacks = json.loads(stacks_path.read_text())
                if mutation == "unknown-step":
                    stacks["stacks"][0]["steps"][0]["method"] = "missing-method"
                else:
                    stacks["stacks"][0]["compound"]["range"] = "1.0x-2.0x"
                stacks_path.write_text(json.dumps(stacks, indent=2) + "\n")

                result = run_script("validate_sources.py", tmp_skill, expected=1)
                self.assertIn(expected, result.stdout)

    def test_improvement_band_floor_policy_rejects_unpromoted_local_receipts(self) -> None:
        cases = (
            (
                "local-reproduced-with-receipts",
                "local_reproduced",
                "keep",
                "is not backed by a local-promotion effective claim",
            ),
            (
                "source-reported",
                "source_reported",
                "remove",
                "improvement_band range must start at 1.0x unless local_reproduced has receipts",
            ),
            (
                "local-reproduced-without-receipts",
                "local_reproduced",
                "empty",
                "improvement_band range must start at 1.0x unless local_reproduced has receipts",
            ),
        )
        for name, provenance, receipt_mode, expected_error in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                tmp_skill = Path(tmp) / "skill"
                shutil.copytree(SKILL / "assets", tmp_skill / "assets")
                guidance_path = tmp_skill / "assets" / "optimization_guidance.yaml"
                guidance = json.loads(guidance_path.read_text())
                method = next(item for item in guidance["methods"] if item["id"] == "native-low-bit-weight-quantization")
                band = method["improvement_band"]
                band["provenance"] = provenance
                band["range"] = "1.2x-2.4x"
                if receipt_mode == "remove":
                    band.pop("receipts", None)
                    band.pop("measured_on", None)
                elif receipt_mode == "empty":
                    band["receipts"] = []
                guidance_path.write_text(json.dumps(guidance, indent=2) + "\n")

                result = run_script("validate_sources.py", tmp_skill, expected=1)
                self.assertIn(expected_error, result.stdout)

    def test_source_reported_ranges_cannot_be_marked_profile_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_skill = Path(tmp) / "skill"
            shutil.copytree(SKILL / "assets", tmp_skill / "assets")
            guidance_path = tmp_skill / "assets" / "optimization_guidance.yaml"
            guidance = json.loads(guidance_path.read_text())
            method = next(
                item for item in guidance["methods"]
                if item["id"] == "continuous-batching-serving"
            )
            band = method["improvement_band"]
            band["provenance"] = "source_reported"
            band["claim_status"] = "eligible-with-profile"
            band["range"] = band.pop("observed_source_range")
            guidance_path.write_text(json.dumps(guidance, indent=2) + "\n")

            result = run_script("validate_sources.py", tmp_skill, expected=1)
            self.assertIn(
                "source_reported improvement_band for continuous-batching-serving "
                "must remain held until locally reproduced",
                result.stdout,
            )

    def test_supported_method_status_requires_classified_compatible_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_skill = Path(tmp) / "skill"
            shutil.copytree(SKILL / "assets", tmp_skill / "assets")
            guidance_path = tmp_skill / "assets" / "optimization_guidance.yaml"
            guidance = json.loads(guidance_path.read_text())
            method = next(
                item for item in guidance["methods"]
                if item["id"] == "generic-audio-prefix-cache"
            )
            method["status"] = "proven-mlx-port"
            guidance_path.write_text(json.dumps(guidance, indent=2) + "\n")

            result = run_script("validate_sources.py", tmp_skill, expected=1)
            self.assertIn(
                "optimization guidance method generic-audio-prefix-cache status proven-mlx-port "
                "lacks synthesized pinned supporting evidence",
                result.stdout,
            )

    def test_model_outcome_registry_preserves_claim_boundaries(self) -> None:
        sources = {source["id"] for source in json.loads((SKILL / "assets" / "sources.yaml").read_text())["sources"]}
        local_sources = {"asset-architectures", "asset-research-backlog"}
        outcomes = json.loads((SKILL / "assets" / "model_outcomes.json").read_text())

        for record in outcomes["records"]:
            with self.subTest(record=record["id"]):
                self.assertTrue(record["worked"])
                self.assertTrue(record["did_not_work"])
                self.assertTrue(record["claim_boundary"])
                self.assertIn("potential_speedup", record)
                self.assertIn("overall", record["potential_speedup"])
                self.assertIn("speculative_decoding", record["potential_speedup"])
                for claim_name in ("overall", "speculative_decoding"):
                    claim = record["potential_speedup"][claim_name]
                    if claim.get("provenance") in {"profile_required", "performance_observation"}:
                        self.assertIsNone(claim.get("range"))
                    else:
                        self.assertRegex(claim["range"], r"\d+(?:\.\d+)?x-\d+(?:\.\d+)?x")
                self.assertTrue(record["potential_speedup"]["overall"]["applies_when"])
                self.assertTrue(record["potential_speedup"]["overall"]["measure"])
                self.assertTrue(record["next_validation"])
                self.assertTrue(set(record["source_ids"]).issubset(sources | local_sources))

        decoder = next(record for record in outcomes["records"] if record["id"] == "decoder-mlx-lm-working-route")
        overall = decoder["potential_speedup"]["overall"]
        self.assertIsNone(overall["range"])
        self.assertEqual(overall["provenance"], "profile_required")
        self.assertEqual(overall["observed_source_range"], "1.0x-4.3x")
        self.assertEqual(overall["claim_status"], "held")
        speculative = decoder["potential_speedup"]["speculative_decoding"]
        self.assertIsNone(speculative["range"])
        self.assertEqual(speculative["provenance"], "performance_observation")
        self.assertEqual(speculative["observed_source_range"], "1.0x-1.3x")

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
            self.assertEqual(manifest["graph"]["op_types"], {"MatMul": 1, "NonMaxSuppression": 1})
            self.assertEqual(manifest["operator_coverage"]["covered"], {"MatMul": 1})
            self.assertEqual(manifest["operator_coverage"]["unsupported_or_unclassified"], {"NonMaxSuppression": 1})
            self.assertEqual(manifest["operator_coverage"]["covered_count"], 1)
            self.assertEqual(manifest["operator_coverage"]["unsupported_or_unclassified_count"], 1)
            self.assertEqual(manifest["graph"]["inputs"][0]["shape"], [1, 2])
            self.assertEqual(manifest["graph"]["outputs"][0]["dtype"], "FLOAT")
            self.assertEqual(manifest["graph"]["initializers"][0]["shape"], [2, 3])
            self.assertEqual(manifest["graph"]["initializers"][0]["parameters"], 6)
            self.assertEqual(manifest["graph"]["initializers"][0]["raw_data_bytes"], 24)
            self.assertTrue(any("NonMaxSuppression" in condition for condition in manifest["hold_conditions"]))
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

    def test_static_inspection_holds_safetensors_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "source_formats" / "safetensors_only", "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["safetensors-checkpoint"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "safetensors-checkpoint")
            self.assertEqual(manifest["safetensors_files"], ["model.safetensors"])
            self.assertEqual(manifest["tensor_count"], 3)
            self.assertIn("lm_head.weight", manifest["tensor_key_samples"])
            self.assertEqual(report["architecture_candidates"][0]["family"], "dense-decoder-transformer")
            self.assertIsNone(report["recommended_family"])
            self.assertTrue(any("Missing config.json" in condition for condition in manifest["hold_conditions"]))
            self.assertIn("safetensors checkpoint is missing required config or provenance metadata", report["recommendation_blockers"])

    def test_static_inspection_reports_flax_orbax_source_format_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "source_formats" / "flax_orbax", "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["flax-orbax"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "flax-orbax")
            self.assertIn("flax_model.msgpack", manifest["msgpack_files"])
            self.assertIn("tree_metadata.json", manifest["metadata_files"])
            self.assertIn("params.encoder.layer_0.kernel", manifest["tree_paths"])
            self.assertIn("orbax_checkpoint/_CHECKPOINT_METADATA", manifest["metadata"])
            self.assertIn("source-format static intake is triage-only", report["recommendation_blockers"][0])

    def test_static_inspection_reports_tensorflow_savedmodel_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "source_formats" / "tensorflow_saved_model", "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["tensorflow-savedmodel"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "tensorflow-savedmodel")
            self.assertEqual(manifest["saved_model_files"], ["saved_model.pbtxt"])
            self.assertIn("serving_default", manifest["signature_keys"])
            self.assertIn("tensorflow/serving/predict", manifest["method_names"])
            self.assertEqual(manifest["operator_counts"], {"MatMul": 1, "NonMaxSuppressionV5": 1})
            self.assertEqual(manifest["operator_coverage"]["covered"], {"MatMul": 1})
            self.assertEqual(manifest["operator_coverage"]["unsupported_or_unclassified"], {"NonMaxSuppressionV5": 1})
            self.assertTrue(manifest["variables"]["present"])
            self.assertIn("variables/variables.index", manifest["variables"]["files"])
            self.assertTrue(any("NonMaxSuppressionV5" in condition for condition in manifest["hold_conditions"]))
            self.assertIn("source-format static intake is triage-only", report["recommendation_blockers"][0])

    def test_static_inspection_reports_keras_archive_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "source_formats" / "keras_archive", "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["keras-archive"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "keras-archive")
            self.assertEqual(manifest["metadata"]["keras_version"], "3.0.0")
            self.assertEqual(manifest["class_name"], "Functional")
            self.assertEqual(manifest["layer_count"], 3)
            self.assertEqual(manifest["layer_class_names"], ["InputLayer", "Dense", "Lambda"])
            self.assertEqual(manifest["layer_coverage"]["covered"], {"Dense": 1, "InputLayer": 1})
            self.assertEqual(manifest["layer_coverage"]["unsupported_or_unclassified"], {"Lambda": 1})
            self.assertEqual(manifest["weight_files"], ["model.weights.h5"])
            self.assertTrue(any("Lambda" in condition for condition in manifest["hold_conditions"]))
            self.assertIn("source-format static intake is triage-only", report["recommendation_blockers"][0])

    def test_static_inspection_reports_coreml_package_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "inspection.json"
            run_script("inspect_model.py", FIXTURES / "source_formats" / "coreml_package", "--output", output)
            report = json.loads(output.read_text())
            self.assertEqual(report["source_format_summary"]["formats"], ["coreml-package"])
            manifest = report["source_format_summary"]["manifests"][0]
            self.assertEqual(manifest["format"], "coreml-package")
            self.assertEqual(manifest["manifest"]["fileFormatVersion"], "1.0.0")
            self.assertEqual(manifest["manifest"]["rootModelIdentifier"], "model")
            self.assertEqual(manifest["manifest"]["item_count"], 2)
            self.assertEqual(manifest["model_files"], ["Data/com.apple.CoreML/model.mlmodel"])
            self.assertEqual(manifest["weight_files"], ["Data/com.apple.CoreML/weights/weight.bin"])
            self.assertEqual(manifest["operator_coverage"]["coverage_state"], "unavailable")
            self.assertIn("Core ML protobuf/spec decoding", manifest["operator_coverage"]["reason"])
            self.assertIn("source-format static intake is triage-only", report["recommendation_blockers"][0])

    def test_port_plan_is_architecture_and_evidence_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            recommendations = Path(tmp) / "recommendations.json"
            plan = Path(tmp) / "PORT_PLAN.md"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", inspection)
            run_script("recommend_optimizations.py", inspection, "--output", recommendations)
            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                FIXTURES / "models" / "decoder",
                "--recommendations",
                recommendations,
                "--output",
                plan,
            )
            text = plan.read_text()
            self.assertIn("dense-decoder-transformer", text)
            self.assertIn("runbook-decoder-transformer.md", text)
            self.assertIn("Source oracle", text)
            self.assertIn("Optimization advice", text)
            self.assertIn("recommend_optimizations.py", text)
            self.assertNotIn("Expected effect", text)
            self.assertIn("Experimental approaches", text)
            self.assertIn("Unreviewed research signals (experimental/review queue)", text)
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
            self.assertEqual(report["schema_version"], 2)
            stack = report["planning_stack"]
            self.assertEqual(stack["id"], "dense-decoder-inference")
            compound = stack["compound"]
            hypothesis = compound["hypothesis_ceiling"]
            self.assertEqual(hypothesis["floor"], "1.0x")
            self.assertIsNone(hypothesis["ceiling"])
            self.assertEqual(hypothesis["provenance"], "withheld")
            self.assertEqual(compound["composition_status"], "withheld")
            self.assertTrue(compound["withheld_reasons"])
            self.assertNotIn("measured", compound)
            rejected = stack["rejected_observations"][0]
            self.assertEqual(rejected["decision"], "rejected")
            self.assertEqual(rejected["measured_ratio"], "0.33x")
            ready_ids = {item["id"] for item in report["ready_candidates"]}
            research_ids = {item["id"] for item in report["research_candidates"]}
            self.assertIn("fast-sdpa", ready_ids)
            self.assertNotIn("moe-expert-dispatch-and-quantization", ready_ids)
            self.assertIn("eagle-medusa-mtp-drafters", research_ids)
            self.assertNotIn("prompt-lookup-ngram-speculation", research_ids)
            text = markdown.read_text()
            self.assertIn("Optimization experiment plan", text)
            self.assertIn("Numeric compound: **withheld**", text)
            self.assertIn("Rejected observed configurations", text)
            self.assertNotIn("Hypothesis ceiling: `2.64x`", text)
            self.assertNotIn("Measured together: `0.33x`", text)
            self.assertIn("Validated by source or theory", text)
            self.assertIn("Experimental approaches", text)

    def test_optimization_recommender_selects_specific_exact_stack_for_moe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inspection = tmp_path / "inspection.json"
            output = tmp_path / "recommendations.json"
            profile = tmp_path / "target-profile.json"
            profile.write_text(json.dumps({
                "schema_version": 1,
                "hardware": {},
                "software": {},
                "capabilities": [],
                "workloads": ["concurrent-serving"],
            }))
            run_script("inspect_model.py", FIXTURES / "models" / "moe", "--output", inspection)
            run_script(
                "recommend_optimizations.py",
                inspection,
                "--target-profile",
                profile,
                "--output",
                output,
                "--limit",
                5,
            )
            report = json.loads(output.read_text())
            self.assertEqual(report["family"], "moe-decoder-transformer")
            self.assertEqual(report["planning_stack"]["id"], "moe-serving")

    def test_workload_specific_vlm_stack_is_not_selected_by_family_or_memory_objective_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps(trusted_inspection_fixture(
                ["vision-language-omni"],
                model_type="llava",
            )))
            report = json.loads(run_script(
                "recommend_optimizations.py",
                inspection,
                "--objective",
                "peak-memory",
            ).stdout)
            self.assertNotIn("planning_stack", report)
            self.assertNotIn(
                "vision-feature-cache",
                {candidate["id"] for candidate in report["ready_candidates"]},
            )

    def test_compose_stack_band_matches_shared_fixture_and_excludes_conflicts(self) -> None:
        case = load_structured(FIXTURES / "stack_compose_case.json")
        actual = compose_stack_band(
            case["stack"],
            case["guidance_methods"],
            case["receipt_assessments"],
        )
        self.assertEqual(actual["composition_status"], "withheld")
        self.assertIsNone(actual["hypothesis_ceiling"]["ceiling"])
        self.assertEqual(actual["hypothesis_ceiling"]["provenance"], "withheld")
        self.assertTrue(any("known-conflicting" in reason for reason in actual["withheld_reasons"]))
        self.assertNotIn("measured", actual)
        self.assertEqual(actual["measured_evidence"][0]["classification"], "performance_observation")
        self.assertEqual(actual["other_metric_upside"], [])
        self.assertTrue(
            any("effective_claims numeric authority" in reason for reason in actual["withheld_reasons"])
        )

    def test_parse_band_rejects_malformed_input(self) -> None:
        for value in ("1.0-3.0x", "fast", "3.0x-1.0x", "0x-1.0x"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_band(value)

    def test_advisor_taxonomy_preserves_experimental_opt_in(self) -> None:
        skill_text = (SKILL / "SKILL.md").read_text()
        taxonomy = json.loads((SKILL / "assets" / "recommendation-taxonomy.yaml").read_text())
        contributor = json.loads((SKILL / "assets" / "contributor_learnings.json").read_text())
        backlog = json.loads((SKILL / "assets" / "research_backlog.json").read_text())

        self.assertIn("This is an experimental approach. Do you want to try it?", skill_text)
        buckets = {bucket["id"]: bucket for bucket in taxonomy["advisor_buckets"]}
        self.assertTrue(buckets["experimental-approach"]["requires_user_opt_in"])
        self.assertEqual(
            taxonomy["status_to_advisor_bucket"]["research-candidate"],
            "experimental-approach",
        )
        self.assertIn(
            "layer-range-network-sharding",
            contributor["advisor_buckets"]["experimental_approaches"]["learning_ids"],
        )
        self.assertIn(
            "block-weight-streaming",
            contributor["advisor_buckets"]["validated"]["learning_ids"],
        )
        self.assertEqual(
            backlog["advisor_status_mapping"]["needs-validation"]["advisor_bucket"],
            "experimental-approach",
        )
        self.assertTrue(backlog["advisor_status_mapping"]["needs-validation"]["requires_user_opt_in"])

    def test_model_advisor_playbook_covers_recent_research_branches(self) -> None:
        skill_text = (SKILL / "SKILL.md").read_text()
        playbook = (SKILL / "references" / "model-advisor-playbook.md").read_text()

        self.assertIn("model-advisor-playbook.md", skill_text)
        for required in [
            "contributor_learnings.json",
            "research_backlog.json",
            "knowledge_graph.json",
            "recommendation-taxonomy.yaml",
            "optimization_guidance.yaml",
            "This is an experimental approach. Do you want to try it?",
            "block-weight-streaming",
            "spatial-grid-sample-kernel",
            "audio-flow-tts-duration-vocoder-gates",
            "layer-eval-watchdog-guard",
            "layer-range-network-sharding",
            "rwkv-ssm-conformer-prototypes",
            "source-formats-beyond-pytorch-hf",
            "non-generative-cv",
            "structured-timeseries-recsys",
            "graph-geometric-scientific",
            "training-as-port-target",
        ]:
            self.assertIn(required, playbook)

    def test_block_weight_streaming_is_scoped_to_repeated_block_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps(trusted_inspection_fixture(["diffusion-flow"])))
            diffusion = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "peak-memory", "--limit", 50,
            ).stdout)
            diffusion_ready = {m["id"] for m in diffusion["ready_candidates"]}
            self.assertIn("block-weight-streaming", diffusion_ready)

            inspection.write_text(json.dumps(trusted_inspection_fixture(["dense-decoder-transformer"])))
            decoder = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "peak-memory", "--limit", 50,
            ).stdout)
            decoder_ids = {m["id"] for m in decoder["ready_candidates"] + decoder["research_candidates"]}
            self.assertNotIn("block-weight-streaming", decoder_ids)

    def test_grid_sample_kernel_is_scoped_to_spatial_ports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps(trusted_inspection_fixture(["diffusion-flow"])))
            diffusion = json.loads(run_script(
                "recommend_optimizations.py", inspection,
                "--objective", "latency", "--limit", 50,
            ).stdout)
            diffusion_ready = {m["id"] for m in diffusion["ready_candidates"]}
            self.assertIn("spatial-grid-sample-kernel", diffusion_ready)

            inspection.write_text(json.dumps(trusted_inspection_fixture(["dense-decoder-transformer"])))
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
            inspection.write_text(json.dumps(trusted_inspection_fixture(["dense-decoder-transformer"])))
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
            inspection.write_text(json.dumps(trusted_inspection_fixture(
                ["dense-decoder-transformer"],
                blockers=["high: remote-code auto_map present"],
            )))
            blocked = json.loads(run_script(
                "recommend_optimizations.py",
                inspection,
                "--family",
                "dense-decoder-transformer",
            ).stdout)
            self.assertTrue(blocked["blocked"])
            self.assertEqual(blocked["ready_candidates"], [])
            self.assertEqual(blocked["research_candidates"], [])
            self.assertTrue(all(not items for items in blocked["advisor_buckets"].values()))
            self.assertNotIn("planning_stack", blocked)
            self.assertEqual(blocked["held_planning_stack"]["id"], "dense-decoder-inference")
            self.assertTrue(blocked["held_candidates"])
            allowed = json.loads(run_script(
                "recommend_optimizations.py",
                inspection,
                "--family",
                "dense-decoder-transformer",
                "--allow-blocked",
            ).stdout)
            self.assertTrue(allowed["blocked"])
            self.assertTrue(allowed["blocked_advice_visible"])
            self.assertTrue(allowed["ready_candidates"])
            self.assertTrue(all(not candidate["execution_allowed"] for candidate in allowed["ready_candidates"]))
            self.assertEqual(allowed["held_planning_stack"]["id"], "dense-decoder-inference")
            self.assertTrue(
                all(not step["execution_allowed"] for step in allowed["held_planning_stack"]["steps"])
            )

    def test_port_plan_never_bypasses_exact_advisor_or_effective_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            plan = Path(tmp) / "PORT_PLAN.md"
            model = FIXTURES / "models" / "autoregressive_audio_lm"
            run_script("inspect_model.py", model, "--output", inspection)
            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                model,
                "--output",
                plan,
            )
            text = plan.read_text()
            self.assertIn("recommend_optimizations.py", text)
            self.assertIn("No optimization candidates are embedded", text)
            self.assertNotIn("Proven/native optimization candidates", text)
            for held_range in ("1.67x", "5.45x", "28.0x", "4.3x"):
                self.assertNotIn(held_range, text)

    def test_recommender_requires_family_and_respects_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = Path(tmp) / "inspection.json"
            inspection.write_text(json.dumps({"recommendation_blockers": []}))
            # Missing safety/provenance fields fail closed, even with a family override.
            missing = run_script("recommend_optimizations.py", inspection, expected=2)
            self.assertIn("Untrusted inspection report", missing.stderr)
            overridden_missing = run_script(
                "recommend_optimizations.py",
                inspection,
                "--family",
                "moe-decoder-transformer",
                expected=2,
            )
            self.assertIn("Untrusted inspection report", overridden_missing.stderr)
            plan_missing = run_script(
                "make_port_plan.py",
                inspection,
                "--family",
                "moe-decoder-transformer",
                "--output",
                Path(tmp) / "PORT_PLAN.md",
                expected=2,
            )
            self.assertIn("Untrusted inspection report", plan_missing.stderr)

            inspection.write_text(json.dumps(trusted_inspection_fixture(["dense-decoder-transformer"])))
            # A family override cannot invent a route that static inspection did not observe.
            output = Path(tmp) / "rec.json"
            result = run_script(
                "recommend_optimizations.py", inspection,
                "--family", "moe-decoder-transformer", "--output", output,
                expected=2,
            )
            self.assertIn("not present in the trusted inspection candidates", result.stderr)
            self.assertFalse(output.exists())
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
            recommendations = Path(tmp) / "recommendations.json"
            plan = Path(tmp) / "PORT_PLAN.md"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", inspection)
            run_script("recommend_optimizations.py", inspection, "--output", recommendations)
            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                FIXTURES / "models" / "decoder",
                "--recommendations",
                recommendations,
                "--output",
                plan,
            )
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

            rec = tmp / "rec.json"
            run_script("recommend_optimizations.py", inspection, "--output", rec)
            self.assertEqual(json.loads(rec.read_text())["schema_version"], 2)
            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                FIXTURES / "models" / "decoder",
                "--recommendations",
                rec,
                "--output",
                tmp / "PORT_PLAN.md",
            )

            # inspection.json is itself a valid --source manifest for validate_weight_map.
            target = {"tensors": [{"key": t["key"], "shape": t["shape"]} for t in data["tensors"]]}
            mapping = {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [
                    {
                        "source": t["key"],
                        "source_shape": t["shape"],
                        "target": t["key"],
                        "target_shape": t["shape"],
                        "transforms": [],
                    }
                    for t in data["tensors"]
                ],
                "ignore": [],
                "unresolved": [],
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

    def test_mlx_project_inspector_reports_existing_runtime_and_contribution_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project = tmp_path / "served_port"
            project.mkdir()
            (project / "serve.py").write_text(
                "\n".join([
                    "import time",
                    "import mlx.core as mx",
                    "import mlx.nn as nn",
                    "from mlx_lm import load, generate",
                    "",
                    "class TinyPort(nn.Module):",
                    "    def __call__(self, x):",
                    "        return mx.fast.scaled_dot_product_attention(x, x, x)",
                    "",
                    "@mx.compile",
                    "def step(x):",
                    "    return mx.eval(x)",
                    "",
                    "def benchmark_and_check():",
                    "    started = time.perf_counter()",
                    "    assert mx.allclose(mx.array([1.0]), mx.array([1.0]))",
                    "    model, tokenizer = load('local-model')",
                    "    generate(model, tokenizer, prompt='hello')",
                    "    return {'tok/s': 1.0 / (time.perf_counter() - started)}",
                    "",
                    "kernel = mx.metal_kernel(name='tiny', input_names=['x'], output_names=['y'], source='')",
                ]) + "\n",
                encoding="utf-8",
            )
            output = tmp_path / "inspection.json"
            markdown = tmp_path / "MLX_INSPECTION.md"
            result = run_script("inspect_mlx_project.py", project, "--output", output, "--markdown", markdown)
            self.assertEqual(result.stdout.strip(), "")
            report = json.loads(output.read_text())
            self.assertEqual(report["inspection_mode"], "local-mlx-project-inspector")
            self.assertEqual(report["health"]["status"], "looks-good")
            self.assertIn("mlx", report["code_surface"]["packages"])
            self.assertIn("mlx_lm", report["code_surface"]["packages"])
            features = report["code_surface"]["features"]
            self.assertTrue(features["compiled_regions"])
            self.assertTrue(features["explicit_eval"])
            self.assertTrue(features["fast_attention"])
            self.assertTrue(features["benchmark_evidence"])
            self.assertTrue(features["parity_evidence"])
            self.assertTrue(features["custom_metal_kernel"])
            candidate_ids = {item["id"] for item in report["contribution_candidates"]}
            self.assertIn("custom-metal-kernel-pattern", candidate_ids)
            self.assertIn("compiled-region-receipt", candidate_ids)
            self.assertIn("Contribution candidates", markdown.read_text())

    def test_mlx_project_inspector_flags_missing_parity_and_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project = tmp_path / "rough_port"
            project.mkdir()
            (project / "app.py").write_text(
                "import mlx.core as mx\n\n"
                "def run(x):\n"
                "    return mx.matmul(x, x)\n",
                encoding="utf-8",
            )
            report = json.loads(run_script("inspect_mlx_project.py", project).stdout)
            self.assertEqual(report["health"]["status"], "proof-gaps")
            opportunity_ids = {item["id"] for item in report["improvement_opportunities"]}
            self.assertIn("add-source-oracle", opportunity_ids)
            self.assertIn("add-benchmark-receipt", opportunity_ids)
            self.assertIn("review-eval-boundaries", opportunity_ids)
            self.assertEqual(report["contribution_candidates"][0]["id"], "no-new-learning-yet")

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

    def test_weight_mapping_accepts_onnx_source_format_initializers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_dir = tmp / "onnx"
            model_dir.mkdir()
            (model_dir / "model.onnx").write_bytes(tiny_onnx_model())
            source = tmp / "inspection.json"
            target = tmp / "target.json"
            mapping = tmp / "map.json"
            output = tmp / "weight-report.json"
            run_script("inspect_model.py", model_dir, "--output", source)
            target.write_text(json.dumps({"tensors": [{"key": "linear.weight", "shape": [2, 3]}]}))
            mapping.write_text(json.dumps({
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "weight",
                    "source_shape": [2, 3],
                    "target": "linear.weight",
                    "target_shape": [2, 3],
                    "transforms": [],
                }],
                "ignore": [],
                "unresolved": [],
            }))
            run_script("validate_weight_map.py", "--source", source, "--target", target, "--mapping", mapping, "--output", output)
            report = json.loads(output.read_text())
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["source_tensors"], 1)
            self.assertEqual(report["checks"][0]["source"], "weight")
            self.assertEqual(report["checks"][0]["source_shape"], [2, 3])

    def test_weight_mapping_accepts_gguf_source_format_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_dir = tmp / "gguf"
            model_dir.mkdir()
            (model_dir / "model.gguf").write_bytes(tiny_gguf_model())
            source = tmp / "inspection.json"
            target = tmp / "target.json"
            mapping = tmp / "map.json"
            output = tmp / "weight-report.json"
            run_script("inspect_model.py", model_dir, "--output", source)
            target.write_text(json.dumps({"tensors": [{"key": "q_proj.weight", "shape": [2, 3]}]}))
            mapping.write_text(json.dumps({
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "blk.0.attn_q.weight",
                    "source_shape": [2, 3],
                    "target": "q_proj.weight",
                    "target_shape": [2, 3],
                    "transforms": [],
                }],
                "ignore": [],
                "unresolved": [],
            }))
            run_script("validate_weight_map.py", "--source", source, "--target", target, "--mapping", mapping, "--output", output)
            report = json.loads(output.read_text())
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["source_tensors"], 1)
            self.assertEqual(report["checks"][0]["source"], "blk.0.attn_q.weight")
            self.assertEqual(report["checks"][0]["source_shape"], [2, 3])

    def test_weight_mapping_rejects_source_format_without_static_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "inspection.json"
            target = tmp / "target.json"
            mapping = tmp / "map.json"
            run_script("inspect_model.py", FIXTURES / "source_formats" / "keras_archive", "--output", source)
            target.write_text(json.dumps({"tensors": []}))
            mapping.write_text(json.dumps({
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [],
                "ignore": [],
                "unresolved": [],
            }))
            result = run_script(
                "validate_weight_map.py",
                "--source", source,
                "--target", target,
                "--mapping", mapping,
                expected=2,
            )
            self.assertIn("do not expose static tensor shapes", result.stderr)

    def test_weight_mapping_rejects_duplicate_and_unknown_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mapping = Path(tmp) / "bad-map.json"
            mapping.write_text(json.dumps({
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [
                    {
                        "source": "linear.weight",
                        "source_shape": [3, 2],
                        "target": "layer.weight",
                        "target_shape": [2, 3],
                        "transforms": [{"op": "transpose", "axes": [1, 0]}],
                    },
                    {
                        "source": "linear.weight",
                        "source_shape": [3, 2],
                        "target": "layer.bias",
                        "target_shape": [3],
                        "transforms": [],
                    },
                ],
                "ignore": [],
                "unresolved": [],
            }))
            result = run_script(
                "validate_weight_map.py",
                "--source", FIXTURES / "manifests" / "source.json",
                "--target", FIXTURES / "manifests" / "target.json",
                "--mapping", mapping,
                expected=2,
            )
            self.assertIn("source tensor mapped more than once", result.stderr)

            mapping.write_text(json.dumps({
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [
                    {
                        "source": "linear.weight",
                        "source_shape": [3, 2],
                        "target": "layer.weight",
                        "target_shape": [2, 3],
                        "transforms": [{"op": "transpose", "axes": [1, 0]}],
                    },
                    {
                        "source": "linear.bias",
                        "source_shape": [3],
                        "target": "layer.bias",
                        "target_shape": [3],
                        "transforms": [],
                    },
                ],
                "ignore": [{"source": "missing.source", "reason": "invalid fixture"}],
                "unresolved": [],
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
            self.assertTrue(any("ignored source tensors are missing" in error for error in report["errors"]))

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
            mapping = {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [
                    {"source": "reshape.w", "source_shape": [4, 3], "target": "reshape.t", "target_shape": [2, 6], "transforms": [{"op": "reshape", "shape": [-1, 6]}]},
                    {"source": "squeeze.w", "source_shape": [1, 5], "target": "squeeze.t", "target_shape": [5], "transforms": [{"op": "squeeze", "axis": 0}]},
                    {"source": "unsqueeze.w", "source_shape": [5], "target": "unsqueeze.t", "target_shape": [1, 5], "transforms": [{"op": "unsqueeze", "axis": 0}]},
                    {"source": "slice.w", "source_shape": [10], "target": "slice.t", "target_shape": [3], "transforms": [{"op": "slice", "axis": 0, "start": 2, "end": 8, "step": 2}]},
                    {"source": "permute.w", "source_shape": [2, 3, 4], "target": "permute.t", "target_shape": [4, 2, 3], "transforms": [{"op": "permute", "axes": [2, 0, 1]}]},
                ],
                "ignore": [],
                "unresolved": [],
            }
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
            mapping = {
                "schema_version": 2,
                "dtype_policy": "keep",
                "entries": [{
                    "source": "x",
                    "source_shape": [2, 3],
                    "target": "y",
                    "target_shape": [3],
                    "transforms": [{"op": "squeeze", "axis": 0}],
                }],
                "ignore": [],
                "unresolved": [],
            }
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

    def test_generation_benchmark_receipt_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmarks" / "fixture.json"
            stub = (
                "print('Prompt: 512 tokens, 800.0 tokens-per-sec'); "
                "print('Generation: 128 tokens, 50.0 tokens-per-sec'); "
                "print('Peak memory: 1.5 GB')"
            )
            run_script(
                "benchmark_generation.py",
                "--label", "fixture",
                "--warmup", "0",
                "--runs", "1",
                "--output", output,
                "--config-note", "quant=4bit",
                "--",
                sys.executable, "-c", stub,
            )

            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["label"], "fixture")
            self.assertIn("platform", receipt["environment"])
            self.assertEqual(receipt["versions"].keys(), {"mlx", "mlx_lm"})
            self.assertEqual(receipt["config_notes"], {"quant": "4bit"})
            self.assertEqual(receipt["runs"][0]["prompt_tokens"], 512)
            self.assertEqual(receipt["runs"][0]["prompt_tps"], 800.0)
            self.assertEqual(receipt["runs"][0]["generation_tokens"], 128)
            self.assertEqual(receipt["runs"][0]["generation_tps"], 50.0)
            self.assertEqual(receipt["runs"][0]["peak_memory_gb"], 1.5)
            self.assertEqual(receipt["aggregate"]["generation_tps"]["median"], 50.0)
            self.assertEqual(receipt["ttft_proxy"]["metric"], "ttft_proxy_s")
            self.assertAlmostEqual(receipt["ttft_proxy"]["median_s"], 0.64)
            self.assertIn("proxy_note", receipt["ttft_proxy"])
            self.assertNotIn("speedup_vs_baseline", receipt)

            index = json.loads((output.parent / "receipts_index.json").read_text())
            self.assertEqual(index["receipts"][0]["label"], "fixture")
            self.assertEqual(index["receipts"][0]["file"], "fixture.json")
            self.assertEqual(index["receipts"][0]["config_notes"], {"quant": "4bit"})
            self.assertIn("-c", index["receipts"][0]["command_summary"])

    def test_generation_benchmark_reports_subprocess_failure_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixture.json"
            stub = "import sys; print('before fail'); print('bad stderr', file=sys.stderr); sys.exit(1)"
            result = run_script(
                "benchmark_generation.py",
                "--label", "fixture",
                "--warmup", "0",
                "--runs", "1",
                "--output", output,
                "--",
                sys.executable, "-c", stub,
                expected=1,
            )
            self.assertIn("measured run 1 failed", result.stderr)
            self.assertIn("before fail", result.stderr)
            self.assertIn("bad stderr", result.stderr)
            self.assertFalse(output.exists())

    def test_generation_benchmark_speedup_requires_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            baseline.write_text(json.dumps({
                "label": "baseline",
                "aggregate": {
                    "generation_tps": {"median": 25.0},
                    "prompt_tps": {"median": 400.0},
                    "ttft_proxy_s": {"median": 2.0},
                },
            }))
            output = tmp_path / "candidate.json"
            stub = (
                "print('Prompt: 512 tokens, 800.0 tokens-per-sec'); "
                "print('Generation: 128 tokens, 50.0 tokens-per-sec'); "
                "print('Peak memory: 1.5 GB')"
            )
            run_script(
                "benchmark_generation.py",
                "--label", "candidate",
                "--warmup", "0",
                "--runs", "1",
                "--baseline-receipt", baseline,
                "--output", output,
                "--",
                sys.executable, "-c", stub,
            )
            receipt = json.loads(output.read_text())
            ratios = receipt["speedup_vs_baseline"]["ratios"]
            self.assertEqual(receipt["speedup_vs_baseline"]["baseline_label"], "baseline")
            self.assertAlmostEqual(ratios["decode_tps"], 2.0)
            self.assertAlmostEqual(ratios["prefill_tps"], 2.0)
            self.assertAlmostEqual(ratios["ttft_proxy_inverse"], 3.125)

    def test_generation_benchmark_fails_on_missing_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixture.json"
            stub = "print('Prompt: 512 tokens, 800.0 tokens-per-sec')"
            result = run_script(
                "benchmark_generation.py",
                "--label", "fixture",
                "--warmup", "0",
                "--runs", "1",
                "--output", output,
                "--",
                sys.executable, "-c", stub,
                expected=2,
            )
            self.assertIn("missing benchmark output pattern", result.stderr)
            self.assertIn("Generation: <tokens> tokens", result.stderr)
            self.assertIn("Peak memory: <GB> GB", result.stderr)
            self.assertFalse(output.exists())

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

    def test_knowledge_curator_builds_delta_graph_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidates = {
                "repositories": [
                    {
                        "kind": "repository",
                        "repo": "fixture/new-mlx-speedups",
                        "head_sha": "abc123",
                        "head_date": "2026-06-30T00:00:00Z",
                        "head_message": "Add speculative decoding draft cache experiments",
                        "url": "https://github.com/fixture/new-mlx-speedups",
                        "topics": ["mlx", "speculative", "decoding"],
                    }
                ],
                "papers": [
                    {
                        "kind": "paper",
                        "query": "speculative decoding",
                        "id": "https://arxiv.org/abs/2211.17192v1",
                        "title": "Fast Inference from Transformers via Speculative Decoding",
                        "updated": "2022-11-29T00:00:00Z",
                        "published": "2022-11-29T00:00:00Z",
                        "authors": ["Fixture Author"],
                        "summary": "Known source fixture for speculative decoding.",
                    },
                    {
                        "kind": "paper",
                        "query": "MLX speculative decoding",
                        "id": "https://arxiv.org/abs/2606.99999",
                        "title": "Fixture Speculative Decoding for MLX",
                        "updated": "2026-06-30T00:00:00Z",
                        "published": "2026-06-30T00:00:00Z",
                        "authors": ["Fixture Author"],
                        "summary": "Draft model speculative decoding acceptance for MLX.",
                    },
                ],
            }
            candidates_path = tmp_path / "candidates.json"
            candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
            graph_path = tmp_path / "knowledge_graph.json"
            delta_path = tmp_path / "knowledge-delta.json"
            markdown_path = tmp_path / "knowledge-delta.md"

            run_script(
                "knowledge_curator.py",
                "--run-id", "fixture-curator",
                "--update-candidates", candidates_path,
                "--previous-graph", tmp_path / "missing-previous-graph.json",
                "--graph-output", graph_path,
                "--delta-output", delta_path,
                "--markdown-output", markdown_path,
            )
            graph = json.loads(graph_path.read_text())
            delta = json.loads(delta_path.read_text())

            self.assertTrue(graph["policy"]["review_only"])
            self.assertGreater(graph["node_count"], 0)
            self.assertGreater(graph["edge_count"], 0)
            self.assertTrue(delta["policy"]["review_only"])
            self.assertTrue(delta["gap_hints"])
            updated_by_locator = {
                item["locator"]: item for item in delta["updated_sources"]
            }
            unpinned_known = updated_by_locator["https://arxiv.org/abs/2211.17192"]
            self.assertEqual(unpinned_known["revision"], "v1")
            self.assertEqual(
                unpinned_known["immutable_locator"],
                "https://arxiv.org/abs/2211.17192v1",
            )
            self.assertEqual(unpinned_known["revision_comparison"], {
                "before": None,
                "after": "v1",
                "status": "comparison_unpinned",
                "basis": "source_registry",
            })
            self.assertTrue(any(
                edge["source"] == unpinned_known["id"]
                and edge["target"] == unpinned_known["known_source_id"]
                and edge["relation"] == "candidate_version_of"
                for edge in graph["edges"]
            ))
            unread_labels = {item["label"] for item in delta["new_unread_sources"]}
            self.assertIn("Fixture Speculative Decoding for MLX", unread_labels)
            lead_targets = {
                match["target"]
                for lead in delta["new_approach_leads"]
                for match in lead["matches"]
            }
            self.assertIn("approach:draft-model-speculation", lead_targets)
            self.assertIn("New Approach Leads", markdown_path.read_text())

    def test_knowledge_curator_retains_candidates_missing_from_bounded_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidates = {"repositories": [], "papers": []}
            previous = {
                "nodes": [
                    {
                        "id": "candidate:paper:retained-paper",
                        "kind": "source_candidate",
                        "candidate_kind": "paper",
                        "label": "Retained paper",
                        "locator": "https://arxiv.org/abs/2601.00001",
                        "read_state": "unread_candidate",
                    }
                ],
                "edges": [
                    {
                        "source": "candidate:paper:retained-paper",
                        "target": "approach:retained-approach",
                        "relation": "candidate_relevant_to",
                        "score": 3,
                    },
                    {
                        "source": "candidate:paper:retained-paper",
                        "target": "source:retained-paper",
                        "relation": "candidate_version_of",
                    },
                ],
            }
            inputs = {
                "sources.json": {
                    "sources": [{
                        "id": "retained-paper",
                        "title": "Retained paper",
                        "url": "https://arxiv.org/abs/2601.00001v1",
                        "kind": "paper",
                        "review_depth": "indexed",
                    }],
                },
                "candidates.json": candidates,
                "guidance.json": {"methods": [{"id": "retained-approach"}]},
                "learnings.json": {"learnings": []},
                "backlog.json": {"items": []},
                "outcomes.json": {"records": []},
                "previous.json": previous,
            }
            for name, payload in inputs.items():
                (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

            graph_path = tmp_path / "knowledge_graph.json"
            run_script(
                "knowledge_curator.py",
                "--sources", tmp_path / "sources.json",
                "--update-candidates", tmp_path / "candidates.json",
                "--optimization-guidance", tmp_path / "guidance.json",
                "--contributor-learnings", tmp_path / "learnings.json",
                "--research-backlog", tmp_path / "backlog.json",
                "--model-outcomes", tmp_path / "outcomes.json",
                "--previous-graph", tmp_path / "previous.json",
                "--graph-output", graph_path,
                "--delta-output", tmp_path / "knowledge-delta.json",
                "--markdown-output", tmp_path / "knowledge-delta.md",
            )

            graph = json.loads(graph_path.read_text())
            node_ids = {node["id"] for node in graph["nodes"]}
            self.assertIn("candidate:paper:retained-paper", node_ids)
            edges = {
                (edge["source"], edge["target"], edge["relation"])
                for edge in graph["edges"]
            }
            self.assertIn(
                ("candidate:paper:retained-paper", "approach:retained-approach", "candidate_relevant_to"),
                edges,
            )
            self.assertIn(
                ("candidate:paper:retained-paper", "source:retained-paper", "candidate_version_of"),
                edges,
            )

    def test_knowledge_curator_ingests_contributor_refresh_as_review_queue(self) -> None:
        import knowledge_curator

        refresh = knowledge_curator.validate_contributor_refresh({
            "schema_version": 1,
            "generated_at": "2026-07-10T00:00:00+00:00",
            "review_only": True,
            "repo": "ml-explore/mlx",
            "source": "https://api.github.com/repos/ml-explore/mlx/contributors",
            "retrieved": "2026-07-10",
            "requested_count": 1000,
            "linked_user_count": 2,
            "anonymous_author_count": 3,
            "top_logins": ["alice", "bob"],
            "api_receipt": {},
        })
        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []
        knowledge_curator.ingest_contributor_refresh(nodes, edges, refresh)

        collection = nodes["review-queue:contributors:ml-explore-mlx"]
        self.assertTrue(collection["review_only"])
        self.assertEqual(collection["anonymous_author_count"], 3)
        self.assertIn("candidate:contributor:alice", nodes)
        self.assertIn("candidate:contributor:bob", nodes)
        self.assertFalse(any("anon" in node_id for node_id in nodes if node_id.startswith("candidate:")))
        self.assertEqual({edge["relation"] for edge in edges}, {"contributor_in_refresh"})

    def test_knowledge_curator_reconciles_and_checks_backlog_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "knowledge_graph.json"
            candidates_path = tmp_path / "update-candidates.json"
            backlog_path = tmp_path / "research_backlog.json"
            nodes = [
                {
                    "id": "backlog:fixture-gap",
                    "kind": "backlog_item",
                    "label": "fixture-gap",
                    "status": "needs-validation",
                    "priority": "P1",
                    "decision_state": "needs-review",
                    "summary": "Fixture validation gap.",
                    "validation_gate": "Pass the fixture parity gate.",
                    "source": "research-runs/fixture/synthesis.json",
                    "affected": ["tests/test_tooling.py"],
                    "families": ["dense-decoder-transformer"],
                },
                {
                    "id": "candidate:paper:fixture",
                    "kind": "source_candidate",
                    "label": "Fixture paper",
                    "locator": "https://example.com/fixture",
                    "read_state": "unread_candidate",
                },
            ]
            graph_path.write_text(json.dumps({
                "schema_version": 1,
                "generated_at": "2026-07-10T03:00:00+00:00",
                "run_id": "fixture-reconcile",
                "policy": {
                    "review_only": True,
                    "auto_promote_sources": False,
                    "auto_modify_recommendations": False,
                },
                "node_count": len(nodes),
                "edge_count": 0,
                "nodes": nodes,
                "edges": [],
            }), encoding="utf-8")
            candidates_path.write_text(json.dumps({
                "schema_version": 1,
                "generated_at": "2026-07-10T02:00:00+00:00",
                "papers": [{"id": "fixture"}],
                "repositories": [{"repo": "fixture/repo"}],
            }), encoding="utf-8")

            run_script(
                "knowledge_curator.py",
                "--reconcile-backlog",
                "--previous-graph", graph_path,
                "--update-candidates", candidates_path,
                "--research-backlog", backlog_path,
            )
            backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
            self.assertEqual(backlog["reviewed"], "2026-07-10")
            self.assertEqual(backlog["items"][0]["id"], "fixture-gap")
            self.assertEqual(backlog["generated_from"]["paper_candidate_count"], 1)
            self.assertEqual(
                backlog["generated_from"]["candidate_read_states"],
                {"unread_candidate": 1},
            )
            run_script(
                "knowledge_curator.py",
                "--check-backlog",
                "--previous-graph", graph_path,
                "--update-candidates", candidates_path,
                "--research-backlog", backlog_path,
            )
            backlog["items"][0]["summary"] = "stale"
            backlog_path.write_text(json.dumps(backlog), encoding="utf-8")
            drift = run_script(
                "knowledge_curator.py",
                "--check-backlog",
                "--previous-graph", graph_path,
                "--update-candidates", candidates_path,
                "--research-backlog", backlog_path,
                expected=1,
            )
            self.assertIn("research backlog drift", drift.stderr)

    def test_knowledge_curator_preserves_arxiv_revision_and_detects_drift(self) -> None:
        import knowledge_curator

        source = knowledge_curator.source_node({
            "id": "fixture-versioned-paper",
            "title": "Versioned paper",
            "url": "https://arxiv.org/abs/2601.12345v1",
            "kind": "paper",
            "review_depth": "synthesized",
            "snapshot": "2601.12345v1",
        })
        known = {source["locator"]: source}
        candidate = {
            "id": "http://arxiv.org/abs/2601.12345v2",
            "title": "Versioned paper",
            "updated": "2026-07-10T00:00:00Z",
        }

        node, state = knowledge_curator.candidate_node_from_paper(
            candidate,
            known,
            set(),
            {},
        )

        self.assertEqual(state, "updated_candidate")
        self.assertEqual(node["locator"], "https://arxiv.org/abs/2601.12345")
        self.assertEqual(node["immutable_locator"], "https://arxiv.org/abs/2601.12345v2")
        self.assertEqual(node["revision"], "v2")
        self.assertEqual(node["revision_comparison"], {
            "before": "v1",
            "after": "v2",
            "status": "changed",
            "basis": "source_registry",
        })

        delta = {
            "already_read_sources": [],
            "new_unread_sources": [],
            "updated_sources": [],
            "new_approach_leads": [],
        }
        edges: list[dict[str, object]] = []
        knowledge_curator.append_candidate({}, edges, node, {}, delta)
        knowledge_curator.add_candidate_lineage(edges, node)
        self.assertEqual(delta["updated_sources"][0]["revision_comparison"], node["revision_comparison"])
        self.assertIn(
            "revision `v1` -> `v2` (changed)",
            knowledge_curator.markdown_items(delta["updated_sources"])[0],
        )
        self.assertEqual(edges, [{
            "source": node["id"],
            "target": "source:fixture-versioned-paper",
            "relation": "candidate_version_of",
            "revision_comparison": node["revision_comparison"],
        }])

        same, same_state = knowledge_curator.candidate_node_from_paper(
            candidate | {"id": "https://arxiv.org/abs/2601.12345v1"},
            known,
            set(),
            {},
        )
        self.assertEqual(same_state, "already_read")
        self.assertEqual(same["revision_comparison"]["status"], "same")

    def test_knowledge_curator_keeps_superseded_methods_out_of_new_leads(self) -> None:
        import knowledge_curator

        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []
        terms = knowledge_curator.build_approaches(
            nodes,
            edges,
            {
                "methods": [
                    {
                        "id": "active-content-cache",
                        "status": "benchmark-required",
                        "category": "multimodal-serving",
                        "recommendation": "Cache repeated multimodal content.",
                    },
                    {
                        "id": "superseded-content-cache",
                        "status": "rejected-or-superseded",
                        "category": "multimodal-serving",
                        "recommendation": "Superseded alias for repeated multimodal content caching.",
                    },
                ]
            },
            {"learnings": []},
            {"items": []},
            {"outcomes": []},
            {},
        )

        self.assertIn("approach:active-content-cache", terms)
        self.assertNotIn("approach:superseded-content-cache", terms)
        self.assertIn("approach:superseded-content-cache", nodes)

    def test_knowledge_curator_matches_pinned_repositories_and_preserves_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pinned_sha = "a" * 40
            updated_sha = "b" * 40
            sources = {
                "sources": [
                    {
                        "id": "fixture-pinned-repo",
                        "title": "Pinned repository",
                        "url": f"https://github.com/Fixture/Pinned-Repo/tree/{pinned_sha}",
                        "kind": "repository",
                        "owner": "Fixture",
                        "topics": ["mlx"],
                        "review_depth": "synthesized",
                        "snapshot": pinned_sha,
                    },
                    {
                        "id": "fixture-updated-repo",
                        "title": "Updated repository",
                        "url": f"https://github.com/Fixture/Updated-Repo/tree/{pinned_sha}",
                        "kind": "repository",
                        "owner": "Fixture",
                        "topics": ["mlx"],
                        "review_depth": "synthesized",
                        "snapshot": pinned_sha,
                    },
                ]
            }
            candidates = {
                "repositories": [
                    {
                        "repo": "fixture/pinned-repo",
                        "url": "https://github.com/fixture/pinned-repo",
                        "head_sha": pinned_sha,
                    },
                    {
                        "repo": "fixture/updated-repo",
                        "url": "https://github.com/fixture/updated-repo",
                        "head_sha": updated_sha,
                    },
                    {
                        "repo": "fixture/pending-review",
                        "url": "https://github.com/fixture/pending-review",
                        "head_sha": updated_sha,
                    },
                ],
                "papers": [],
            }
            previous = {
                "generated_at": "2026-07-09T00:00:00+00:00",
                "nodes": [
                    {
                        "id": "candidate:repository:fixture-pinned-repo",
                        "locator": "https://github.com/fixture/pinned-repo",
                        "head_sha": pinned_sha,
                    },
                    {
                        "id": "candidate:repository:fixture-updated-repo",
                        "locator": "https://github.com/fixture/updated-repo",
                        "head_sha": pinned_sha,
                    },
                    {
                        "id": "candidate:repository:fixture-pending-review",
                        "locator": "https://github.com/fixture/pending-review",
                        "head_sha": updated_sha,
                        "read_state": "updated_candidate",
                    },
                ],
            }
            inputs = {
                "sources.json": sources,
                "candidates.json": candidates,
                "guidance.json": {"methods": []},
                "learnings.json": {"learnings": []},
                "backlog.json": {"items": []},
                "outcomes.json": {"records": []},
                "previous.json": previous,
            }
            for name, payload in inputs.items():
                (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

            graph_path = tmp_path / "knowledge_graph.json"
            delta_path = tmp_path / "knowledge-delta.json"
            markdown_path = tmp_path / "knowledge-delta.md"
            run_script(
                "knowledge_curator.py",
                "--run-id", "fixture-pinned-repositories",
                "--sources", tmp_path / "sources.json",
                "--update-candidates", tmp_path / "candidates.json",
                "--optimization-guidance", tmp_path / "guidance.json",
                "--contributor-learnings", tmp_path / "learnings.json",
                "--research-backlog", tmp_path / "backlog.json",
                "--model-outcomes", tmp_path / "outcomes.json",
                "--previous-graph", tmp_path / "previous.json",
                "--graph-output", graph_path,
                "--delta-output", delta_path,
                "--markdown-output", markdown_path,
            )

            graph = json.loads(graph_path.read_text())
            delta = json.loads(delta_path.read_text())
            by_id = {node["id"]: node for node in graph["nodes"]}
            pinned = by_id["candidate:repository:fixture-pinned-repo"]
            updated = by_id["candidate:repository:fixture-updated-repo"]
            pending = by_id["candidate:repository:fixture-pending-review"]
            source = by_id["source:fixture-pinned-repo"]

            self.assertEqual(pinned["read_state"], "already_read")
            self.assertEqual(pinned["known_source_id"], "source:fixture-pinned-repo")
            self.assertEqual(pinned["locator"], "https://github.com/fixture/pinned-repo")
            self.assertEqual(
                source["locator"],
                f"https://github.com/Fixture/Pinned-Repo/tree/{pinned_sha}",
            )
            self.assertEqual(updated["read_state"], "updated_candidate")
            self.assertEqual(updated["known_source_id"], "source:fixture-updated-repo")
            self.assertEqual(pending["read_state"], "updated_candidate")
            self.assertEqual(pinned["revision_comparison"], {
                "before": pinned_sha,
                "after": pinned_sha,
                "status": "same",
                "basis": "source_registry",
            })
            self.assertEqual(updated["revision_comparison"], {
                "before": pinned_sha,
                "after": updated_sha,
                "status": "changed",
                "basis": "source_registry",
            })
            self.assertEqual(pending["revision_comparison"], {
                "before": updated_sha,
                "after": updated_sha,
                "status": "same",
                "basis": "previous_graph",
            })
            lineage = {
                (edge["source"], edge["target"], edge["relation"])
                for edge in graph["edges"]
                if edge["relation"] == "candidate_version_of"
            }
            self.assertIn(
                (pinned["id"], "source:fixture-pinned-repo", "candidate_version_of"),
                lineage,
            )
            self.assertIn(
                (updated["id"], "source:fixture-updated-repo", "candidate_version_of"),
                lineage,
            )
            self.assertIn(
                "fixture/pinned-repo",
                {item["label"] for item in delta["already_read_sources"]},
            )
            self.assertIn(
                "fixture/updated-repo",
                {item["label"] for item in delta["updated_sources"]},
            )
            updated_delta = next(
                item for item in delta["updated_sources"]
                if item["label"] == "fixture/updated-repo"
            )
            self.assertEqual(
                updated_delta["known_source_id"],
                "source:fixture-updated-repo",
            )
            self.assertEqual(updated_delta["revision_comparison"], updated["revision_comparison"])

    def test_nightly_knowledge_curator_scaffolds_graph_and_campaign_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contributor_fixture = {
                "linked_pages": [
                    {
                        "status_code": 200,
                        "headers": {"ETag": "linked-fixture"},
                        "body": [{"login": "alice"}, {"login": "bob"}],
                    }
                ],
                "anonymous_pages": [
                    {
                        "status_code": 200,
                        "headers": {"ETag": "anon-fixture"},
                        "body": [{"login": "anon-hidden"}],
                    }
                ],
            }
            contributor_fixture_path = tmp_path / "contributors.json"
            contributor_fixture_path.write_text(json.dumps(contributor_fixture), encoding="utf-8")
            output_root = tmp_path / "runs"
            update_output = tmp_path / "update-candidates.json"
            contributor_output = tmp_path / "contributor-refresh.json"
            graph_output = tmp_path / "knowledge_graph.json"
            research_backlog = tmp_path / "research_backlog.json"
            shutil.copyfile(SKILL / "assets" / "research_backlog.json", research_backlog)

            run_script(
                "nightly_knowledge_curator.py",
                "--run-id", "fixture-nightly",
                "--output-root", output_root,
                "--offline-update-fixture", FIXTURES / "updates" / "offline.json",
                "--offline-contributor-fixture", contributor_fixture_path,
                "--update-output", update_output,
                "--contributor-output", contributor_output,
                "--graph-output", graph_output,
                "--previous-graph", graph_output,
                "--research-backlog", research_backlog,
                "--agent-count", "2",
            )
            run_dir = output_root / "fixture-nightly"
            receipt = json.loads((run_dir / "nightly-run.json").read_text())
            campaign = json.loads((run_dir / "research-loop" / "campaign.json").read_text())

            self.assertTrue(receipt["review_only"])
            self.assertEqual(len(receipt["commands"]), 5)
            self.assertFalse(os.path.isabs(receipt["graph_output"]))
            self.assertFalse(os.path.isabs(receipt["delta_output"]))
            self.assertFalse(os.path.isabs(receipt["delta_markdown"]))
            self.assertFalse(os.path.isabs(receipt["research_loop_output"]))
            for command in receipt["commands"]:
                self.assertFalse(os.path.isabs(command["cwd"]))
                for item in command["command"]:
                    self.assertFalse(os.path.isabs(item))
            self.assertTrue(update_output.exists())
            self.assertTrue(contributor_output.exists())
            self.assertTrue(graph_output.exists())
            self.assertIn("generated_from", json.loads(research_backlog.read_text()))
            self.assertTrue((run_dir / "knowledge-delta.json").exists())
            self.assertTrue((run_dir / "research-loop" / "subagents.json").exists())
            self.assertEqual(campaign["wave_count"], 1)
            self.assertEqual(campaign["waves"][0]["agent_count"], 2)

    def test_contributor_collector_follows_links_caps_and_redacts_anonymous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_base = "https://api.github.test"
            linked_page_2 = f"{api_base}/repos/ml-explore/mlx/contributors?per_page=2&page=2"
            anon_page_2 = f"{api_base}/repos/ml-explore/mlx/contributors?per_page=2&page=2&anon=true"
            fixture = {
                "linked_pages": [
                    {
                        "status_code": 200,
                        "headers": {
                            "Link": f"<{linked_page_2}>; rel=\"next\", <{linked_page_2}>; rel=\"last\"",
                            "ETag": "linked-1",
                            "Last-Modified": "Sat, 27 Jun 2026 09:02:43 GMT",
                            "X-RateLimit-Limit": "60",
                            "X-RateLimit-Remaining": "58",
                            "X-RateLimit-Used": "2",
                            "X-RateLimit-Resource": "core",
                            "X-GitHub-Api-Version-Selected": "2022-11-28",
                        },
                        "body": [{"login": "alice"}, {"login": "bob"}],
                    },
                    {
                        "url": linked_page_2,
                        "status_code": 200,
                        "headers": {
                            "ETag": "linked-2",
                            "Last-Modified": "Sat, 27 Jun 2026 09:02:43 GMT",
                            "X-RateLimit-Remaining": "57",
                        },
                        "body": [{"login": "carol"}, {"login": "dropped-by-cap"}],
                    },
                ],
                "anonymous_pages": [
                    {
                        "status_code": 200,
                        "headers": {
                            "Link": f"<{anon_page_2}>; rel=\"next\", <{anon_page_2}>; rel=\"last\"",
                            "ETag": "anon-1",
                            "Last-Modified": "Sat, 27 Jun 2026 09:02:43 GMT",
                        },
                        "body": [{"login": "anon-a"}, {"login": "anon-b"}],
                    },
                    {
                        "url": anon_page_2,
                        "status_code": 200,
                        "headers": {"ETag": "anon-2"},
                        "body": [{"login": "anon-c"}, {"login": "anon-d"}],
                    },
                ],
            }
            fixture_path = tmp_path / "contributors.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            output = tmp_path / "report.json"
            run_script(
                "collect_contributors.py",
                "--requested-count", "3",
                "--per-page", "2",
                "--api-base", api_base,
                "--offline-fixture", fixture_path,
                "--output", output,
                "--access-date", "2026-06-27",
            )
            report = json.loads(output.read_text())
            self.assertTrue(report["review_only"])
            self.assertEqual(report["linked_user_count"], 3)
            self.assertEqual(report["anonymous_author_count"], 3)
            self.assertEqual(report["top_logins"], ["alice", "bob", "carol"])
            self.assertEqual(report["api_receipt"]["linked_stop_reason"], "requested_count_reached")
            self.assertEqual(report["api_receipt"]["anonymous_stop_reason"], "requested_count_reached")
            self.assertEqual(report["api_receipt"]["linked_pages"][1]["retained_count"], 1)
            first_receipt = report["api_receipt"]["linked_pages"][0]
            self.assertEqual(first_receipt["status_code"], 200)
            self.assertEqual(first_receipt["etag"], "linked-1")
            self.assertEqual(first_receipt["rate_limit"]["remaining"], "58")
            self.assertIn("next", first_receipt["link_rels"])
            self.assertNotIn("anon-a", output.read_text())
            self.assertNotIn("anon-d", output.read_text())

    def test_contributor_collector_stops_when_link_header_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = {
                "linked_pages": [
                    {
                        "status_code": 200,
                        "headers": {"ETag": "only-linked-page"},
                        "body": [{"login": "solo"}],
                    }
                ],
                "anonymous_pages": [
                    {
                        "status_code": 200,
                        "headers": {"ETag": "only-anon-page"},
                        "body": [{"login": "anon-hidden-1"}, {"login": "anon-hidden-2"}],
                    }
                ],
            }
            fixture_path = tmp_path / "contributors.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            output = tmp_path / "report.json"
            run_script(
                "collect_contributors.py",
                "--requested-count", "10",
                "--per-page", "2",
                "--api-base", "https://api.github.test",
                "--offline-fixture", fixture_path,
                "--output", output,
            )
            report = json.loads(output.read_text())
            self.assertEqual(report["linked_user_count"], 1)
            self.assertEqual(report["anonymous_author_count"], 2)
            self.assertEqual(report["pages_fetched"], 1)
            self.assertEqual(report["api_receipt"]["linked_stop_reason"], "link_header_exhausted")
            self.assertEqual(report["api_receipt"]["anonymous_stop_reason"], "link_header_exhausted")
            self.assertNotIn("anon-hidden", output.read_text())

    def test_research_loop_scaffold_includes_multi_source_sampling_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "scaffold-run"
            run_script(
                "research_loop.py",
                "--run-id", "scaffold-loop",
                "--objective", "Plan a non-GitHub MLX research pass",
                "--agent-count", 2,
                "--output-dir", output_dir,
            )
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            subagents = json.loads((output_dir / "subagents.json").read_text())
            self.assertEqual(synthesis["finding_count"], 0)
            self.assertEqual(synthesis["execution_counts"]["scaffolded_not_run"], 2)
            self.assertEqual(synthesis["assignment_planner"]["mode"], "config-order")
            self.assertEqual(synthesis["subagent_dispatch"]["execution_mode"], "offline-scaffold")
            self.assertEqual(subagents["execution_mode"], "offline-scaffold")
            self.assertEqual(subagents["agent_count"], 2)
            self.assertGreater(synthesis["planned_non_github_sample_target_count"], 0)
            self.assertGreater(synthesis["planned_source_lane_counts"]["official_docs"], 0)
            self.assertGreater(synthesis["planned_sample_target_counts"]["papers"], 0)
            self.assertEqual(synthesis["sampling_coverage"]["sampled_target_count"], 0)
            self.assertGreater(synthesis["sampling_coverage"]["unsampled_target_count"], 0)
            self.assertEqual(synthesis["evidence_matrix"]["unique_source_count"], 0)
            self.assertEqual(synthesis["evidence_matrix"]["source_citation_count"], 0)
            official_lane = next(
                lane for lane in synthesis["evidence_matrix"]["source_lanes"]
                if lane["source_lane"] == "official_docs"
            )
            self.assertEqual(official_lane["status"], "uncited")
            self.assertEqual(synthesis["review_gate"]["status"], "pass")
            self.assertIn("assignment_planner", assignments)
            first = assignments["assignments"][0]
            self.assertEqual(first["planning"]["persona_id"], "official-docs-cartographer")
            self.assertEqual(first["sampling_coverage"]["sampled_target_count"], 0)
            self.assertEqual(first["sample_plan"][0]["source_lane"], "official_docs")
            self.assertIn("evidence_role", first["sample_plan"][0])
            self.assertGreater(len(first["sample_plan"][0]["targets"]), 0)
            self.assertIn("handoff", first)
            self.assertEqual(first["handoff"]["assignment_path"], subagents["agents"][0]["assignment_path"])
            assignment_packet = json.loads((output_dir / first["handoff"]["assignment_path"]).read_text())
            self.assertEqual(assignment_packet["persona_id"], "official-docs-cartographer")
            self.assertEqual(assignment_packet["result_contract"]["path"], first["handoff"]["result_path"])
            self.assertIn("Do not execute remote model code.", assignment_packet["constraints"])
            self.assertTrue((output_dir / first["handoff"]["prompt_path"]).exists())
            self.assertIn("Sampling plan", first["prompt"])
            self.assertIn("MLX documentation index", first["prompt"])
            blog = output_dir / "blogs" / "official-docs-cartographer.md"
            blog_text = blog.read_text()
            self.assertIn("Planned sampling", blog_text)
            self.assertIn("MLX documentation index", blog_text)
            self.assertIn("## Sources sampled\n- None", blog_text)
            self.assertIn("Sampling coverage", blog_text)
            self.assertIn("Unmatched planned targets", blog_text)
            summary = (output_dir / "synthesis.md").read_text()
            self.assertIn("Review gate: pass", summary)

    def test_research_loop_dynamic_planner_selects_gap_personas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dynamic-run"
            run_script(
                "research_loop.py",
                "--run-id", "dynamic-loop",
                "--objective", "Investigate ranking recommender package release gaps beyond GitHub",
                "--agent-count", 3,
                "--gap-hint", "ranking",
                "--gap-hint", "recommender",
                "--gap-hint", "package",
                "--output-dir", output_dir,
            )
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            planner = synthesis["assignment_planner"]
            selected = [row["persona_id"] for row in planner["selected_personas"]]
            held = [row["persona_id"] for row in planner["held_personas"]]
            self.assertEqual(planner["mode"], "dynamic")
            self.assertEqual(assignments["assignment_planner"]["mode"], "dynamic")
            self.assertEqual(selected[0], "coverage-skeptic")
            self.assertIn("huggingface-ecosystem-sampler", selected)
            self.assertIn("package-registry-scout", selected)
            self.assertIn("official-docs-cartographer", held)
            self.assertNotEqual(selected, [
                "official-docs-cartographer",
                "paper-architecture-scout",
                "huggingface-ecosystem-sampler",
            ])
            self.assertGreater(planner["selected_source_lane_counts"]["hugging_face"], 0)
            self.assertGreater(planner["selected_source_lane_counts"]["packages"], 0)
            for row in planner["selected_personas"]:
                self.assertGreater(row["score"], 0)
                self.assertTrue(row["reasons"])
            summary = (output_dir / "synthesis.md").read_text()
            self.assertIn("Assignment planner: dynamic", summary)
            self.assertIn("coverage-skeptic", summary)

    def test_research_loop_writes_campaign_manifest_for_subagent_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "campaign-run"
            run_script(
                "research_loop.py",
                "--run-id", "campaign-loop",
                "--objective", "Coordinate external MLX research subagents",
                "--agent-count", 2,
                "--min-sampled-targets", 1,
                "--require-source-lane", "official_docs",
                "--output-dir", output_dir,
            )
            campaign = json.loads((output_dir / "campaign.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            subagents = json.loads((output_dir / "subagents.json").read_text())
            self.assertEqual(synthesis["campaign_manifest"]["path"], "campaign.json")
            self.assertEqual(synthesis["campaign_manifest"]["markdown_path"], "campaign.md")
            self.assertEqual(campaign["wave_count"], 1)
            self.assertTrue(campaign["review_only"])
            self.assertEqual(campaign["orchestration_contract"]["dispatch_model"], "one review-only subagent per campaign wave agent")
            wave = campaign["waves"][0]
            self.assertEqual(wave["output_dir"], ".")
            self.assertEqual(wave["subagent_manifest_path"], "subagents.json")
            self.assertEqual(wave["assignments_path"], "assignments.json")
            self.assertEqual(wave["synthesis_path"], "synthesis.json")
            self.assertEqual(wave["agent_count"], 2)
            self.assertTrue(wave["launch"]["parallel_safe"])
            self.assertEqual(wave["launch"]["max_parallel_agents"], 2)
            self.assertEqual(
                wave["launch"]["expected_result_paths"],
                [agent["result_path"] for agent in subagents["agents"]],
            )
            self.assertEqual(wave["agents"][0]["assignment_path"], subagents["agents"][0]["assignment_path"])
            self.assertEqual(wave["agents"][0]["prompt_path"], subagents["agents"][0]["prompt_path"])
            self.assertEqual(wave["agents"][0]["result_path"], subagents["agents"][0]["result_path"])
            self.assertFalse(wave["wave_dependency"]["requires_prior_wave_ingestion"])
            command_args = wave["ingest"]["command_args"]
            self.assertEqual(command_args[command_args.index("--run-id") + 1], "campaign-loop")
            self.assertEqual(command_args[command_args.index("--objective") + 1], "Coordinate external MLX research subagents")
            self.assertEqual(command_args[command_args.index("--agent-count") + 1], "2")
            self.assertEqual(command_args[command_args.index("--assignment-mode") + 1], "config-order")
            self.assertEqual(command_args[command_args.index("--min-sampled-targets") + 1], "1")
            self.assertEqual(command_args[command_args.index("--require-source-lane") + 1], "official_docs")
            self.assertEqual(command_args[command_args.index("--output-dir") + 1], str(output_dir))
            self.assertIn("--ingest-subagent-results", command_args)
            campaign_markdown = (output_dir / "campaign.md").read_text()
            self.assertIn("Research Campaign campaign-loop", campaign_markdown)
            self.assertIn("ingest command args", campaign_markdown)
            self.assertIn("official-docs-cartographer", campaign_markdown)

    def test_research_loop_campaign_command_paths_are_portable_inside_skill_root(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="campaign-portable-", dir=SKILL))
        try:
            run_script(
                "research_loop.py",
                "--run-id", "portable-campaign-loop",
                "--objective", "Keep campaign rerun commands portable",
                "--iterations", 2,
                "--agent-count", 1,
                "--output-dir", output_dir,
            )
            campaign_path = output_dir / "campaign.json"
            campaign_text = campaign_path.read_text()
            campaign_markdown = (output_dir / "campaign.md").read_text()
            self.assertNotIn(str(output_dir), campaign_text)
            self.assertNotIn(str(SKILL), campaign_text)
            self.assertNotIn(str(output_dir), campaign_markdown)
            self.assertNotIn(str(SKILL), campaign_markdown)

            campaign = json.loads(campaign_text)
            first_wave, second_wave = campaign["waves"]
            first_output = f"{output_dir.name}/iterations/01"
            second_output = f"{output_dir.name}/iterations/02"
            first_command = first_wave["ingest"]["command_args"]
            second_command = second_wave["ingest"]["command_args"]
            scaffold = first_wave["next_wave_scaffold"]
            self.assertEqual(first_command[first_command.index("--output-dir") + 1], first_output)
            self.assertEqual(second_command[second_command.index("--output-dir") + 1], second_output)
            self.assertEqual(scaffold["output_dir"], second_output)
            scaffold_command = scaffold["command_args"]
            self.assertEqual(scaffold_command[scaffold_command.index("--output-dir") + 1], second_output)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_research_loop_campaign_manifest_records_iteration_waves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "campaign-iterative-run"
            run_script(
                "research_loop.py",
                "--run-id", "campaign-iterative-loop",
                "--objective", "Coordinate multiple external subagent waves",
                "--iterations", 2,
                "--agent-count", 2,
                "--min-sampled-targets", 1,
                "--require-explicit-sampling-receipts",
                "--require-worker-blog-contract",
                "--output-dir", output_dir,
            )
            campaign = json.loads((output_dir / "campaign.json").read_text())
            loop = json.loads((output_dir / "loop.json").read_text())
            first = json.loads((output_dir / "iterations" / "01" / "synthesis.json").read_text())
            second = json.loads((output_dir / "iterations" / "02" / "synthesis.json").read_text())
            self.assertEqual(loop["campaign_manifest"]["path"], "campaign.json")
            self.assertEqual(loop["campaign_manifest"]["wave_count"], 2)
            self.assertEqual(campaign["wave_count"], 2)
            self.assertEqual(campaign["iteration_cap"], 2)
            first_wave, second_wave = campaign["waves"]
            self.assertEqual(first_wave["run_id"], "campaign-iterative-loop-i01")
            self.assertEqual(second_wave["run_id"], "campaign-iterative-loop-i02")
            self.assertEqual(first_wave["output_dir"], "iterations/01")
            self.assertEqual(second_wave["output_dir"], "iterations/02")
            self.assertEqual(first_wave["subagent_manifest_path"], "iterations/01/subagents.json")
            self.assertEqual(second_wave["subagent_manifest_path"], "iterations/02/subagents.json")
            self.assertTrue(first_wave["wave_dependency"]["requires_ingest_before_next_wave"])
            self.assertFalse(first_wave["wave_dependency"]["requires_prior_wave_ingestion"])
            self.assertTrue(second_wave["wave_dependency"]["requires_prior_wave_ingestion"])
            self.assertEqual(first_wave["assignment_mode"], first["assignment_planner"]["mode"])
            self.assertEqual(second_wave["assignment_mode"], second["assignment_planner"]["mode"])
            self.assertEqual(second_wave["assignment_mode"], "dynamic")
            self.assertEqual(second_wave["gap_hints"], first["next_gap_hints"])
            scaffold = first["next_wave_scaffold"]
            self.assertTrue(first["next_wave_expected"])
            self.assertFalse(second["next_wave_expected"])
            self.assertNotIn("next_wave_scaffold", second)
            self.assertEqual(first_wave["next_wave_scaffold"], scaffold)
            self.assertEqual(scaffold["next_iteration"], 2)
            self.assertEqual(scaffold["iteration_count"], 2)
            self.assertEqual(scaffold["assignment_mode"], "dynamic")
            self.assertEqual(scaffold["gap_hints"], first["next_gap_hints"])
            scaffold_command = scaffold["command_args"]
            self.assertEqual(scaffold_command[scaffold_command.index("--run-id") + 1], "campaign-iterative-loop-i02")
            self.assertEqual(scaffold_command[scaffold_command.index("--iteration-index") + 1], "2")
            self.assertEqual(scaffold_command[scaffold_command.index("--iteration-count-total") + 1], "2")
            self.assertIn("--require-explicit-sampling-receipts", scaffold_command)
            self.assertIn("--require-worker-blog-contract", scaffold_command)
            self.assertEqual(scaffold_command[scaffold_command.index("--min-sampled-targets") + 1], "1")
            self.assertEqual(scaffold_command[scaffold_command.index("--output-dir") + 1], str(output_dir / "iterations" / "02"))
            second_command = second_wave["ingest"]["command_args"]
            for hint in second_wave["gap_hints"]:
                self.assertIn(hint, second_command)
                self.assertIn(hint, scaffold_command)
            self.assertEqual(second_command[second_command.index("--output-dir") + 1], str(output_dir / "iterations" / "02"))
            self.assertTrue(first_wave["agents"][0]["result_path"].startswith("iterations/01/agents/"))
            self.assertTrue(second_wave["agents"][0]["result_path"].startswith("iterations/02/agents/"))
            campaign_markdown = (output_dir / "campaign.md").read_text()
            self.assertIn("next-wave scaffold command args", campaign_markdown)

    def test_research_campaign_runner_runs_agents_and_ingests_wave(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "runner-campaign"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            run_script(
                "research_loop.py",
                "--run-id", "runner-campaign-loop",
                "--objective", "Run campaign agents from campaign receipt",
                "--agent-count", 2,
                "--output-dir", output_dir,
            )
            run_script(
                "run_research_campaign.py",
                "--campaign", output_dir / "campaign.json",
                "--agent-command", f"{sys.executable} {fake_executor}",
                "--workers", 2,
                "--execution-timeout", 10,
            )
            receipt = json.loads((output_dir / "campaign-run.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            self.assertEqual(receipt["state"], "completed")
            self.assertFalse(receipt["skip_ingest"])
            self.assertEqual(receipt["failure_count"], 0)
            self.assertEqual(len(receipt["waves"]), 1)
            wave = receipt["waves"][0]
            self.assertEqual(wave["state"], "wave_completed")
            self.assertEqual(wave["ingest"]["state"], "ingest_completed")
            self.assertEqual(wave["agent_count"], 2)
            self.assertTrue(all(agent["state"] == "agent_completed" for agent in wave["agents"]))
            self.assertTrue(all(agent["result_exists"] for agent in wave["agents"]))
            self.assertTrue((output_dir / wave["agents"][0]["stdout_path"]).exists())
            self.assertTrue((output_dir / wave["ingest"]["stdout_path"]).exists())
            self.assertEqual(synthesis["finding_count"], 2)
            self.assertTrue(synthesis["ingested_subagent_results"])
            self.assertEqual(synthesis["execution_counts"]["subagent_result_ingested"], 2)
            self.assertIn("Research Campaign Run runner-campaign-loop", (output_dir / "campaign-run.md").read_text())

    def test_research_campaign_runner_follows_next_wave_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign_parent = Path(tmp) / "adaptive-campaign"
            first_wave_dir = campaign_parent / "01"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            run_script(
                "research_loop.py",
                "--run-id", "adaptive-runner-loop-i01",
                "--objective", "Run adaptive local campaign waves from observed gaps",
                "--agent-count", 2,
                "--iteration-index", 1,
                "--iteration-count-total", 2,
                "--min-sampled-targets", 1,
                "--output-dir", first_wave_dir,
            )
            initial_campaign = json.loads((first_wave_dir / "campaign.json").read_text())
            initial_wave = initial_campaign["waves"][0]
            self.assertIn("next_wave_scaffold", initial_wave)
            ingest_command = initial_wave["ingest"]["command_args"]
            self.assertEqual(ingest_command[ingest_command.index("--iteration-index") + 1], "1")
            self.assertEqual(ingest_command[ingest_command.index("--iteration-count-total") + 1], "2")

            run_script(
                "run_research_campaign.py",
                "--campaign", first_wave_dir / "campaign.json",
                "--agent-command", f"{sys.executable} {fake_executor}",
                "--workers", 2,
                "--execution-timeout", 10,
                "--follow-next-wave-scaffold",
                "--max-followed-waves", 2,
            )
            receipt = json.loads((first_wave_dir / "campaign-run.json").read_text())
            self.assertEqual(receipt["state"], "completed")
            self.assertTrue(receipt["follow_next_wave_scaffold"])
            self.assertEqual(receipt["followed_scaffold_count"], 1)
            self.assertEqual(receipt["failure_count"], 0)
            self.assertEqual(len(receipt["waves"]), 2)
            first_wave, second_wave = receipt["waves"]
            self.assertEqual(first_wave["campaign_path"], "01/campaign.json")
            self.assertEqual(first_wave["next_wave_scaffold"]["state"], "scaffold_completed")
            self.assertEqual(first_wave["next_wave_scaffold"]["generated_campaign_path"], "02/campaign.json")
            self.assertEqual(first_wave["next_wave_scaffold"]["source"], "refreshed_campaign")
            self.assertEqual(second_wave["source"], "next_wave_scaffold")
            self.assertEqual(second_wave["campaign_path"], "02/campaign.json")
            self.assertEqual(second_wave["campaign_root"], "02")
            self.assertEqual(second_wave["state"], "wave_completed")
            self.assertEqual(second_wave["ingest"]["state"], "ingest_completed")
            second_root = campaign_parent / second_wave["campaign_root"]
            self.assertTrue((second_root / second_wave["agents"][0]["result_path"]).exists())

            first_synthesis = json.loads((first_wave_dir / "synthesis.json").read_text())
            second_synthesis = json.loads((campaign_parent / "02" / "synthesis.json").read_text())
            self.assertEqual(first_synthesis["iteration_count"], 2)
            self.assertIn("next_wave_scaffold", first_synthesis)
            self.assertTrue(first_synthesis["ingested_subagent_results"])
            self.assertEqual(second_synthesis["iteration"], 2)
            self.assertEqual(second_synthesis["iteration_count"], 2)
            self.assertTrue(second_synthesis["ingested_subagent_results"])
            self.assertEqual(second_synthesis["assignment_planner"]["mode"], "dynamic")
            self.assertNotIn("next_wave_scaffold", second_synthesis)
            campaign_run_markdown = (first_wave_dir / "campaign-run.md").read_text()
            self.assertIn("Follow next-wave scaffolds: True", campaign_run_markdown)
            self.assertIn("Scaffold Followups", campaign_run_markdown)
            self.assertIn("scaffold_completed", campaign_run_markdown)

    def test_research_campaign_runner_fails_loud_and_preserves_worker_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "runner-fail-campaign"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            run_script(
                "research_loop.py",
                "--run-id", "runner-fail-loop",
                "--objective", "Preserve campaign runner failure receipts",
                "--agent-count", 1,
                "--output-dir", output_dir,
            )
            os.environ["MLX_FAKE_EXECUTOR_FAIL"] = "1"
            try:
                result = run_script(
                    "run_research_campaign.py",
                    "--campaign", output_dir / "campaign.json",
                    "--agent-command", f"{sys.executable} {fake_executor}",
                    "--execution-timeout", 10,
                    expected=2,
                )
            finally:
                os.environ.pop("MLX_FAKE_EXECUTOR_FAIL", None)
            self.assertIn("campaign run failed", result.stderr)
            receipt = json.loads((output_dir / "campaign-run.json").read_text())
            wave = receipt["waves"][0]
            agent = wave["agents"][0]
            self.assertEqual(receipt["state"], "failed")
            self.assertEqual(receipt["failure_count"], 1)
            self.assertEqual(wave["state"], "wave_failed")
            self.assertEqual(wave["ingest"]["state"], "not_started")
            self.assertEqual(agent["state"], "agent_failed")
            self.assertEqual(agent["exit_code"], 3)
            self.assertIn("agent command exited with 3", agent["failure_reason"])
            stderr_text = (output_dir / agent["stderr_path"]).read_text()
            self.assertIn("forced fake executor failure", stderr_text)
            self.assertFalse((output_dir / agent["result_path"]).exists())
            self.assertIn("agent command exited with 3", (output_dir / "campaign-run.md").read_text())

    def test_research_loop_iterations_feed_next_gap_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "iterative-run"
            run_script(
                "research_loop.py",
                "--run-id", "iterative-loop",
                "--objective", "Broaden MLX porting evidence beyond GitHub",
                "--iterations", 2,
                "--min-sampled-targets", 2,
                "--min-non-github-lanes", 3,
                "--require-source-lane", "hugging_face",
                "--offline-fixture", FIXTURES / "research_loop" / "offline_findings.json",
                "--output-dir", output_dir,
            )
            loop = json.loads((output_dir / "loop.json").read_text())
            first = json.loads((output_dir / "iterations" / "01" / "synthesis.json").read_text())
            second = json.loads((output_dir / "iterations" / "02" / "synthesis.json").read_text())
            self.assertEqual(loop["iteration_count"], 2)
            self.assertEqual(loop["total_finding_count"], 6)
            self.assertEqual(loop["sampling_coverage"]["sampled_target_count"], 4)
            self.assertEqual(loop["sampling_coverage"]["unplanned_source_count"], 2)
            self.assertEqual(loop["promotion_review"]["promotion_ready_count"], 2)
            self.assertEqual(loop["promotion_review"]["validation_backlog_count"], 4)
            self.assertEqual(loop["promotion_review"]["rejected_count"], 0)
            dossier = loop["learning_dossier"]
            self.assertTrue(dossier["review_only"])
            self.assertFalse(dossier["auto_modify_recommendations"])
            self.assertFalse(dossier["auto_promote_sources"])
            self.assertEqual(dossier["learned_findings"]["total_count"], 6)
            self.assertEqual(dossier["learned_findings"]["promotion_ready_count"], 2)
            self.assertEqual(dossier["learned_findings"]["validation_backlog_count"], 4)
            self.assertEqual(dossier["coverage_overview"]["sampled_target_count"], 4)
            self.assertEqual(dossier["coverage_overview"]["source_citation_count"], 6)
            self.assertEqual(dossier["coverage_overview"]["unique_source_observation_count"], 6)
            self.assertEqual(dossier["coverage_overview"]["source_lane_citation_counts"]["official_docs"], 2)
            self.assertIn("hugging_face", dossier["coverage_overview"]["non_github_lanes_covered"])
            self.assertTrue(dossier["evidence_gaps"]["thin_source_lanes"])
            self.assertTrue(dossier["evidence_gaps"]["uncited_source_lanes"])
            self.assertEqual(dossier["blog_health"]["failed_count"], 0)
            self.assertTrue(dossier["validation_backlog"])
            self.assertTrue(any(action["kind"] == "gap_hint" for action in dossier["next_research_actions"]))
            self.assertTrue(any(action["kind"] == "thin_source_lane" for action in dossier["next_research_actions"]))
            self.assertEqual(loop["iterations"][0]["promotion_review"]["promotion_ready_count"], 1)
            self.assertEqual(loop["iterations"][1]["promotion_review"]["validation_backlog_count"], 2)
            ready_entry = loop["promotion_review"]["promotion_ready"][0]
            self.assertEqual(ready_entry["iteration"], 1)
            self.assertEqual(ready_entry["iteration_run_id"], "iterative-loop-i01")
            self.assertEqual(ready_entry["iteration_output_dir"], "iterations/01")
            self.assertEqual(loop["review_gate"]["status"], "pass")
            self.assertEqual(first["review_gate"]["status"], "pass")
            self.assertEqual(first["assignment_planner"]["mode"], "config-order")
            self.assertEqual(second["assignment_planner"]["mode"], "dynamic")
            self.assertIn("benchmark", first["next_gap_hints"])
            self.assertIn("hugging_face", first["next_gap_hints"])
            self.assertEqual(second["gap_hints"], first["next_gap_hints"])
            self.assertEqual(loop["iterations"][0]["output_dir"], "iterations/01")
            self.assertIn("loop.md", {path.name for path in output_dir.iterdir()})
            loop_markdown = (output_dir / "loop.md").read_text()
            self.assertIn("Final gap hints", loop_markdown)
            self.assertIn("sampling coverage", loop_markdown)
            self.assertIn("Review gate: pass", loop_markdown)
            self.assertIn("## Promotion Review", loop_markdown)
            self.assertIn("promotion ready: 2", loop_markdown)
            self.assertIn("i1: official-custom-metal-validation", loop_markdown)
            self.assertIn("iterative-loop-i02", loop_markdown)
            self.assertIn("## Learning Dossier", loop_markdown)
            self.assertIn("learned findings: 6", loop_markdown)
            self.assertIn("source citations: 6", loop_markdown)
            self.assertIn("### Next Research Actions", loop_markdown)
            self.assertIn("thin_source_lane", loop_markdown)

    def test_research_loop_until_review_gate_stops_after_passing_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "adaptive-pass-run"
            result = run_script(
                "research_loop.py",
                "--run-id", "adaptive-pass-loop",
                "--objective", "Stop once enough non-GitHub evidence is sampled",
                "--iterations", 3,
                "--until-review-gate",
                "--min-sampled-targets", 2,
                "--min-non-github-lanes", 3,
                "--require-source-lane", "hugging_face",
                "--offline-fixture", FIXTURES / "research_loop" / "offline_findings.json",
                "--output-dir", output_dir,
            )
            loop = json.loads((output_dir / "loop.json").read_text())
            self.assertIn("across 1 iterations", result.stdout)
            self.assertEqual(loop["iteration_count"], 1)
            self.assertEqual(loop["iteration_cap"], 3)
            self.assertTrue(loop["until_review_gate"])
            self.assertEqual(loop["stop_reason"], "review_gate_passed")
            self.assertTrue(loop["stopped_after_review_gate"])
            self.assertFalse(loop["iteration_cap_exhausted"])
            self.assertEqual(loop["review_gate"]["status"], "pass")
            self.assertTrue((output_dir / "iterations" / "01" / "synthesis.json").exists())
            self.assertFalse((output_dir / "iterations" / "02").exists())
            loop_markdown = (output_dir / "loop.md").read_text()
            self.assertIn("Stop reason: review_gate_passed", loop_markdown)

    def test_research_loop_until_review_gate_uses_coverage_hints_until_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "adaptive-cap-run"
            run_script(
                "research_loop.py",
                "--run-id", "adaptive-cap-loop",
                "--objective", "Keep sampling until enough evidence is returned",
                "--iterations", 2,
                "--until-review-gate",
                "--agent-count", 2,
                "--min-sampled-targets", 1,
                "--output-dir", output_dir,
            )
            loop = json.loads((output_dir / "loop.json").read_text())
            first = json.loads((output_dir / "iterations" / "01" / "synthesis.json").read_text())
            second = json.loads((output_dir / "iterations" / "02" / "synthesis.json").read_text())
            self.assertEqual(loop["iteration_count"], 2)
            self.assertEqual(loop["iteration_cap"], 2)
            self.assertEqual(loop["stop_reason"], "iteration_cap_exhausted")
            self.assertFalse(loop["stopped_after_review_gate"])
            self.assertTrue(loop["iteration_cap_exhausted"])
            self.assertEqual(loop["review_gate"]["status"], "fail")
            self.assertIn("sampled_planned_targets observed 0, required 1", loop["review_gate"]["blocked_reasons"][0])
            self.assertTrue(first["next_gap_hints"])
            self.assertTrue({"official_docs", "papers"} & set(first["next_gap_hints"]))
            self.assertEqual(second["gap_hints"], first["next_gap_hints"])
            self.assertEqual(second["assignment_planner"]["mode"], "dynamic")
            loop_markdown = (output_dir / "loop.md").read_text()
            self.assertIn("Stop reason: iteration_cap_exhausted", loop_markdown)

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
            self.assertEqual(synthesis["assignment_planner"]["mode"], "config-order")
            self.assertEqual(synthesis["sampling_coverage"]["sampled_target_count"], 2)
            self.assertEqual(synthesis["sampling_coverage"]["unplanned_source_count"], 1)
            self.assertEqual(synthesis["review_gate"]["status"], "pass")
            self.assertEqual(synthesis["blog_contract"]["failed_count"], 0)
            self.assertTrue(all(receipt["contract_status"] == "pass" for receipt in synthesis["blog_receipts"]))
            self.assertIn("official_docs", synthesis["non_github_lanes_covered"])
            self.assertIn("hugging_face", synthesis["non_github_lanes_covered"])
            self.assertIn("technical_blogs", synthesis["non_github_lanes_covered"])
            self.assertGreater(synthesis["planned_non_github_sample_target_count"], 0)
            self.assertIn("planned_source_lane_counts", synthesis)
            self.assertIn("planned_sample_target_counts", synthesis)
            self.assertIn("tests/fixtures/research_loop/offline_findings.json", synthesis["offline_fixture"])
            self.assertEqual(len(assignments["assignments"]), 6)
            self.assertEqual(assignments["assignments"][0]["execution"]["state"], "fixture_ingested")
            self.assertEqual(assignments["assignments"][-1]["execution"]["state"], "scaffolded_not_run")
            self.assertIn("sampling_coverage", assignments["assignments"][0])
            self.assertEqual(assignments["assignments"][0]["sampling_coverage"]["unplanned_source_count"], 1)
            self.assertEqual(assignments["assignments"][2]["sampling_coverage"]["sampled_target_count"], 1)
            prompt = assignments["assignments"][0]["prompt"]
            self.assertIn("Do not execute remote model code", prompt)
            blog = output_dir / "blogs" / "official-docs-cartographer.md"
            self.assertTrue(blog.exists())
            blog_text = blog.read_text()
            self.assertIn("Planned sampling", blog_text)
            self.assertIn("Sampling coverage", blog_text)
            self.assertIn("MLX custom extensions documentation", blog_text)
            self.assertIn("Candidate findings", blog_text)
            self.assertIn("official-custom-metal-validation", blog_text)

    def test_research_loop_builds_evidence_matrix(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            shared_source = {
                "title": "MLX documentation index",
                "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                "accessed": "2026-06-27",
                "kind": "official-doc",
                "sampled_target_title": "MLX documentation index",
                "sampled_target_locator": "https://ml-explore.github.io/mlx/build/html/index.html",
            }
            fixture = tmp_path / "evidence_matrix.json"
            fixture.write_text(json.dumps({
                "agents": [
                    {
                        "persona_id": "official-docs-cartographer",
                        "findings": [
                            {
                                "id": "shared-official",
                                "title": "Shared official source",
                                "summary": "The shared source is cited by one official-docs finding.",
                                "source_lane": "official_docs",
                                "sources": [
                                    shared_source,
                                    {
                                        "title": "MLX custom Metal kernels",
                                        "url": "https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html",
                                        "accessed": "2026-06-27",
                                        "kind": "official-doc",
                                        "sampled_target_title": "MLX custom Metal kernels",
                                        "sampled_target_locator": "https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html",
                                    },
                                ],
                                "decision": "adopted",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Evidence matrix rows must preserve source links.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["Repeated citation is not promotion evidence."],
                                "required_next_validation": "Review the evidence matrix before editing guidance.",
                            }
                        ],
                    },
                    {
                        "persona_id": "paper-architecture-scout",
                        "findings": [
                            {
                                "id": "shared-paper",
                                "title": "Shared source from paper lane",
                                "summary": "The same source URL can be cited by another researcher and lane.",
                                "source_lane": "papers",
                                "sources": [shared_source],
                                "decision": "held",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Evidence matrix rows must deduplicate by normalized source locator.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["This proves matrix behavior only."],
                                "required_next_validation": "Inspect source-lane coverage before promotion.",
                            }
                        ],
                    },
                ],
            }), encoding="utf-8")
            output_dir = tmp_path / "evidence-matrix-run"
            run_script(
                "research_loop.py",
                "--run-id", "evidence-matrix-loop",
                "--agent-count", 2,
                "--offline-fixture", fixture,
                "--output-dir", output_dir,
            )
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            matrix = synthesis["evidence_matrix"]
            self.assertTrue(matrix["review_only"])
            self.assertEqual(matrix["unique_source_count"], 2)
            self.assertEqual(matrix["source_citation_count"], 3)
            self.assertEqual(matrix["source_lane_counts"]["official_docs"], 2)
            self.assertEqual(matrix["source_lane_counts"]["papers"], 1)
            self.assertEqual(matrix["finding_decision_counts"]["adopted"], 1)
            self.assertEqual(matrix["finding_decision_counts"]["held"], 1)
            shared = next(
                source for source in matrix["sources"]
                if source["key"] == "https://ml-explore.github.io/mlx/build/html/index.html"
            )
            self.assertEqual(shared["citation_count"], 2)
            self.assertEqual(shared["finding_count"], 2)
            self.assertEqual(set(shared["finding_ids"]), {"shared-official", "shared-paper"})
            self.assertEqual(set(shared["persona_ids"]), {"official-docs-cartographer", "paper-architecture-scout"})
            self.assertEqual(set(shared["source_lanes"]), {"official_docs", "papers"})
            self.assertEqual(shared["decision_counts"], {"adopted": 1, "held": 1})
            self.assertEqual(matrix["top_sources"][0]["key"], shared["key"])
            self.assertTrue(any(lane["source_lane"] == "packages" for lane in matrix["uncited_source_lanes"]))
            summary = (output_dir / "synthesis.md").read_text()
            self.assertIn("## Evidence Matrix", summary)
            self.assertIn("unique sources: 2", summary)
            self.assertIn("MLX documentation index - 2 citation(s)", summary)
            self.assertIn("packages: 0/", summary)
            self.assertIn("### Thin Source Lanes", summary)
            self.assertIn("### Uncited Source Lanes", summary)

    def test_research_loop_builds_promotion_review_ledger(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            fixture = tmp_path / "promotion_ledger.json"
            source = {
                "title": "MLX documentation index",
                "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                "accessed": "2026-06-27",
                "kind": "official-doc",
            }
            fixture.write_text(json.dumps({
                "agents": [
                    {
                        "persona_id": "official-docs-cartographer",
                        "findings": [
                            {
                                "id": "adopted-complete",
                                "title": "Complete adopted finding",
                                "summary": "This finding has enough metadata to enter promotion review.",
                                "source_lane": "official_docs",
                                "sources": [source],
                                "decision": "adopted",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Run a local parity fixture before editing the skill.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["The test fixture proves ledger routing only."],
                                "required_next_validation": "Run the targeted parity fixture and update manifests.",
                            },
                            {
                                "id": "adopted-thin",
                                "title": "Thin adopted finding",
                                "summary": "This adopted finding lacks promotion metadata and must be held.",
                                "source_lane": "official_docs",
                                "sources": [source],
                                "decision": "adopted",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Run a local parity fixture before editing the skill.",
                                "affects": ["references/deep-research-loop.md"],
                            },
                            {
                                "id": "needs-validation-backlog",
                                "title": "Needs validation finding",
                                "summary": "This finding belongs in the validation backlog.",
                                "source_lane": "official_docs",
                                "sources": [source],
                                "decision": "needs-validation",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Add a reproducible MLX validation path.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["No local path yet."],
                                "required_next_validation": "Build the missing fixture.",
                            },
                            {
                                "id": "rejected-finding",
                                "title": "Rejected finding",
                                "summary": "This finding must remain out of promotion review.",
                                "source_lane": "official_docs",
                                "sources": [source],
                                "decision": "rejected",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "No validation; rejected for this campaign.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["Rejected fixture row."],
                                "required_next_validation": "None.",
                            },
                        ],
                    }
                ],
            }), encoding="utf-8")
            output_dir = tmp_path / "promotion-ledger-run"
            run_script(
                "research_loop.py",
                "--run-id", "promotion-ledger-loop",
                "--agent-count", 1,
                "--offline-fixture", fixture,
                "--output-dir", output_dir,
            )
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            ledger = synthesis["promotion_review"]
            self.assertEqual(ledger["promotion_ready_count"], 1)
            self.assertEqual(ledger["promotion_ready"][0]["id"], "adopted-complete")
            self.assertEqual(ledger["validation_backlog_count"], 2)
            backlog = {item["id"]: item for item in ledger["validation_backlog"]}
            self.assertIn("adopted-thin", backlog)
            self.assertIn("missing required next validation", backlog["adopted-thin"]["promotion_blockers"])
            self.assertIn("missing rollback or caveat metadata", backlog["adopted-thin"]["promotion_blockers"])
            self.assertIn("needs-validation-backlog", backlog)
            self.assertEqual(backlog["needs-validation-backlog"]["promotion_blockers"], ["decision is needs-validation"])
            self.assertEqual(ledger["rejected_count"], 1)
            self.assertEqual(ledger["rejected"][0]["id"], "rejected-finding")
            summary = (output_dir / "synthesis.md").read_text()
            self.assertIn("## Promotion Review", summary)
            self.assertIn("adopted-complete", summary)
            self.assertIn("adopted-thin (adopted)", summary)
            self.assertIn("rejected-finding", summary)

    def test_research_loop_review_gate_fails_loud_after_writing_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate-fail-run"
            result = run_script(
                "research_loop.py",
                "--run-id", "gate-fail-loop",
                "--objective", "Require sampled evidence before skill updates",
                "--agent-count", 2,
                "--min-sampled-targets", 1,
                "--fail-on-review-gate",
                "--output-dir", output_dir,
                expected=2,
            )
            self.assertIn("review gate failed", result.stderr)
            self.assertTrue((output_dir / "assignments.json").exists())
            self.assertTrue((output_dir / "synthesis.md").exists())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            self.assertEqual(synthesis["finding_count"], 0)
            self.assertEqual(synthesis["review_gate"]["status"], "fail")
            self.assertFalse(synthesis["review_gate"]["ready_for_skill_update"])
            self.assertIn("sampled_planned_targets observed 0, required 1", synthesis["review_gate"]["blocked_reasons"])
            summary = (output_dir / "synthesis.md").read_text()
            self.assertIn("Review gate: fail", summary)

    def test_research_loop_review_gate_passes_fixture_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate-pass-run"
            run_script(
                "research_loop.py",
                "--run-id", "gate-pass-loop",
                "--objective", "Require multi-source evidence before skill updates",
                "--min-sampled-targets", 2,
                "--min-non-github-lanes", 3,
                "--require-source-lane", "hugging_face",
                "--offline-fixture", FIXTURES / "research_loop" / "offline_findings.json",
                "--output-dir", output_dir,
            )
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            gate = synthesis["review_gate"]
            self.assertEqual(gate["status"], "pass")
            self.assertTrue(gate["ready_for_skill_update"])
            self.assertEqual(gate["requirements"]["min_sampled_targets"], 2)
            self.assertEqual(gate["requirements"]["min_non_github_lanes"], 3)
            self.assertEqual(gate["requirements"]["required_source_lanes"], ["hugging_face"])
            self.assertTrue(all(check["status"] == "pass" for check in gate["checks"]))

    def test_research_loop_accepts_valid_explicit_sampling_receipts(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            fixture = tmp_path / "explicit_valid.json"
            fixture.write_text(json.dumps({
                "agents": [
                    {
                        "persona_id": "official-docs-cartographer",
                        "decision_notes": ["Explicit receipt points this source at the planned docs-index target."],
                        "findings": [
                            {
                                "id": "explicit-receipt-valid",
                                "title": "Explicit target receipt can prove sampled coverage",
                                "summary": "A returned source can identify which planned sample target it satisfied.",
                                "source_lane": "official_docs",
                                "sources": [
                                    {
                                        "title": "MLX quick start documentation",
                                        "url": "https://ml-explore.github.io/mlx/build/html/usage/quick_start.html",
                                        "accessed": "2026-06-27",
                                        "kind": "official-doc",
                                        "sampled_target_title": "MLX documentation index",
                                        "sampled_target_locator": "https://ml-explore.github.io/mlx/build/html/index.html",
                                    }
                                ],
                                "decision": "held",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Explicit target receipts must match the assignment sample plan.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["This proves receipt validation only."],
                                "required_next_validation": "Run a live campaign and inspect returned target receipts.",
                            }
                        ],
                    }
                ],
            }), encoding="utf-8")
            output_dir = tmp_path / "explicit-valid-run"
            run_script(
                "research_loop.py",
                "--run-id", "explicit-valid-loop",
                "--agent-count", 1,
                "--offline-fixture", fixture,
                "--min-sampled-targets", 1,
                "--require-explicit-sampling-receipts",
                "--fail-on-review-gate",
                "--output-dir", output_dir,
            )
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            coverage = synthesis["sampling_coverage"]
            self.assertEqual(synthesis["review_gate"]["status"], "pass")
            self.assertEqual(coverage["sampled_target_count"], 1)
            self.assertEqual(coverage["valid_explicit_sampling_receipt_count"], 1)
            self.assertEqual(coverage["invalid_explicit_sampling_receipt_count"], 0)
            self.assertEqual(coverage["sampled_target_missing_explicit_receipt_count"], 0)
            first_target = coverage["assignments"][0]["sampled_targets"][0]
            self.assertEqual(first_target["explicit_receipts"][0]["sampled_target_title"], "MLX documentation index")
            blog = (output_dir / "blogs" / "official-docs-cartographer.md").read_text()
            self.assertIn("Explicit sampling receipts: 1 valid, 0 invalid", blog)

    def test_research_loop_required_explicit_sampling_receipts_fails_when_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            fixture = tmp_path / "explicit_missing.json"
            fixture.write_text(json.dumps({
                "agents": [
                    {
                        "persona_id": "official-docs-cartographer",
                        "findings": [
                            {
                                "id": "explicit-receipt-missing",
                                "title": "Missing explicit target receipt should block strict runs",
                                "summary": "The source matches a planned target by URL but omits the explicit target receipt.",
                                "source_lane": "official_docs",
                                "sources": [
                                    {
                                        "title": "MLX documentation index",
                                        "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                                        "accessed": "2026-06-27",
                                        "kind": "official-doc",
                                    }
                                ],
                                "decision": "held",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Strict sampling runs require explicit target receipts.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["This proves the review gate only."],
                                "required_next_validation": "Add sampled_target_title or sampled_target_locator to returned sources.",
                            }
                        ],
                    }
                ],
            }), encoding="utf-8")
            output_dir = tmp_path / "explicit-missing-run"
            result = run_script(
                "research_loop.py",
                "--run-id", "explicit-missing-loop",
                "--agent-count", 1,
                "--offline-fixture", fixture,
                "--min-sampled-targets", 1,
                "--require-explicit-sampling-receipts",
                "--fail-on-review-gate",
                "--output-dir", output_dir,
                expected=2,
            )
            self.assertIn("explicit_sampling_receipts observed 0, required 1", result.stderr)
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            self.assertEqual(synthesis["review_gate"]["status"], "fail")
            self.assertEqual(synthesis["sampling_coverage"]["sampled_target_missing_explicit_receipt_count"], 1)

    def test_research_loop_required_explicit_sampling_receipts_fails_invalid_claims(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            fixture = tmp_path / "explicit_invalid.json"
            fixture.write_text(json.dumps({
                "agents": [
                    {
                        "persona_id": "official-docs-cartographer",
                        "findings": [
                            {
                                "id": "explicit-receipt-invalid",
                                "title": "Invalid explicit target receipt should block strict runs",
                                "summary": "The source claims a target outside the planned assignment sample plan.",
                                "source_lane": "official_docs",
                                "sources": [
                                    {
                                        "title": "MLX documentation index",
                                        "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                                        "accessed": "2026-06-27",
                                        "kind": "official-doc",
                                        "sampled_target_locator": "https://example.com/not-in-the-plan",
                                    }
                                ],
                                "decision": "held",
                                "evidence_level": "deterministic-test-fixture",
                                "validation_gate": "Explicit target receipts must stay inside the assignment sample plan.",
                                "affects": ["references/deep-research-loop.md"],
                                "caveats": ["This proves invalid receipt detection only."],
                                "required_next_validation": "Use the planned target title or locator from the assignment packet.",
                            }
                        ],
                    }
                ],
            }), encoding="utf-8")
            output_dir = tmp_path / "explicit-invalid-run"
            result = run_script(
                "research_loop.py",
                "--run-id", "explicit-invalid-loop",
                "--agent-count", 1,
                "--offline-fixture", fixture,
                "--min-sampled-targets", 1,
                "--require-explicit-sampling-receipts",
                "--fail-on-review-gate",
                "--output-dir", output_dir,
                expected=2,
            )
            self.assertIn("invalid_explicit_sampling_receipts observed 1, required <= 0", result.stderr)
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            coverage = synthesis["sampling_coverage"]
            self.assertEqual(coverage["invalid_explicit_sampling_receipt_count"], 1)
            invalid = coverage["assignments"][0]["invalid_sampling_receipts"][0]
            self.assertEqual(invalid["sampled_target_locator"], "https://example.com/not-in-the-plan")

    def test_research_loop_rejects_non_positive_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_script(
                "research_loop.py",
                "--iterations", 0,
                "--output-dir", Path(tmp) / "out",
                expected=2,
            )
            self.assertIn("--iterations must be positive", result.stderr)

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

    def test_research_loop_ingests_external_subagent_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "external-subagent-run"
            run_script(
                "research_loop.py",
                "--run-id", "external-subagent-loop",
                "--objective", "Collect externally spawned subagent results",
                "--agent-count", 1,
                "--output-dir", output_dir,
            )
            subagents = json.loads((output_dir / "subagents.json").read_text())
            agent = subagents["agents"][0]
            result_path = output_dir / agent["result_path"]
            result_path.write_text(json.dumps({
                "persona_id": agent["persona_id"],
                "decision_notes": ["External subagent result was written after dispatch."],
                "findings": [
                    {
                        "id": "external-subagent-official-docs",
                        "title": "External subagent official docs receipt",
                        "summary": "A separately spawned researcher can write a validated result JSON to the handoff path.",
                        "source_lane": "official_docs",
                        "sources": [
                            {
                                "title": "MLX documentation index",
                                "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                                "accessed": "2026-06-27",
                                "kind": "official-doc",
                            }
                        ],
                        "decision": "held",
                        "evidence_level": "deterministic-test-fixture",
                        "validation_gate": "External subagent receipts must pass schema validation before synthesis.",
                        "affects": ["references/deep-research-loop.md"],
                        "caveats": ["This proves ingestion, not a real recommendation."],
                        "required_next_validation": "Run a live external subagent campaign before promotion.",
                    }
                ],
            }), encoding="utf-8")
            run_script(
                "research_loop.py",
                "--run-id", "external-subagent-loop",
                "--objective", "Collect externally spawned subagent results",
                "--agent-count", 1,
                "--ingest-subagent-results",
                "--output-dir", output_dir,
            )
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            subagents = json.loads((output_dir / "subagents.json").read_text())
            assignment = assignments["assignments"][0]
            execution = assignment["execution"]
            self.assertEqual(synthesis["finding_count"], 1)
            self.assertTrue(synthesis["ingested_subagent_results"])
            self.assertEqual(synthesis["execution_counts"]["subagent_result_ingested"], 1)
            self.assertEqual(synthesis["evidence_matrix"]["unique_source_count"], 1)
            self.assertEqual(synthesis["evidence_matrix"]["source_citation_count"], 1)
            self.assertEqual(synthesis["subagent_dispatch"]["execution_mode"], "external-subagent")
            self.assertEqual(subagents["execution_mode"], "external-subagent")
            self.assertEqual(execution["kind"], "external-subagent")
            self.assertEqual(execution["state"], "subagent_result_ingested")
            self.assertEqual(execution["result_path"], agent["result_path"])
            packet = json.loads((output_dir / execution["assignment_path"]).read_text())
            self.assertEqual(packet["execution"]["state"], "subagent_result_ingested")
            self.assertEqual(packet["blog"]["source"], "generated")
            blog = output_dir / execution["blog_path"]
            self.assertIn("external-subagent-official-docs", blog.read_text())

    def test_research_loop_preserves_external_subagent_authored_blog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "external-blog-run"
            run_script(
                "research_loop.py",
                "--run-id", "external-blog-loop",
                "--objective", "Preserve external worker authored blogs",
                "--agent-count", 1,
                "--output-dir", output_dir,
            )
            subagents = json.loads((output_dir / "subagents.json").read_text())
            agent = subagents["agents"][0]
            result_path = output_dir / agent["result_path"]
            blog_path = output_dir / agent["blog_path"]
            result_path.write_text(json.dumps({
                "persona_id": agent["persona_id"],
                "decision_notes": ["External worker also wrote a blog."],
                "findings": [
                    {
                        "id": "external-blog-finding",
                        "title": "External worker blog receipt",
                        "summary": "A separately spawned researcher can write a blog to the handoff path.",
                        "source_lane": "official_docs",
                        "sources": [
                            {
                                "title": "MLX documentation index",
                                "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                                "accessed": "2026-06-27",
                                "kind": "official-doc",
                            }
                        ],
                        "decision": "held",
                        "evidence_level": "deterministic-test-fixture",
                        "validation_gate": "External blog receipts must be contract-checked.",
                        "affects": ["references/deep-research-loop.md"],
                        "caveats": ["This proves external blog preservation only."],
                        "required_next_validation": "Run a live external subagent campaign.",
                    }
                ],
            }), encoding="utf-8")
            blog_path.write_text("# Worker-authored external blog\n\n## Assignment\nExternal blog body.\n", encoding="utf-8")
            run_script(
                "research_loop.py",
                "--run-id", "external-blog-loop",
                "--objective", "Preserve external worker authored blogs",
                "--agent-count", 1,
                "--ingest-subagent-results",
                "--output-dir", output_dir,
            )
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            assignment = assignments["assignments"][0]
            self.assertEqual(assignment["blog"]["source"], "executor-authored")
            self.assertEqual(assignment["blog"]["contract_status"], "fail")
            self.assertIn("Sources sampled", assignment["blog"]["missing_sections"])
            self.assertIn("Worker-authored external blog", (output_dir / assignment["blog"]["path"]).read_text())
            self.assertTrue((output_dir / assignment["blog"]["generated_blog_path"]).exists())
            self.assertEqual(synthesis["blog_contract"]["worker_authored_failed_count"], 1)

    def test_research_loop_ingest_subagent_results_fails_on_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "external-missing-run"
            result = run_script(
                "research_loop.py",
                "--run-id", "external-missing-loop",
                "--objective", "Fail loudly when external subagent result is missing",
                "--agent-count", 1,
                "--ingest-subagent-results",
                "--output-dir", output_dir,
                expected=2,
            )
            self.assertIn("subagent result ingestion failed", result.stderr)
            self.assertIn("agents/official-docs-cartographer.result.json", result.stderr)

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
            subagents = json.loads((output_dir / "subagents.json").read_text())
            self.assertEqual(synthesis["finding_count"], 2)
            self.assertEqual(synthesis["execution_counts"]["executor_completed"], 2)
            self.assertEqual(synthesis["execution_counts"]["executor_failed"], 0)
            self.assertEqual(synthesis["execution_counts"]["scaffolded_not_run"], 0)
            self.assertIn("executor_command", synthesis)
            self.assertEqual(synthesis["evidence_matrix"]["unique_source_count"], 1)
            self.assertEqual(synthesis["evidence_matrix"]["source_citation_count"], 2)
            self.assertEqual(synthesis["executor_workers"], 1)
            self.assertEqual(synthesis["executor_actual_workers"], 1)
            self.assertEqual(synthesis["subagent_dispatch"]["execution_mode"], "local-executor")
            self.assertEqual(subagents["execution_mode"], "local-executor")
            self.assertEqual(subagents["executor_command"], f"{sys.executable} {fake_executor}")
            for assignment in assignments["assignments"]:
                execution = assignment["execution"]
                self.assertEqual(execution["kind"], "local-executor")
                self.assertEqual(execution["state"], "executor_completed")
                self.assertEqual(execution["exit_code"], 0)
                self.assertEqual(execution["executor_workers"], 1)
                self.assertEqual(execution["executor_actual_workers"], 1)
                self.assertIn("sample_plan", assignment)
                self.assertEqual(execution["assignment_path"], assignment["handoff"]["assignment_path"])
                self.assertEqual(execution["prompt_path"], assignment["handoff"]["prompt_path"])
                self.assertEqual(execution["result_path"], assignment["handoff"]["result_path"])
                self.assertEqual(execution["blog_path"], assignment["handoff"]["blog_path"])
                for key in ("assignment_path", "prompt_path", "stdout_path", "stderr_path", "result_path"):
                    self.assertTrue((output_dir / execution[key]).exists(), key)
                result = json.loads((output_dir / execution["result_path"]).read_text())
                self.assertEqual(result["persona_id"], assignment["persona_id"])
                packet = json.loads((output_dir / execution["assignment_path"]).read_text())
                self.assertEqual(packet["execution"]["state"], "executor_completed")
                self.assertEqual(packet["paths"]["result"], execution["result_path"])
                self.assertEqual(packet["environment"]["MLX_RESEARCH_ASSIGNMENT_PATH"], execution["assignment_path"])
                self.assertEqual(packet["blog"]["source"], "generated")
                self.assertEqual(assignment["blog"]["source"], "generated")
                self.assertEqual(execution["blog_source"], "generated")
                prompt = (output_dir / execution["prompt_path"]).read_text()
                self.assertIn("Sampling plan", prompt)
            blog = output_dir / "blogs" / "official-docs-cartographer.md"
            self.assertIn("fake-executor-official-docs-cartographer", blog.read_text())

    def test_research_loop_preserves_executor_authored_blog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "executor-blog-run"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            os.environ["MLX_FAKE_EXECUTOR_WRITE_BLOG"] = "1"
            try:
                run_script(
                    "research_loop.py",
                    "--run-id", "executor-blog-loop",
                    "--objective", "Preserve worker authored research blogs",
                    "--agent-count", 1,
                    "--executor-command", f"{sys.executable} {fake_executor}",
                    "--output-dir", output_dir,
                )
            finally:
                os.environ.pop("MLX_FAKE_EXECUTOR_WRITE_BLOG", None)
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            subagents = json.loads((output_dir / "subagents.json").read_text())
            assignment = assignments["assignments"][0]
            execution = assignment["execution"]
            blog_text = (output_dir / execution["blog_path"]).read_text()
            self.assertIn("Worker-authored research blog", blog_text)
            self.assertNotIn("fake-executor-official-docs-cartographer", blog_text)
            self.assertEqual(execution["blog_source"], "executor-authored")
            self.assertEqual(assignment["blog"]["source"], "executor-authored")
            self.assertTrue(assignment["blog"]["preserved_executor_blog"])
            generated_path = output_dir / assignment["blog"]["generated_blog_path"]
            self.assertTrue(generated_path.exists())
            self.assertIn("fake-executor-official-docs-cartographer", generated_path.read_text())
            packet = json.loads((output_dir / execution["assignment_path"]).read_text())
            self.assertEqual(packet["blog"]["source"], "executor-authored")
            self.assertEqual(subagents["agents"][0]["blog_source"], "executor-authored")
            self.assertEqual(synthesis["blog_receipts"][0]["source"], "executor-authored")
            self.assertEqual(synthesis["blog_receipts"][0]["contract_status"], "fail")
            self.assertIn("Planned sampling", synthesis["blog_receipts"][0]["missing_sections"])
            self.assertEqual(synthesis["blog_contract"]["worker_authored_failed_count"], 1)

    def test_research_loop_strict_worker_blog_contract_fails_after_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "strict-blog-run"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            os.environ["MLX_FAKE_EXECUTOR_WRITE_BLOG"] = "1"
            try:
                result = run_script(
                    "research_loop.py",
                    "--run-id", "strict-blog-loop",
                    "--objective", "Require worker blogs to include all research sections",
                    "--agent-count", 1,
                    "--executor-command", f"{sys.executable} {fake_executor}",
                    "--require-worker-blog-contract",
                    "--output-dir", output_dir,
                    expected=2,
                )
            finally:
                os.environ.pop("MLX_FAKE_EXECUTOR_WRITE_BLOG", None)
            self.assertIn("worker blog contract failed", result.stderr)
            self.assertTrue((output_dir / "assignments.json").exists())
            self.assertTrue((output_dir / "synthesis.json").exists())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            receipt = synthesis["blog_receipts"][0]
            self.assertEqual(receipt["source"], "executor-authored")
            self.assertEqual(receipt["contract_status"], "fail")
            self.assertIn("Sources sampled", receipt["missing_sections"])
            self.assertEqual(synthesis["blog_contract"]["worker_authored_failed_count"], 1)
            generated_path = output_dir / receipt["generated_blog_path"]
            self.assertTrue(generated_path.exists())
            summary = (output_dir / "synthesis.md").read_text()
            self.assertIn("worker-authored failed", summary)

    def test_research_loop_executor_workers_run_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "parallel-run"
            marker_dir = tmp_path / "markers"
            fake_executor = FIXTURES / "research_loop" / "fake_executor.py"
            os.environ["MLX_FAKE_EXECUTOR_SLEEP"] = "0.2"
            os.environ["MLX_FAKE_EXECUTOR_CONCURRENCY_DIR"] = str(marker_dir)
            try:
                run_script(
                    "research_loop.py",
                    "--run-id", "parallel-loop",
                    "--objective", "Prove parallel local executor receipt flow",
                    "--agent-count", 3,
                    "--executor-workers", 3,
                    "--executor-command", f"{sys.executable} {fake_executor}",
                    "--output-dir", output_dir,
                )
            finally:
                os.environ.pop("MLX_FAKE_EXECUTOR_SLEEP", None)
                os.environ.pop("MLX_FAKE_EXECUTOR_CONCURRENCY_DIR", None)
            assignments = json.loads((output_dir / "assignments.json").read_text())
            synthesis = json.loads((output_dir / "synthesis.json").read_text())
            self.assertEqual(synthesis["finding_count"], 3)
            self.assertEqual(synthesis["executor_workers"], 3)
            self.assertEqual(synthesis["executor_actual_workers"], 3)
            self.assertEqual([row["execution"]["assignment_index"] for row in assignments["assignments"]], [0, 1, 2])
            self.assertEqual([agent["persona_id"] for agent in synthesis["held"]], [
                "official-docs-cartographer",
                "paper-architecture-scout",
                "huggingface-ecosystem-sampler",
            ])
            active_counts = [
                json.loads(path.read_text())["active_count"]
                for path in marker_dir.glob("*.json")
            ]
            self.assertEqual(len(active_counts), 3)
            self.assertGreater(max(active_counts), 1)
            for assignment in assignments["assignments"]:
                execution = assignment["execution"]
                self.assertEqual(execution["executor_workers"], 3)
                self.assertEqual(execution["executor_actual_workers"], 3)
                stdout = (output_dir / execution["stdout_path"]).read_text()
                self.assertIn("active workers observed", stdout)

    def test_research_loop_rejects_non_positive_executor_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_script(
                "research_loop.py",
                "--executor-workers", 0,
                "--output-dir", Path(tmp) / "out",
                expected=2,
            )
            self.assertIn("--executor-workers must be positive", result.stderr)

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
