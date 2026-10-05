"""Offline survey contracts: source inspection never grants execution or promotion."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import inference_engine_advisor as advisor
import mlx_doctor as doctor
from _common import SkillError


class EngineAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.data = advisor.load_registry()

    def test_inventory_has_pinned_distinct_projects_and_explicit_depth(self):
        self.assertEqual(len(self.data["engines"]), 24)
        self.assertEqual(len(self.data["methods"]), 14)
        for engine in self.data["engines"]:
            self.assertEqual(engine["local_validation"], "not-run")
            self.assertRegex(engine["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(engine["limitations"])

    def test_layers_are_not_conflated(self):
        rows = {e["id"]: e for e in self.data["engines"]}
        self.assertNotEqual(rows["vllm-metal"]["repository"], rows["vllm-mlx"]["repository"])
        self.assertEqual(rows["mlx-vlm"]["category"], "reference-library")
        self.assertEqual(rows["splash"]["category"], "native-specialized")
        self.assertEqual(rows["basert"]["category"], "opaque-engine")
        self.assertEqual(rows["llama-cpp"]["category"], "non-mlx-comparator")
        self.assertIn("proprietary", rows["basert"]["license_note"])

    def test_every_method_has_gates_measurement_rollback_and_sources(self):
        for method in self.data["methods"]:
            for field in ("approach", "correctness", "measure", "rollback", "source_ids"):
                self.assertTrue(method[field], (method["id"], field))

    def test_all_workload_queries_are_research_only_and_deterministic(self):
        for workload in advisor.WORKLOADS:
            with self.subTest(workload=workload):
                result = advisor.shortlist(self.data, workload)
                self.assertEqual(result, advisor.shortlist(copy.deepcopy(self.data), workload))
                self.assertFalse(result["execution_allowed"])
                self.assertIn("not-performance-ranked", result["order"])
                self.assertTrue(result["candidates"])
                self.assertTrue(all(c["local_validation"] == "not-run" for c in result["candidates"]))

    def test_upstream_only_or_non_mlx_comparators_are_opt_in(self):
        normal = advisor.shortlist(self.data, "coding-agent")
        expanded = advisor.shortlist(self.data, "coding-agent", include_comparators=True)
        self.assertNotIn("basert", {e["id"] for e in normal["candidates"]})
        self.assertIn("basert", {e["id"] for e in expanded["candidates"]})
        self.assertFalse(expanded["execution_allowed"])

    def test_blockers_are_preserved_for_every_candidate(self):
        result = advisor.shortlist(self.data, "multimodal", blockers=["unsafe model code", "missing weights"])
        self.assertEqual(result["inspection_blockers"], ["unsafe model code", "missing weights"])
        self.assertTrue(all(c["eligibility"] == "blocked-by-inspection" for c in result["candidates"]))

    def test_unknown_workload_fails(self):
        with self.assertRaises(SkillError):
            advisor.shortlist(self.data, "fastest")

    def test_registry_rejects_unknown_fields_or_fabricated_local_evidence(self):
        for field, value in (("speedup", 100), ("local_validation", "passed")):
            data = copy.deepcopy(self.data)
            data["engines"][0][field] = value
            with self.subTest(field=field), self.assertRaises(SkillError):
                advisor.validate_registry(data)

    def test_registry_rejects_unpinned_or_mismatched_locators(self):
        for mutate in (
            lambda e: e.update(revision="main"),
            lambda e: e["evidence"][0].update(url="https://example.com/README.md"),
            lambda e: e["evidence"][0].update(sha256="0"),
            lambda e: e["evidence"][0].update(path="../outside.py"),
        ):
            data = copy.deepcopy(self.data); mutate(data["engines"][0])
            with self.assertRaises(SkillError):
                advisor.validate_registry(data)

    def test_registry_rejects_missing_readme(self):
        data = copy.deepcopy(self.data)
        data["engines"][0]["evidence"] = [e for e in data["engines"][0]["evidence"] if e["path"].lower() != "readme.md"]
        with self.assertRaises(SkillError):
            advisor.validate_registry(data)

    def test_registry_rejects_duplicate_engines_and_method_references(self):
        data = copy.deepcopy(self.data); data["engines"].append(data["engines"][0])
        with self.assertRaises(SkillError): advisor.validate_registry(data)
        data = copy.deepcopy(self.data); data["methods"][0]["source_ids"] = ["invented"]
        with self.assertRaises(SkillError): advisor.validate_registry(data)
        data = copy.deepcopy(self.data); data["engines"][0]["methods"] = ["invented"]
        with self.assertRaises(SkillError): advisor.validate_registry(data)

    def test_registry_rejects_malformed_method_scalars(self):
        for field in ("priority", "lossiness", "approach"):
            data = copy.deepcopy(self.data); data["methods"][0][field] = []
            with self.subTest(field=field), self.assertRaises(SkillError): advisor.validate_registry(data)

    def test_loader_rejects_duplicate_keys_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "registry.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(text)
                with self.assertRaises(SkillError): advisor.load_registry(path)

    def test_loader_rejects_symlink_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "registry.json"; path.write_text(" " * (advisor.MAX_BYTES + 1))
            with self.assertRaises(SkillError): advisor.load_registry(path)
            link = Path(raw) / "link.json"; link.symlink_to(advisor.REGISTRY)
            with self.assertRaises(SkillError): advisor.load_registry(link)

    def test_generated_markdown_and_site_are_current(self):
        self.assertEqual(advisor.REPORT.read_text(), advisor.render(self.data))
        self.assertEqual(advisor.SITE_REPORT.read_text(), advisor.render_html(self.data))
        self.assertIn("not a performance leaderboard", advisor.render_html(self.data))

    def test_html_escapes_registry_text(self):
        data = copy.deepcopy(self.data); data["engines"][0]["scope"] = '<script>alert("not executable")</script>'
        output = advisor.render_html(data)
        self.assertNotIn('<script>alert(', output)
        self.assertIn('&lt;script&gt;', output)

    def test_cli_validation_and_query_are_offline(self):
        for arguments in (("--validate",), ("--check",), ("--workload", "audio")):
            result = subprocess.run([sys.executable, str(SCRIPTS / "inference_engine_advisor.py"), *arguments], text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsInstance(json.loads(result.stdout), dict)

    def test_module_import_does_not_load_frameworks(self):
        code = "import sys; sys.path.insert(0," + repr(str(SCRIPTS)) + "); import inference_engine_advisor; assert not any(k.split('.')[0] in {'mlx','torch','transformers'} for k in sys.modules)"
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_doctor_shortlist_retains_model_blockers(self):
        args = doctor.parse_args([str(ROOT / "tests/fixtures/models/decoder"), "--workload", "coding-agent"])
        raw = {"schema_version": 1, "recommendation_blockers": ["injected canonical blocker"]}
        with patch.object(doctor, "inspect_input", return_value=raw):
            result = doctor.diagnose(args)
        self.assertFalse(result["ok"])
        self.assertIn("injected canonical blocker", result["serving_candidates"]["inspection_blockers"])
        self.assertTrue(all(x["eligibility"] == "blocked-by-inspection" for x in result["serving_candidates"]["candidates"]))

    def test_doctor_without_workload_does_not_choose_an_engine(self):
        report = doctor.diagnose(doctor.parse_args([]))
        self.assertIsNone(report["serving_candidates"])
        self.assertFalse(report["architecture"]["native_execution_verified"])

    def test_doctor_workload_choices_follow_registry_contract(self):
        for workload in advisor.WORKLOADS:
            self.assertEqual(doctor.parse_args(["--workload", workload]).workload, workload)


if __name__ == "__main__":
    unittest.main()
