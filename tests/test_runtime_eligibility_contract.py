"""Runtime availability gates are independent of research and performance claims."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import SkillError  # noqa: E402
from generate_claim_catalog import build_claim_catalog  # noqa: E402
from recommend_optimizations import (  # noqa: E402
    prepare_candidate,
    runtime_eligibility_holds,
    validate_guidance_runtime_requirements,
    validate_runtime_requirements,
)
from validate_benchmarks import build_assessment_report  # noqa: E402


METHOD_ID = "fixture-reviewed-runtime"
REQUIREMENTS = {
    "minimum_software": {"mlx": "0.32.4", "macos": "26.2"},
    "hardware": {"chip_family_any": ["m5"]},
}


def method() -> dict[str, object]:
    return {
        "id": METHOD_ID,
        "status": "native-mlx",
        "category": "runtime",
        "applies_to": ["dense-decoder-transformer"],
        "objectives": ["throughput"],
        "expected_effect": "Profile-required; verify the runtime and parity before benchmarking.",
        "validation_gates": ["runtime probe and parity on the target"],
        "runtime_requirements": copy.deepcopy(REQUIREMENTS),
    }


def profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "software": {"mlx": "0.32.4", "macos": "26.2"},
        "hardware": {"chip": "Apple M5 Pro"},
        "capabilities": [],
        "workloads": [],
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class RuntimeEligibilityContractTests(unittest.TestCase):
    def test_registry_contract_rejects_invalid_or_open_ended_requirements(self) -> None:
        malformed = [
            None, {}, [], True,
            {"minimum_versions": {"mlx": "0.32.4"}},
            {"minimum_software": {}},
            {"minimum_software": {"unknown_runtime": "1.0"}},
            {"minimum_software": {"mlx": True}},
            {"minimum_software": {"mlx": "0.32.4rc1"}},
            {"minimum_software": {"mlx": "0.32.4.dev1"}},
            {"minimum_software": {"mlx": "0.32.4+local"}},
            {"minimum_software": {"mlx": "0.32.4.1"}},
            {"minimum_software": {"mlx": "00.32.4"}},
            {"minimum_software": {"mlx": " 0.32.4"}},
            {"hardware": {}},
            {"hardware": {"chip_family_any": []}},
            {"hardware": {"chip_family_any": "m5"}},
            {"hardware": {"chip_family_any": ["m5", "m5"]}},
            {"hardware": {"chip_family_any": ["m5", {}]}},
            {"hardware": {"chip_family_any": ["M5"]}},
            {"hardware": {"chip_family_any": ["m05"]}},
            {"hardware": {"chip_family_any": ["m5+"], "future_chips": True}},
        ]
        for requirements in malformed:
            with self.subTest(requirements=requirements):
                candidate = method()
                candidate["runtime_requirements"] = requirements
                with self.assertRaisesRegex(SkillError, "runtime_requirements"):
                    validate_runtime_requirements(candidate)
        for requirements in (
            REQUIREMENTS,
            {"minimum_software": {"mlx_lm": "1", "mlx_audio": "0.4", "mlx_vlm": "0.3.1"}},
            {"hardware": {"chip_family_any": ["m5", "m6"]}},
        ):
            with self.subTest(valid=requirements):
                self.assertEqual(
                    validate_runtime_requirements({"id": METHOD_ID, "runtime_requirements": requirements}),
                    requirements,
                )

    def test_missing_requirements_preserve_legacy_eligibility(self) -> None:
        candidate = method()
        del candidate["runtime_requirements"]
        self.assertIsNone(validate_runtime_requirements(candidate))
        self.assertEqual(runtime_eligibility_holds(candidate, None), [])
        taxonomy = json.loads((SKILL / "assets/recommendation-taxonomy.yaml").read_text())
        prepared = prepare_candidate(candidate, None, [], None, taxonomy)
        self.assertTrue(prepared["execution_allowed"])
        self.assertEqual(prepared["eligibility_holds"], [])

    def test_target_release_comparison_is_numeric_and_unknown_versions_are_held(self) -> None:
        candidate = method()
        for release in ("0.32.4", "0.32.10", "0.100", "1", "1.0", "1.0.0"):
            target = profile()
            target["software"]["mlx"] = release
            with self.subTest(valid_release=release):
                self.assertEqual(runtime_eligibility_holds(candidate, target), [])
        for release in ("0.32.3", "0.32", "0.9", "0"):
            target = profile()
            target["software"]["mlx"] = release
            with self.subTest(old_release=release):
                holds = runtime_eligibility_holds(candidate, target)
                self.assertTrue(any("software.mlx >= 0.32.4" in item and "older" in item for item in holds))
        for release in (
            None, True, False, 0.33, 1, [], {}, "", "unknown", "latest", "v0.32.4",
            "0.32.4rc1", "0.32.4.dev1", "0.32.4+local", "0.32.4.0", "-1", "1e3",
            "0.32.04", " 0.32.4", "0.32.4\n", "\u0660.32.4", "1" * 100,
        ):
            target = profile()
            target["software"]["mlx"] = release
            with self.subTest(unknown_release=release):
                holds = runtime_eligibility_holds(candidate, target)
                self.assertTrue(any("software.mlx" in item and "missing or invalid" in item for item in holds))
        self.assertEqual(len(runtime_eligibility_holds(candidate, None)), 3)
        target = profile()
        target["software"]["macos"] = "26.1"
        self.assertTrue(any("software.macos >= 26.2" in item for item in runtime_eligibility_holds(candidate, target)))
        candidate["runtime_requirements"] = {"minimum_software": {"mlx": "1.0.0"}}
        target["software"]["mlx"] = "1"
        self.assertEqual(runtime_eligibility_holds(candidate, target), [])

    def test_chip_matching_requires_whole_reviewed_family(self) -> None:
        candidate = method()
        for chip in ("M5", "Apple M5", "Apple M5 Pro", "Apple M5 Max", "Apple M5 Ultra", " apple  m5  max "):
            target = profile()
            target["hardware"]["chip"] = chip
            with self.subTest(valid_chip=chip):
                self.assertEqual(runtime_eligibility_holds(candidate, target), [])
        for chip in (
            "Apple M4 Pro", "Apple M6", "M50", "Apple M5Pro", "M5 Pro Max",
            "Not Apple M5", "Apple M5-compatible", "Apple M5 Max (CPU)", "\uff2d5", None, True, [],
        ):
            target = profile()
            target["hardware"]["chip"] = chip
            with self.subTest(held_chip=chip):
                self.assertTrue(runtime_eligibility_holds(candidate, target))
        candidate["runtime_requirements"]["hardware"]["chip_family_any"].append("m6")
        target = profile()
        target["hardware"]["chip"] = "Apple M6 Pro"
        self.assertEqual(runtime_eligibility_holds(candidate, target), [])

    def test_runtime_eligibility_cannot_promote_a_numeric_claim(self) -> None:
        candidate = method()
        candidate["expected_effect"] = "A source reports 2.0x; target measurement is required."
        candidate["improvement_band"] = {
            "provenance": "source_reported", "range": "1.0x-2.0x",
            "metric": "throughput", "claim_status": "held", "evidence_lineage_ids": [],
        }
        catalog = build_claim_catalog(
            {"schema_version": 1, "methods": [candidate]},
            {"sources": []}, {"schema_version": 1, "assessments": []},
            assessments_available=False,
        )
        taxonomy = json.loads((SKILL / "assets/recommendation-taxonomy.yaml").read_text())
        for target, executable in ((profile(), True), (None, False)):
            with self.subTest(target=target):
                prepared = prepare_candidate(candidate, catalog["claims"][0], [], target, taxonomy)
                self.assertEqual(prepared["execution_allowed"], executable)
                self.assertIsNone(prepared["improvement_band"]["range"])
                self.assertTrue(prepared["claim_holds"])
                self.assertNotIn("2.0x", prepared["expected_effect"])

    def make_isolated_cli_fixture(self, tmp: Path) -> tuple[Path, Path, Path]:
        """Exercise real subprocesses with a tiny independent canonical registry."""
        skill = tmp / "skill"
        scripts = skill / "scripts"
        shutil.copytree(SCRIPTS, scripts, ignore=shutil.ignore_patterns("__pycache__"))
        assets = skill / "assets"
        benchmarks = assets / "benchmarks"
        benchmarks.mkdir(parents=True)
        for name in ("architectures.yaml", "recommendation-taxonomy.yaml"):
            shutil.copyfile(SKILL / "assets" / name, assets / name)
        candidate = method()
        baseline = copy.deepcopy(candidate)
        baseline["id"] = "fixture-baseline"
        del baseline["runtime_requirements"]
        guidance = {"schema_version": 1, "reviewed": "2026-11-05", "methods": [baseline, candidate]}
        sources = {"schema_version": 1, "sources": []}
        assessments = build_assessment_report(benchmarks)
        write_json(assets / "optimization_guidance.yaml", guidance)
        write_json(assets / "sources.yaml", sources)
        write_json(benchmarks / "receipt_assessments.json", assessments)
        write_json(assets / "effective_claims.json", build_claim_catalog(
            guidance, sources, assessments, assessments_available=True,
        ))
        write_json(assets / "knowledge_graph.json", {"schema_version": 1, "nodes": [], "edges": []})
        write_json(assets / "optimization_stacks.yaml", {"schema_version": 1, "stacks": [{
            "id": "fixture-runtime-stack", "label": "Runtime eligibility fixture", "status": "planning-only",
            "families": ["dense-decoder-transformer"], "primary_metric": "throughput",
            "steps": [
                {"method": "fixture-baseline", "lossiness": "lossless", "gate": "parity"},
                {"method": METHOD_ID, "lossiness": "lossless", "gate": "runtime probe and parity"},
            ],
            "composition_notes": [], "compound": {"measured_together": False, "receipts": []},
        }]})
        model = tmp / "model"
        shutil.copytree(ROOT / "tests/fixtures/models/decoder", model)
        inspection = tmp / "inspection.json"
        self.cli(scripts, "inspect_model.py", model, "--output", inspection)
        self.assertEqual(json.loads(inspection.read_text())["recommendation_blockers"], [])
        return scripts, model, inspection

    def cli(self, scripts: Path, name: str, *arguments: object, expected: int = 0) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, str(scripts / name), *(str(value) for value in arguments)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, expected, result.stderr or result.stdout)
        return result

    def test_cli_and_recomputed_plan_hold_runtime_incompatibility_even_with_allow_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            scripts, model, inspection = self.make_isolated_cli_fixture(tmp)
            old = profile()
            old["software"]["mlx"] = "0.32.3"
            for target, executable in ((None, False), (old, False), (profile(), True)):
                with self.subTest(target=target):
                    output, markdown, plan = tmp / "advice.json", tmp / "advice.md", tmp / "plan.md"
                    args = [inspection, "--output", output, "--markdown", markdown, "--allow-blocked"]
                    if target is not None:
                        target_path = tmp / "target.json"
                        write_json(target_path, target)
                        args += ["--target-profile", target_path]
                    self.cli(scripts, "recommend_optimizations.py", *args)
                    report = json.loads(output.read_text())
                    ids = lambda items: {item["id"] for item in items}
                    self.assertEqual(METHOD_ID in ids(report["ready_candidates"]), executable)
                    advisor_ids = ids([item for bucket in report["advisor_buckets"].values() for item in bucket])
                    self.assertEqual(METHOD_ID in advisor_ids, executable)
                    self.assertEqual(
                        METHOD_ID in {item["method"] for item in report["planning_stack"]["steps"]}, executable,
                    )
                    if not executable:
                        held = next(item for item in report["held_candidates"] if item["id"] == METHOD_ID)
                        self.assertFalse(held["execution_allowed"])
                        self.assertTrue(held["eligibility_holds"])
                        rendered = markdown.read_text()
                        self.assertIn("runtime compatibility evidence", rendered)
                        self.assertIn("software.mlx >= 0.32.4", rendered)
                        self.assertIn("missing or invalid" if target is None else "older", rendered)
                    self.cli(
                        scripts, "make_port_plan.py", inspection, "--artifact-root", model,
                        "--recommendations", output, "--output", plan,
                    )
                    plan_text = plan.read_text()
                    actionable = plan_text.split("### Held runtime-dependent methods", 1)[0]
                    self.assertEqual(METHOD_ID in actionable, executable)
                    if not executable:
                        self.assertIn("### Held runtime-dependent methods", plan_text)
                        self.assertIn("These methods are excluded from the actionable plan.", plan_text)
                        self.assertIn("software.mlx >= 0.32.4", plan_text)

            # The diagnostic override must also preserve runtime holds when
            # there is a real intake blocker, not only when the flag is idle.
            config_path = model / "config.json"
            config = json.loads(config_path.read_text())
            config["auto_map"] = {"AutoModel": "modeling_custom.CustomModel"}
            write_json(config_path, config)
            self.cli(scripts, "inspect_model.py", model, "--output", inspection)
            self.assertTrue(json.loads(inspection.read_text())["recommendation_blockers"])
            write_json(target_path, old)
            self.cli(
                scripts, "recommend_optimizations.py", inspection, "--family", "dense-decoder-transformer",
                "--target-profile", target_path, "--allow-blocked", "--output", output,
            )
            blocked = json.loads(output.read_text())
            self.assertTrue(blocked["blocked_advice_visible"])
            self.assertNotIn(METHOD_ID, {item["id"] for item in blocked["ready_candidates"]})
            self.assertTrue(all(
                METHOD_ID not in {item["id"] for item in items}
                for items in blocked["advisor_buckets"].values()
            ))
            held = next(item for item in blocked["held_candidates"] if item["id"] == METHOD_ID)
            self.assertFalse(held["execution_allowed"])
            self.assertNotIn(METHOD_ID, {item["method"] for item in blocked["held_planning_stack"]["steps"]})
            self.cli(
                scripts, "make_port_plan.py", inspection, "--family", "dense-decoder-transformer",
                "--artifact-root", model, "--recommendations", output, "--output", plan,
            )
            self.assertNotIn(METHOD_ID, plan.read_text())

    def test_invalid_registry_fails_before_objective_and_family_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            scripts, _, inspection = self.make_isolated_cli_fixture(tmp)
            guidance_path = scripts.parent / "assets/optimization_guidance.yaml"
            guidance = json.loads(guidance_path.read_text())
            invalid = guidance["methods"][1]
            invalid["applies_to"] = ["encoder-transformer"]
            invalid["runtime_requirements"]["minimum_software"]["mlx"] = "latest"
            write_json(guidance_path, guidance)
            with self.assertRaisesRegex(SkillError, "runtime_requirements"):
                validate_guidance_runtime_requirements(guidance)
            result = self.cli(
                scripts, "recommend_optimizations.py", inspection, "--objective", "ttft", expected=2,
            )
            self.assertIn("runtime_requirements.minimum_software.mlx", result.stderr)


if __name__ == "__main__":
    unittest.main()
