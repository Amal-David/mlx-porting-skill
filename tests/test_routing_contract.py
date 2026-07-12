"""Adversarial contract tests for architecture routing and hybrid plans."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_tooling import FIXTURES, SKILL, run_script

ROUTING_FIXTURES = FIXTURES / "routing"


def write_static_model(root: Path, fixture: dict[str, object]) -> None:
    """Write a tiny, non-executable model with the fixture's routing signals."""
    (root / "config.json").write_text(
        json.dumps(fixture["config"], indent=2) + "\n",
        encoding="utf-8",
    )
    tensor_keys = [str(key) for key in fixture["tensor_keys"]]
    header: dict[str, object] = {}
    offset = 0
    for key in tensor_keys:
        header[key] = {"dtype": "F16", "shape": [1], "data_offsets": [offset, offset + 2]}
        offset += 2
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + (b"\x00" * offset)
    )


def inspect_fixture(name: str, tmp: Path) -> dict[str, object]:
    fixture = json.loads((ROUTING_FIXTURES / name).read_text(encoding="utf-8"))
    model = tmp / fixture["name"]
    model.mkdir()
    write_static_model(model, fixture)
    output = tmp / f"{fixture['name']}.json"
    run_script("inspect_model.py", model, "--output", output)
    return json.loads(output.read_text(encoding="utf-8"))


class RoutingContractTests(unittest.TestCase):
    def test_weak_unknown_and_tied_routes_stop_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            for fixture_name in (
                "weak-shared-config.json",
                "unknown-architecture.json",
                "ambiguous-synthetic-route.json",
                "conflicting-hybrid-identity.json",
            ):
                with self.subTest(fixture=fixture_name):
                    report = inspect_fixture(fixture_name, tmp)
                    self.assertIsNone(report["recommended_family"])
                    self.assertEqual(report["recommended_families"], [])
                    self.assertEqual(report["routing_decision"]["status"], "ambiguous")
                    self.assertTrue(
                        any("architecture routing is ambiguous" in blocker for blocker in report["recommendation_blockers"]),
                        report["recommendation_blockers"],
                    )

    def test_confident_existing_family_keeps_compatible_primary_route(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "decoder.json"
            run_script("inspect_model.py", FIXTURES / "models" / "decoder", "--output", output)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["routing_decision"]["status"], "recommended")
            self.assertEqual(report["recommended_family"], "dense-decoder-transformer")
            self.assertEqual(report["recommended_families"], ["dense-decoder-transformer"])
            self.assertEqual(report["recommended_runbooks"], ["references/runbook-decoder-transformer.md"])
            self.assertGreaterEqual(
                report["routing_decision"]["winner_score"],
                report["routing_decision"]["minimum_score"],
            )

    def test_current_mlx_lm_hybrids_emit_composable_traits_and_all_runbooks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            for fixture_path in sorted(ROUTING_FIXTURES.glob("*-hybrid.json")):
                fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                with self.subTest(fixture=fixture_path.name):
                    report = inspect_fixture(fixture_path.name, tmp)
                    self.assertEqual(report["routing_decision"]["status"], fixture["expected_status"])
                    self.assertEqual(report["recommended_family"], fixture["expected_primary_family"])
                    self.assertEqual(report["recommended_families"], fixture["expected_families"])
                    self.assertEqual(report["recommended_runbooks"], fixture["expected_runbooks"])
                    self.assertTrue(set(fixture["expected_traits"]).issubset(report["architecture_traits"]))
                    self.assertEqual(report["architecture_profile"]["primary_family"], fixture["expected_primary_family"])
                    reference = report["architecture_profile"]["official_reference"]
                    self.assertRegex(reference, r"/blob/[0-9a-f]{40}/")
                    self.assertIn("ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd", reference)

    def test_known_overlapping_aliases_have_explicit_composition_profiles(self) -> None:
        registry = json.loads((SKILL / "assets" / "architectures.yaml").read_text(encoding="utf-8"))
        alias_families: dict[str, set[str]] = {}
        for family in registry["families"]:
            for alias in family.get("model_type_aliases", []):
                alias_families.setdefault(alias, set()).add(family["id"])
        duplicate_aliases = {
            alias: families for alias, families in alias_families.items() if len(families) > 1
        }
        profiles_by_alias = {
            alias: profile
            for profile in registry["hybrid_profiles"]
            for alias in profile.get("model_type_aliases", [])
        }
        self.assertTrue(duplicate_aliases)
        self.assertEqual(sorted(set(duplicate_aliases) - set(profiles_by_alias)), [])
        for alias, families in duplicate_aliases.items():
            with self.subTest(alias=alias):
                component_families = {
                    component["family"] for component in profiles_by_alias[alias]["components"]
                }
                self.assertTrue(families.issubset(component_families))

    def test_whisper_overlap_routes_to_both_asr_and_encoder_decoder_runbooks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            report = inspect_fixture("whisper-composition.json", Path(raw_tmp))
            fixture = json.loads(
                (ROUTING_FIXTURES / "whisper-composition.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["routing_decision"]["status"], fixture["expected_status"])
            self.assertEqual(report["recommended_family"], fixture["expected_primary_family"])
            self.assertEqual(report["recommended_families"], fixture["expected_families"])
            self.assertEqual(report["recommended_runbooks"], fixture["expected_runbooks"])
            self.assertTrue(set(fixture["expected_traits"]).issubset(report["architecture_traits"]))

    def test_hybrid_port_plan_surfaces_every_required_route(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection_data = inspect_fixture("qwen3-next-hybrid.json", tmp)
            inspection = tmp / "inspection.json"
            inspection.write_text(json.dumps(inspection_data), encoding="utf-8")
            plan = tmp / "PORT_PLAN.md"
            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                tmp / "qwen3-next-hybrid",
                "--output",
                plan,
            )
            text = plan.read_text(encoding="utf-8")
            self.assertIn("Primary family: `moe-decoder-transformer`", text)
            self.assertIn(
                "Secondary families: `ssm-recurrent-hybrid`, `dense-decoder-transformer`",
                text,
            )
            for runbook in inspection_data["recommended_runbooks"]:
                self.assertIn(f"`{runbook}`", text)
            for trait in inspection_data["architecture_traits"]:
                self.assertIn(f"`{trait}`", text)
            self.assertIn("`ssm-recurrent-hybrid`: recurrent/SSM state", text)
            self.assertIn("`dense-decoder-transformer`: per-layer KV cache", text)

    def test_hybrid_family_override_reorders_but_preserves_the_full_inspected_route(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection_data = inspect_fixture("qwen3-next-hybrid.json", tmp)
            inspection = tmp / "inspection.json"
            recommendations = tmp / "recommendations.json"
            plan = tmp / "PORT_PLAN.md"
            inspection.write_text(json.dumps(inspection_data), encoding="utf-8")
            override = "ssm-recurrent-hybrid"

            run_script(
                "recommend_optimizations.py",
                inspection,
                "--family",
                override,
                "--output",
                recommendations,
                "--limit",
                100,
            )
            report = json.loads(recommendations.read_text(encoding="utf-8"))
            expected_families = [
                override,
                *[family for family in inspection_data["recommended_families"] if family != override],
            ]
            self.assertEqual(report["family"], override)
            self.assertEqual(report["families"], expected_families)
            self.assertEqual(report["match_context"]["families"], sorted(expected_families))

            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                tmp / "qwen3-next-hybrid",
                "--family",
                override,
                "--recommendations",
                recommendations,
                "--output",
                plan,
            )
            text = plan.read_text(encoding="utf-8")
            self.assertIn(f"Primary family: `{override}`", text)
            for family in expected_families[1:]:
                self.assertIn(f"`{family}`", text)
            for runbook in inspection_data["recommended_runbooks"]:
                self.assertIn(f"`{runbook}`", text)
            for trait in inspection_data["architecture_traits"]:
                self.assertIn(f"`{trait}`", text)

    def test_registry_does_not_treat_generic_model_as_encoder_evidence(self) -> None:
        registry = json.loads((SKILL / "assets" / "architectures.yaml").read_text(encoding="utf-8"))
        encoder = next(family for family in registry["families"] if family["id"] == "encoder-transformer")
        self.assertNotIn("Model", encoder["class_patterns"])


if __name__ == "__main__":
    unittest.main()
