"""Canonical curriculum contracts for the public MLX learning atlas."""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
ASSETS = SKILL / "assets"
LEARNING = ASSETS / "learning_paths.json"
GENERATOR = SKILL / "scripts" / "generate_site_data.py"
SITE_DATA = ROOT / "site" / "data.js"
GLOBAL_PREFIX = "window.MLX_PORTING_SITE_DATA = "
if str(GENERATOR.parent) not in sys.path:
    sys.path.insert(0, str(GENERATOR.parent))

CHECKPOINTS = (
    "inspect",
    "oracle",
    "implement",
    "map",
    "parity",
    "profile",
    "optimize",
    "publish",
)
JOURNEYS = {
    "qwen25-dense-decoder": ("proven", {"dense-decoder-transformer"}),
    "whisper-style-asr": (
        "simulation",
        {"automatic-speech-recognition", "encoder-decoder-transformer"},
    ),
    "flux-style-diffusion": ("simulation", {"diffusion-flow"}),
    "llava-style-vlm": (
        "simulation",
        {"vision-language-omni", "encoder-transformer", "dense-decoder-transformer"},
    ),
}
OPTIMIZATION_FAMILIES = {
    "evaluation-scheduling",
    "native-operators-compilation",
    "layout-numerics",
    "state-memory",
    "compression",
    "inference-algorithms",
    "serving-pipeline",
    "custom-backend",
}
REQUIRED_FOUNDATIONS = {
    "what-is-mlx",
    "what-is-porting",
    "read-the-model",
    "correctness-rail",
    "first-divergence",
    "profile-bottleneck",
    "benchmark-honestly",
    "publish-proof",
}
REQUIRED_TRANSLATIONS = {
    "array-and-device",
    "lazy-evaluation",
    "weight-map",
    "tensor-layout",
    "numerical-dtype",
    "explicit-state",
    "stable-compilation",
    "custom-extensions",
    "timing-and-sync",
    "framework-bridges",
}
REQUIRED_GLOSSARY = {
    "mlx",
    "eager graph",
    "source oracle",
    "parity",
    "route",
    "runbook",
    "checkpoint",
    "weight map",
    "kv cache",
    "schema-2",
    "golden scenario",
    "receipt",
    "promotion-ready",
    "execution attestation",
    "targetprofile",
    "source-key coverage",
    "proof boundary",
}
REQUIRED_LEARNING_SOURCES = {
    "mlx-docs",
    "mlx-doc-quick-start",
    "mlx-doc-unified-memory",
    "mlx-doc-lazy",
    "mlx-doc-compile",
    "mlx-doc-framework-conversion",
    "mlx-doc-fast-sdpa",
    "mlx-repo",
    "mlx-examples-repo",
    "mlx-lm-repo",
}
TEACHING_FIELDS = {
    "plain_language",
    "why_it_matters",
    "pytorch_cuda",
    "mlx_translation",
    "example",
    "common_trap",
    "proof_check",
    "next_step",
}
NODE_FIELDS = {
    "concept",
    "why_mlx_differs",
    "inspect",
    "prerequisite",
    "proof",
    "evidence_state",
}


def load_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected a mapping in {path}")
    return value


def parse_site_data() -> dict[str, object]:
    text = SITE_DATA.read_text(encoding="utf-8")
    if not text.startswith(GLOBAL_PREFIX) or not text.endswith(";\n"):
        raise AssertionError("site data must remain one deterministic global assignment")
    value = json.loads(text[len(GLOBAL_PREFIX):-2])
    if not isinstance(value, dict):
        raise AssertionError("site data payload is not a mapping")
    return value


class LearningAtlasContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.learning = load_mapping(LEARNING)
        cls.architectures = load_mapping(ASSETS / "architectures.yaml")
        cls.guidance = load_mapping(ASSETS / "optimization_guidance.yaml")
        cls.sources = load_mapping(ASSETS / "sources.yaml")
        cls.techniques = load_mapping(ASSETS / "techniques.yaml")
        cls.outcomes = load_mapping(ASSETS / "model_outcomes.json")

    def test_curriculum_has_one_canonical_order_and_exactly_four_honest_journeys(self) -> None:
        self.assertEqual(self.learning["schema_version"], 1)
        self.assertEqual(tuple(self.learning["checkpoint_order"]), CHECKPOINTS)
        journey_rows = self.learning["journeys"]
        self.assertEqual(len(journey_rows), 4)
        self.assertEqual(len({row["id"] for row in journey_rows}), 4)
        journeys = {row["id"]: row for row in journey_rows}
        self.assertEqual(set(journeys), set(JOURNEYS))

        for journey_id, (status, architecture_ids) in JOURNEYS.items():
            journey = journeys[journey_id]
            with self.subTest(journey=journey_id):
                self.assertEqual(journey["status"], status)
                self.assertEqual(set(journey["architecture_ids"]), architecture_ids)
                self.assertEqual(set(journey["checkpoint_notes"]), set(CHECKPOINTS))
                self.assertTrue(all(note.strip() for note in journey["checkpoint_notes"].values()))
                self.assertTrue(journey["proof_boundary"].strip())
                self.assertTrue(journey["component_path"])
                self.assertTrue(journey["optimization_method_ids"])
                node_ids = [node["id"] for node in journey["component_path"]]
                self.assertEqual(len(node_ids), len(set(node_ids)))
                for node in journey["component_path"]:
                    self.assertTrue(NODE_FIELDS.issubset(node))
                    self.assertIn(node["checkpoint"], CHECKPOINTS)
                    self.assertEqual(node["evidence_state"], status)

        self.assertEqual(journeys["qwen25-dense-decoder"]["proof_ladder_rungs"], 29)
        self.assertEqual(
            journeys["qwen25-dense-decoder"]["model_id"],
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        self.assertEqual(
            journeys["qwen25-dense-decoder"]["model_outcome_id"],
            "qwen25-05b-instruct-local-worked-port",
        )
        qwen_outcome = next(
            row for row in self.outcomes["records"]
            if row["id"] == journeys["qwen25-dense-decoder"]["model_outcome_id"]
        )
        self.assertEqual(
            journeys["qwen25-dense-decoder"]["proof_boundary"],
            qwen_outcome["claim_boundary"],
        )
        self.assertEqual(
            journeys["whisper-style-asr"]["hybrid_profile_id"],
            "whisper-asr-seq2seq",
        )
        self.assertNotIn("proof_ladder_rungs", journeys["whisper-style-asr"])
        self.assertNotIn("proof_ladder_rungs", journeys["flux-style-diffusion"])
        self.assertNotIn("proof_ladder_rungs", journeys["llava-style-vlm"])

    def test_all_registry_references_are_canonical_and_optimization_methods_are_partitioned(self) -> None:
        architecture_ids = {row["id"] for row in self.architectures["families"]}
        guidance_by_id = {row["id"]: row for row in self.guidance["methods"]}
        guidance_ids = set(guidance_by_id)
        technique_ids = {row["id"] for row in self.techniques["techniques"]}
        source_rows = {row["id"]: row for row in self.sources["sources"]}
        hybrid_profile_ids = {row["id"] for row in self.architectures["hybrid_profiles"]}
        outcome_ids = {row["id"] for row in self.outcomes["records"]}

        journey_architectures = {
            identifier
            for journey in self.learning["journeys"]
            for identifier in journey["architecture_ids"]
        }
        self.assertTrue(journey_architectures.issubset(architecture_ids))
        self.assertIn("whisper-asr-seq2seq", hybrid_profile_ids)
        self.assertIn("qwen25-05b-instruct-local-worked-port", outcome_ids)

        optimization_rows = {row["id"]: row for row in self.learning["optimization_families"]}
        self.assertEqual(set(optimization_rows), OPTIMIZATION_FAMILIES)
        categorized = [
            method
            for row in optimization_rows.values()
            for method in row["method_ids"]
        ]
        self.assertEqual(set(categorized), guidance_ids)
        self.assertEqual(len(categorized), len(set(categorized)), "methods must have one family")
        self.assertTrue(
            {guidance_by_id[identifier]["technique_id"] for identifier in categorized}.issubset(
                technique_ids,
            ),
        )
        self.assertIn("fast-sdpa", optimization_rows["native-operators-compilation"]["method_ids"])
        self.assertIn("draft-model-speculation", optimization_rows["inference-algorithms"]["method_ids"])
        self.assertNotIn(
            "draft-model-speculation",
            optimization_rows["native-operators-compilation"]["method_ids"],
        )
        for row in optimization_rows.values():
            self.assertTrue({"bottleneck", "proof_gate", "rollback"}.issubset(row))

        for journey in self.learning["journeys"]:
            self.assertTrue(set(journey["optimization_method_ids"]).issubset(guidance_ids))
        journey_by_id = {row["id"]: row for row in self.learning["journeys"]}
        self.assertNotIn("fast-sdpa", journey_by_id["flux-style-diffusion"]["optimization_method_ids"])
        self.assertNotIn(
            "audio-streaming-and-cache",
            journey_by_id["whisper-style-asr"]["optimization_method_ids"],
        )

        learning_source_ids = set(self.learning["official_learning_source_ids"])
        self.assertTrue(REQUIRED_LEARNING_SOURCES.issubset(learning_source_ids))
        self.assertTrue(learning_source_ids.issubset(source_rows))
        for source_id in learning_source_ids:
            parsed = urlsplit(source_rows[source_id]["url"])
            with self.subTest(source=source_id):
                self.assertEqual(parsed.scheme, "https")
                self.assertIn(parsed.netloc, {"ml-explore.github.io", "github.com"})
                if parsed.netloc == "github.com":
                    self.assertTrue(parsed.path.startswith("/ml-explore/"))

    def test_foundations_translation_lens_and_glossary_cover_the_teaching_contract(self) -> None:
        foundations = {row["id"]: row for row in self.learning["foundations"]}
        translations = {row["id"]: row for row in self.learning["translation_lens"]}
        glossary = {row["term"].casefold(): row for row in self.learning["glossary"]}

        self.assertTrue(REQUIRED_FOUNDATIONS.issubset(foundations))
        self.assertEqual(set(translations), REQUIRED_TRANSLATIONS)
        self.assertTrue(REQUIRED_GLOSSARY.issubset(glossary))
        for collection in (foundations, translations):
            for identifier, row in collection.items():
                with self.subTest(item=identifier):
                    self.assertTrue(TEACHING_FIELDS.issubset(row))
                    for field in TEACHING_FIELDS:
                        self.assertIsInstance(row[field], str)
                        self.assertTrue(row[field].strip())
        for term, row in glossary.items():
            with self.subTest(term=term):
                self.assertTrue(row["definition"].strip())

    def test_generator_validates_dangling_learning_references(self) -> None:
        spec = importlib.util.spec_from_file_location("generate_site_data", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        architecture_ids = {row["id"] for row in self.architectures["families"]}
        guidance_by_id = {row["id"]: row for row in self.guidance["methods"]}
        technique_ids = {row["id"] for row in self.techniques["techniques"]}
        source_ids = {row["id"] for row in self.sources["sources"]}
        hybrid_profile_ids = {row["id"] for row in self.architectures["hybrid_profiles"]}
        outcome_ids = {row["id"] for row in self.outcomes["records"]}
        module.validate_learning(
            self.learning,
            architecture_ids,
            hybrid_profile_ids,
            guidance_by_id,
            technique_ids,
            source_ids,
            outcome_ids,
        )

        bad = copy.deepcopy(self.learning)
        bad["journeys"][0]["architecture_ids"].append("missing-family")
        with self.assertRaisesRegex(module.SkillError, "missing-family"):
            module.validate_learning(
                bad,
                architecture_ids,
                hybrid_profile_ids,
                guidance_by_id,
                technique_ids,
                source_ids,
                outcome_ids,
            )

        bad = copy.deepcopy(self.learning)
        bad["optimization_families"][0]["method_ids"].append("missing-method")
        with self.assertRaisesRegex(module.SkillError, "missing-method"):
            module.validate_learning(
                bad,
                architecture_ids,
                hybrid_profile_ids,
                guidance_by_id,
                technique_ids,
                source_ids,
                outcome_ids,
            )

        for field in ("title", "bottleneck", "proof_gate", "rollback"):
            bad = copy.deepcopy(self.learning)
            bad["optimization_families"][0][field] = ""
            with self.assertRaisesRegex(module.SkillError, field):
                module.validate_learning(
                    bad,
                    architecture_ids,
                    hybrid_profile_ids,
                    guidance_by_id,
                    technique_ids,
                    source_ids,
                    outcome_ids,
                )

        for invalid_boundary in (None, "   "):
            bad = copy.deepcopy(self.learning)
            if invalid_boundary is None:
                del bad["journeys"][0]["proof_boundary"]
            else:
                bad["journeys"][0]["proof_boundary"] = invalid_boundary
            with self.assertRaisesRegex(module.SkillError, "proof boundary"):
                module.validate_learning(
                    bad,
                    architecture_ids,
                    hybrid_profile_ids,
                    guidance_by_id,
                    technique_ids,
                    source_ids,
                    outcome_ids,
                )

        bad_guidance = copy.deepcopy(guidance_by_id)
        categorized_method = self.learning["optimization_families"][0]["method_ids"][0]
        bad_guidance[categorized_method]["technique_id"] = "missing-technique"
        with self.assertRaisesRegex(module.SkillError, "missing-technique"):
            module.validate_learning(
                self.learning,
                architecture_ids,
                hybrid_profile_ids,
                bad_guidance,
                technique_ids,
                source_ids,
                outcome_ids,
            )

        bad = copy.deepcopy(self.learning)
        bad["official_learning_source_ids"].append("missing-source")
        with self.assertRaisesRegex(module.SkillError, "missing-source"):
            module.validate_learning(
                bad,
                architecture_ids,
                hybrid_profile_ids,
                guidance_by_id,
                technique_ids,
                source_ids,
                outcome_ids,
            )

        bad = copy.deepcopy(self.learning)
        bad["journeys"][0]["model_outcome_id"] = "missing-outcome"
        with self.assertRaisesRegex(module.SkillError, "missing-outcome"):
            module.validate_learning(
                bad,
                architecture_ids,
                hybrid_profile_ids,
                guidance_by_id,
                technique_ids,
                source_ids,
                outcome_ids,
            )

        bad = copy.deepcopy(self.learning)
        bad["journeys"][1]["hybrid_profile_id"] = "missing-profile"
        with self.assertRaisesRegex(module.SkillError, "missing-profile"):
            module.validate_learning(
                bad,
                architecture_ids,
                hybrid_profile_ids,
                guidance_by_id,
                technique_ids,
                source_ids,
                outcome_ids,
            )

    def test_generated_site_payload_contains_the_validated_canonical_curriculum(self) -> None:
        generated = parse_site_data()["learning"]
        for key, value in self.learning.items():
            if key != "journeys":
                self.assertEqual(generated[key], value)
                continue
            self.assertEqual(len(generated[key]), len(value))
            for canonical, rendered in zip(value, generated[key], strict=True):
                for field, field_value in canonical.items():
                    self.assertEqual(rendered[field], field_value)
                self.assertTrue(rendered["runbooks"])

        source_by_id = {row["id"]: row for row in self.sources["sources"]}
        self.assertEqual(
            generated["official_learning_sources"],
            [
                {
                    "id": identifier,
                    "title": source_by_id[identifier]["title"],
                    "url": source_by_id[identifier]["url"],
                }
                for identifier in self.learning["official_learning_source_ids"]
            ],
        )
        guidance_by_id = {row["id"]: row for row in self.guidance["methods"]}
        claims = load_mapping(ASSETS / "effective_claims.json")
        claim_state_by_method = {
            row["method_id"]: row["promotion_state"]
            for row in claims["claims"]
        }
        expected_methods = {
            method
            for family in self.learning["optimization_families"]
            for method in family["method_ids"]
        }
        generated_methods = {row["id"]: row for row in generated["guidance_methods"]}
        self.assertEqual(set(generated_methods), expected_methods)
        for identifier, row in generated_methods.items():
            with self.subTest(method=identifier):
                canonical = guidance_by_id[identifier]
                for field in (
                    "technique_id",
                    "status",
                    "applies_to",
                    "recommendation",
                    "tradeoffs",
                    "validation_gates",
                    "rollback_conditions",
                ):
                    self.assertEqual(row[field], canonical[field])
                self.assertNotIn(canonical["expected_effect"], row["expected_effect"])
                self.assertIn("No numeric effect is claimed", row["expected_effect"])
                self.assertEqual(
                    row["claim_eligibility"],
                    claim_state_by_method.get(identifier, "not-catalogued"),
                )
                self.assertEqual(row["numeric_authority"], "effective_claims")

    def test_research_report_uses_the_canonical_numeric_record_count(self) -> None:
        report = (ROOT / "RESEARCH_REPORT.md").read_text(encoding="utf-8")
        claims = load_mapping(ASSETS / "effective_claims.json")
        expected = len(claims["claims"])
        match = re.search(
            r"At this release boundary, all (\d+) catalogued numeric records are withheld\.",
            report,
        )
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), expected)

    def test_public_learning_payload_contains_no_numeric_speedup_claim(self) -> None:
        learning = parse_site_data()["learning"]
        claims = load_mapping(ASSETS / "effective_claims.json")["claims"]
        claim_by_method = {claim["method_id"]: claim for claim in claims}
        for method in learning["guidance_methods"]:
            claim = claim_by_method.get(method["id"])
            expected = (
                {
                    "range": claim["effective_range"],
                    "metric": claim["metric"],
                    "target_constraints": claim["target_constraints"],
                    "experiment_fingerprint": claim["experiment_fingerprint"],
                }
                if claim is not None and claim["promotion_state"] == "local-promotion"
                else None
            )
            with self.subTest(method=method["id"]):
                self.assertEqual(method["numeric_claim"], expected)
        serialized = json.dumps(learning, ensure_ascii=False)
        for forbidden_key in ("observed_range", "profile_eligible_range", "improvement_band"):
            self.assertNotIn(forbidden_key, serialized)

    def test_current_snapshot_counts_and_nightly_graph_cover_the_canonical_sources(self) -> None:
        source_rows = self.sources["sources"]
        classified = sum(
            bool(row.get("support_scope")) and bool(row.get("claim_types"))
            for row in source_rows
        )
        unclassified = len(source_rows) - classified
        script_count = len(list((SKILL / "scripts").glob("*.py")))
        expected_source_copy = (
            f"{len(source_rows)} evidence sources with explicit review depth; "
            f"{classified} currently carry classified"
        )
        expected_report_copy = (
            f"| Evidence sources | {len(source_rows)} | Every source has review depth; "
            f"{classified} carry classified support scope and claim types, while "
            f"{unclassified} remain intentionally unclassified. |"
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        report = (ROOT / "RESEARCH_REPORT.md").read_text(encoding="utf-8")
        docs = (ROOT / "site" / "docs" / "index.html").read_text(encoding="utf-8")
        self.assertIn(expected_source_copy, readme)
        self.assertIn(f"- {script_count} inspectable Python scripts", readme)
        self.assertIn(expected_report_copy, report)
        self.assertIn(f"| Python scripts | {script_count} |", report)
        self.assertIn("T-7007 knowledge-layer state:\n697 nodes, 499 edges", report)
        self.assertIn(
            f"registry snapshot reviewed on {self.sources['reviewed']}",
            docs,
        )

        graph = load_mapping(ASSETS / "knowledge_graph.json")
        graph_source_ids = {
            node["source_id"]
            for node in graph["nodes"]
            if node.get("kind") == "source" and isinstance(node.get("source_id"), str)
        }
        self.assertEqual({row["id"] for row in source_rows} - graph_source_ids, set())


if __name__ == "__main__":
    unittest.main()
