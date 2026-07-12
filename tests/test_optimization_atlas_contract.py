"""Canonical contracts for the parity-gated optimization study atlas."""
from __future__ import annotations

import json
import importlib.util
import re
import shutil
import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEARNING = ROOT / "mlx-model-porting" / "assets" / "learning_paths.json"
GUIDANCE = ROOT / "mlx-model-porting" / "assets" / "optimization_guidance.yaml"
STACKS = ROOT / "mlx-model-porting" / "assets" / "optimization_stacks.yaml"
SITE_DATA = ROOT / "site" / "data.js"
DOCS = ROOT / "site" / "docs" / "index.html"
SCRIPT = ROOT / "site" / "optimization.js"
GENERATOR = ROOT / "mlx-model-porting" / "scripts" / "generate_site_data.py"
if str(GENERATOR.parent) not in sys.path:
    sys.path.insert(0, str(GENERATOR.parent))
GLOBAL_PREFIX = "window.MLX_PORTING_SITE_DATA = "
ASSETS = ROOT / "mlx-model-porting" / "assets"

FAMILY_IDS = (
    "evaluation-scheduling",
    "native-operators-compilation",
    "layout-numerics",
    "state-memory",
    "compression",
    "inference-algorithms",
    "serving-pipeline",
    "custom-backend",
)


def load_site_data() -> dict[str, object]:
    text = SITE_DATA.read_text(encoding="utf-8")
    if not text.startswith(GLOBAL_PREFIX) or not text.endswith(";\n"):
        raise AssertionError("site data must remain one deterministic global assignment")
    return json.loads(text[len(GLOBAL_PREFIX):-2])


class TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


class OptimizationAtlasContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.learning = json.loads(LEARNING.read_text(encoding="utf-8"))
        cls.guidance = json.loads(GUIDANCE.read_text(encoding="utf-8"))
        cls.site_data = load_site_data()
        cls.node = shutil.which("node")

    def test_taxonomy_covers_every_method_once_and_keeps_key_branches_separate(self) -> None:
        families = self.learning["optimization_families"]
        self.assertEqual(tuple(row["id"] for row in families), FAMILY_IDS)
        categorized = [method for family in families for method in family["method_ids"]]
        canonical = [method["id"] for method in self.guidance["methods"]]
        self.assertCountEqual(categorized, canonical)
        self.assertEqual(len(categorized), len(set(categorized)))
        family_by_method = {
            method: family["id"] for family in families for method in family["method_ids"]
        }
        self.assertEqual(family_by_method["fast-sdpa"], "native-operators-compilation")
        self.assertEqual(family_by_method["draft-model-speculation"], "inference-algorithms")
        self.assertNotEqual(
            family_by_method["fast-sdpa"],
            family_by_method["draft-model-speculation"],
        )
        for family in families:
            for field in ("title", "bottleneck", "proof_gate", "rollback"):
                self.assertTrue(family[field].strip())

    def test_generated_methods_are_explanatory_linked_and_numeric_fail_closed(self) -> None:
        learning = self.site_data["learning"]
        methods = learning["guidance_methods"]
        self.assertEqual(len(methods), len(self.guidance["methods"]))
        required = {
            "id", "title", "status", "applies_to", "recommendation", "expected_effect",
            "tradeoffs", "validation_gates", "rollback_conditions", "evidence_links",
            "claim_eligibility", "numeric_authority", "numeric_claim", "quality_gated",
            "family_id", "prerequisite", "proof_gate", "quality_gate",
            "advisor",
        }
        for method in methods:
            with self.subTest(method=method["id"]):
                self.assertTrue(required.issubset(method))
                self.assertTrue(method["title"].strip())
                self.assertTrue(method["applies_to"])
                self.assertTrue(method["tradeoffs"])
                self.assertTrue(method["validation_gates"])
                self.assertTrue(method["rollback_conditions"])
                self.assertTrue(method["evidence_links"])
                self.assertIn("requires_user_opt_in", method["advisor"])
                self.assertEqual(method["canonical_source"], method["evidence_links"][0])
                self.assertTrue(all(link["review_depth"] != "indexed" for link in method["evidence_links"]))
                self.assertTrue(all(link["support_scope"] for link in method["evidence_links"]))
                self.assertEqual(method["numeric_authority"], "effective_claims")
                self.assertIsNone(method["numeric_claim"])
                self.assertNotRegex(method["expected_effect"], r"(?:\d+(?:\.\d+)?\s*[x×%])")
                self.assertIn("Targets ", method["expected_effect"])
                self.assertTrue(all(link["url"].startswith("https://") for link in method["evidence_links"]))
        generator = GENERATOR.read_text(encoding="utf-8")
        self.assertIn('claim.get("promotion_state") == "local-promotion"', generator)
        self.assertNotIn('claim.get("promotion_state") == "effective"', generator)
        expected_canonical_sources = {
            "audio-streaming-and-cache": "mlx-audio-release-044",
            "multimodal-content-prefix-cache": "vllm-mlx-mllm-batch-source",
            "native-low-bit-weight-quantization": "mlx-doc-core-quantize",
            "spatial-grid-sample-kernel": "katlun-grid-sample-source",
            "video-input-budgeting": "vllm-mlx-multimodal-guide",
        }
        by_id = {method["id"]: method for method in methods}
        for method_id, source_id in expected_canonical_sources.items():
            self.assertEqual(by_id[method_id]["canonical_source"]["id"], source_id)

    def test_synthetic_local_promotion_keeps_range_metric_scope_and_fingerprint_together(self) -> None:
        spec = importlib.util.spec_from_file_location("generate_site_data", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sources = json.loads((ASSETS / "sources.yaml").read_text(encoding="utf-8"))["sources"]
        techniques = json.loads((ASSETS / "techniques.yaml").read_text(encoding="utf-8"))["techniques"]
        source_by_id = {row["id"]: row for row in sources}
        technique_by_id = {row["id"]: row for row in techniques}
        guidance_by_id = {row["id"]: row for row in self.guidance["methods"]}
        architecture_by_id = {
            row["id"]: row for row in self.site_data["architectures"]["families"]
        }
        advisor_by_status = {
            method["status"]: method["advisor"]
            for method in self.site_data["learning"]["guidance_methods"]
        }
        claim = {
            "promotion_state": "local-promotion",
            "effective_range": "1.2x-1.3x",
            "metric": "wall-time",
            "target_constraints": {"hardware": {"chip": "fixture-chip"}},
            "experiment_fingerprint": "f" * 64,
        }
        payload = module.build_learning_payload(
            self.learning,
            source_by_id,
            technique_by_id,
            guidance_by_id,
            {"fast-sdpa": claim},
            architecture_by_id,
            advisor_by_status,
        )
        method = next(row for row in payload["guidance_methods"] if row["id"] == "fast-sdpa")
        self.assertEqual(method["claim_eligibility"], "local-promotion")
        self.assertEqual(method["numeric_claim"], {
            "range": claim["effective_range"],
            "metric": claim["metric"],
            "target_constraints": claim["target_constraints"],
            "experiment_fingerprint": claim["experiment_fingerprint"],
        })
        self.assertNotIn("No numeric effect is claimed", method["expected_effect"])

    def test_quality_gated_methods_are_explicit_and_canonical(self) -> None:
        expected = set(self.learning["method_quality_gates"])
        generated = {
            method["id"] for method in self.site_data["learning"]["guidance_methods"]
            if method["quality_gated"]
        }
        self.assertEqual(generated, expected)
        self.assertTrue({
            "bf16-weight-cast",
            "native-low-bit-weight-quantization",
            "uniform-kv-quantization",
            "adaptive-kv-quantization",
            "visual-token-pruning-or-merge",
            "moe-expert-dispatch-and-quantization",
            "video-input-budgeting",
        }.issubset(expected))
        stacks = json.loads(STACKS.read_text(encoding="utf-8"))
        conditionally_lossy = {
            step["method"]
            for stack in stacks["stacks"]
            for step in stack["steps"]
            if step.get("lossiness") == "conditionally-lossy"
        }
        self.assertTrue(conditionally_lossy.issubset(expected))

    def test_docs_progressively_enhance_eight_semantic_families(self) -> None:
        html = DOCS.read_text(encoding="utf-8")
        parser = TagCollector()
        parser.feed(html)
        fallback = [attrs for tag, attrs in parser.tags if tag == "div" and "data-optimization-fallback" in attrs]
        root = [attrs for tag, attrs in parser.tags if tag == "div" and "data-optimization-root" in attrs]
        radios = [
            attrs for tag, attrs in parser.tags
            if tag == "input" and attrs.get("name") == "optimization-family"
        ]
        self.assertEqual(len(fallback), 1)
        self.assertNotIn("hidden", fallback[0])
        self.assertEqual(len(root), 1)
        self.assertIn("hidden", root[0])
        self.assertEqual(tuple(radio["value"] for radio in radios), ("not-measured", *FAMILY_IDS))
        self.assertIn("Parity first. Profile second. One experiment at a time.", html)
        self.assertIn("data-optimization-methods", html)
        self.assertIn("data-optimization-proof-gate", html)
        fallback_method_ids = re.findall(r'data-optimization-fallback-method="([^"]+)"', html)
        canonical_method_ids = [
            method for family in self.learning["optimization_families"] for method in family["method_ids"]
        ]
        self.assertCountEqual(fallback_method_ids, canonical_method_ids)
        self.assertEqual(len(fallback_method_ids), len(set(fallback_method_ids)))
        for rejected_id in ("content-prefix-cache-vlm", "cuda-graphs-decode-capture"):
            self.assertRegex(
                html,
                rf'data-optimization-fallback-method="{re.escape(rejected_id)}"[^>]*>'
                rf'[^<]*(?:<small>)?[^<]*(?:rejected|superseded)',
            )

    def test_dependency_free_script_exports_validation_and_mounting_api(self) -> None:
        self.assertTrue(SCRIPT.is_file())
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\brequire\s*\(")
        self.assertNotRegex(source, r"\bimport\s+(?:[^.(]|$)")
        checked = subprocess.run(
            [self.node or "node", "--check", str(SCRIPT)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        harness = """
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const optimization = require("./site/optimization.js");
const sandbox = {window: {}};
vm.runInNewContext(fs.readFileSync("./site/data.js", "utf8"), sandbox);
const learning = sandbox.window.MLX_PORTING_SITE_DATA.learning;
for (const name of ["validateOptimizationLearning", "statusKind", "selectionEligibility", "exportExperimentPlan", "mountOptimizationAtlas"]) {
  assert.equal(typeof optimization[name], "function", `${name} must be exported`);
}
assert.equal(optimization.selectionEligibility({}, {
  parityDeclared: false, familySelected: true, researchOptIn: true, journeyCurated: true,
}).allowed, false);
assert.equal(optimization.selectionEligibility({}, {
  parityDeclared: true, familySelected: false, researchOptIn: true, journeyCurated: true,
}).allowed, false);
assert.equal(optimization.selectionEligibility({status: "research-candidate", advisor: {
  id: "experimental-approach", requires_user_opt_in: true,
}}, {parityDeclared: true, familySelected: true, researchOptIn: false, journeyCurated: true}).allowed, false);
assert.equal(optimization.selectionEligibility({status: "research-candidate", advisor: {
  id: "experimental-approach", requires_user_opt_in: true,
}}, {parityDeclared: true, familySelected: true, researchOptIn: true, journeyCurated: true}).allowed, true);
assert.equal(optimization.selectionEligibility({status: "rejected-or-superseded", advisor: {
  id: "rejected-do-not-use", requires_user_opt_in: false,
}}, {parityDeclared: true, familySelected: true, researchOptIn: true, journeyCurated: true}).allowed, false);
assert.equal(optimization.selectionEligibility({status: "native-mlx", advisor: {
  id: "validated-source-theory", requires_user_opt_in: false,
}}, {parityDeclared: true, familySelected: true, researchOptIn: true, journeyCurated: false}).allowed, false);
assert.equal(optimization.validateOptimizationLearning(learning), true);
for (const mutate of [
  (copy) => { copy.guidance_methods[0].status = "mystery-supported"; },
  (copy) => { copy.guidance_methods.find((method) => method.status === "research-candidate").advisor = {
    id: "validated-source-theory", label: "Wrong", description: "Wrong", requires_user_opt_in: false,
  }; },
  (copy) => { copy.guidance_methods[0].family_id = "compression"; },
  (copy) => { copy.guidance_methods[0].quality_gated = true; copy.guidance_methods[0].quality_gate = []; },
  (copy) => { copy.guidance_methods[0].evidence_links[0].url = "javascript:alert(1)"; },
  (copy) => { copy.guidance_methods[0].numeric_claim = {range: "2x"}; },
  (copy) => { copy.guidance_methods[0].claim_eligibility = "local-promotion"; copy.guidance_methods[0].numeric_claim = null; },
  (copy) => { copy.journeys = []; },
  (copy) => { copy.journeys[0].runbooks = []; },
  (copy) => { copy.journeys[0].runbooks[0].path = "../escape.md"; },
  (copy) => { copy.guidance_methods[0].canonical_source.url = "javascript:alert(1)"; },
  (copy) => { copy.guidance_methods[0].evidence_links[0].review_depth = "unknown"; },
  (copy) => { copy.guidance_methods[0].evidence_links[0].claim_types = [3]; },
  (copy) => { copy.guidance_methods[0].claim_eligibility = "local-promotion"; copy.guidance_methods[0].numeric_claim = {
    range: "2x", metric: "wall-time", target_constraints: {}, experiment_fingerprint: "f".repeat(64),
  }; },
]) {
  const copy = JSON.parse(JSON.stringify(learning));
  mutate(copy);
  assert.equal(optimization.validateOptimizationLearning(copy), false);
}
const methodById = new Map(learning.guidance_methods.map((method) => [method.id, method]));
for (const journey of learning.journeys) {
  const selectedId = journey.optimization_method_ids[0];
  const plan = optimization.exportExperimentPlan(new Set([selectedId]), methodById, journey);
  assert.ok(plan.includes(journey.title));
  assert.ok(plan.includes(journey.proof_boundary));
  assert.ok(plan.includes(selectedId));
  for (const runbook of journey.runbooks) assert.ok(plan.includes(runbook.path));
}
"""
        result = subprocess.run(
            [self.node or "node", "-e", harness], cwd=ROOT, check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_port_plan_export_contains_real_runbooks_and_tested_commands(self) -> None:
        atlas = (ROOT / "site" / "atlas.js").read_text(encoding="utf-8")
        self.assertIn("Tested repository handoff", atlas)
        self.assertIn("journey.runbooks", atlas)
        for command in (
            "inspect_model.py MODEL_PATH --output inspection.json",
            "recommend_optimizations.py inspection.json",
            "audit_skill.py --strict mlx-model-porting",
        ):
            self.assertIn(command, atlas)
        for journey in self.site_data["learning"]["journeys"]:
            self.assertTrue(journey["runbooks"])
            self.assertTrue(all(runbook["path"].startswith("references/runbook-") for runbook in journey["runbooks"]))


if __name__ == "__main__":
    unittest.main()
