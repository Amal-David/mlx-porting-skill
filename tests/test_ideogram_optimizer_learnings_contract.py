"""Guards for evidence-gated optimizer learnings derived from fal's Ideogram V4 write-up."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
ASSETS = SKILL / "assets"
REFERENCES = SKILL / "references"

SOURCE_IDS = {
    "fal-ideogram-dspark-2026-07-08",
    "paper-2607-05147",
    "fal-ideogram-v4-serving-2026-07-09",
    "paper-2601-20088",
    "paper-2405-14867",
}

TECHNIQUE_IDS = {
    "dspark",
    "diffusion-quantization-aware-distillation",
    "cfg-guidance-distillation",
    "diffusion-step-distillation",
}


class IdeogramOptimizerLearningContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = json.loads(
            (ASSETS / "sources.yaml").read_text(encoding="utf-8")
        )["sources"]
        cls.techniques = json.loads(
            (ASSETS / "techniques.yaml").read_text(encoding="utf-8")
        )["techniques"]

    def test_new_sources_are_context_or_paper_evidence_not_mlx_support(self) -> None:
        by_id = {source["id"]: source for source in self.sources}
        self.assertTrue(SOURCE_IDS.issubset(by_id))
        for source_id in SOURCE_IDS:
            with self.subTest(source=source_id):
                source = by_id[source_id]
                self.assertEqual(source["review_depth"], "synthesized")
                self.assertIn(source["support_scope"], {"context_only", "paper_only"})
                self.assertNotIn("mlx_implementation", source["claim_types"])
                self.assertNotEqual(source["evidence_class"], "local_benchmark_artifact")

    def test_new_techniques_remain_research_candidates_with_real_gates(self) -> None:
        source_ids = {source["id"] for source in self.sources}
        by_id = {technique["id"]: technique for technique in self.techniques}
        self.assertTrue(TECHNIQUE_IDS.issubset(by_id))
        for technique_id in TECHNIQUE_IDS:
            with self.subTest(technique=technique_id):
                technique = by_id[technique_id]
                self.assertEqual(technique["status"], "research-candidate")
                self.assertTrue(set(technique["evidence"]).issubset(source_ids))
                self.assertIn("MLX", technique["validation_gate"])
                self.assertTrue(technique["rollback_condition"])
                self.assertIsNone(
                    re.search(r"\b\d+(?:\.\d+)?x\b", technique["validation_gate"])
                )

    def test_diffusion_runbook_preserves_cost_axes_and_quality_boundaries(self) -> None:
        runbook = re.sub(
            r"\s+",
            " ",
            (REFERENCES / "runbook-diffusion-flow.md").read_text(encoding="utf-8"),
        )
        for text in (
            "denoising steps x denoiser branches per step x cost per branch",
            "do not multiply unmeasured speedups",
            "A falling teacher/student loss is not",
            "not supported MLX speed claims",
            "Do not translate a CUTLASS epilogue",
        ):
            self.assertIn(text, runbook)

    def test_serving_and_kernel_guides_reject_hardware_transfer_shortcuts(self) -> None:
        serving = re.sub(
            r"\s+", " ",
            (REFERENCES / "decoding-and-serving.md").read_text(encoding="utf-8"),
        )
        kernels = re.sub(
            r"\s+", " ",
            (REFERENCES / "compile-and-kernels.md").read_text(encoding="utf-8"),
        )
        quantization = re.sub(
            r"\s+", " ",
            (REFERENCES / "quantization.md").read_text(encoding="utf-8"),
        )
        for text in (
            "acceptance survival by draft position",
            "single-user and intended concurrent serving loads",
            "not Apple Silicon performance claims",
        ):
            self.assertIn(text, serving)
        for text in (
            "Do not call a separate post-op Metal dispatch a fused quantized-matmul epilogue",
            "NVIDIA-specific",
            "reversible conversion map",
        ):
            self.assertIn(text, kernels)
        for text in (
            "Re-profile after lowering the matmul",
            "Training loss alone is not a quality gate",
            "neither proves",
        ):
            self.assertIn(text, quantization)


if __name__ == "__main__":
    unittest.main()
