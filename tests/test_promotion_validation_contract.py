"""Cross-registry guards for benchmark observation and claim promotion state."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_tooling import SKILL, run_script


class PromotionValidationContractTests(unittest.TestCase):
    def _copy_skill(self, root: Path) -> Path:
        copied = root / "skill"
        shutil.copytree(SKILL / "assets", copied / "assets")
        return copied

    def test_current_observation_receipts_resolve_to_nonpromoted_assessments(self) -> None:
        guidance = json.loads((SKILL / "assets" / "optimization_guidance.yaml").read_text(encoding="utf-8"))
        assessments = json.loads(
            (SKILL / "assets" / "benchmarks" / "receipt_assessments.json").read_text(encoding="utf-8")
        )
        by_receipt = {row["receipt"]: row for row in assessments["assessments"]}
        observed = [
            method
            for method in guidance["methods"]
            if (method.get("improvement_band") or {}).get("provenance") == "performance_observation"
        ]
        self.assertTrue(observed)
        for method in observed:
            with self.subTest(method=method["id"]):
                band = method["improvement_band"]
                self.assertEqual(band["claim_status"], "held")
                self.assertTrue(band["receipts"])
                for receipt in band["receipts"]:
                    self.assertIn(receipt, by_receipt)
                    self.assertNotEqual(by_receipt[receipt]["classification"], "rejected")

    def test_local_reproduced_claim_requires_generated_local_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            skill = self._copy_skill(Path(raw_tmp))
            guidance_path = skill / "assets" / "optimization_guidance.yaml"
            guidance = json.loads(guidance_path.read_text(encoding="utf-8"))
            method = next(
                item for item in guidance["methods"]
                if item["id"] == "native-low-bit-weight-quantization"
            )
            method["improvement_band"]["provenance"] = "local_reproduced"
            method["improvement_band"]["claim_status"] = "eligible-with-profile"
            guidance_path.write_text(json.dumps(guidance, indent=2) + "\n", encoding="utf-8")

            result = run_script("validate_sources.py", skill, expected=1)
            self.assertIn(
                "local_reproduced improvement_band for native-low-bit-weight-quantization "
                "is not backed by a local-promotion effective claim",
                result.stdout,
            )

    def test_observation_requires_assessment_and_matching_receipt_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            skill = self._copy_skill(Path(raw_tmp))
            assessment_path = skill / "assets" / "benchmarks" / "receipt_assessments.json"
            assessments = json.loads(assessment_path.read_text(encoding="utf-8"))
            assessments["assessments"] = [
                row for row in assessments["assessments"] if row["receipt"] != "quant-4bit.json"
            ]
            assessment_path.write_text(json.dumps(assessments, indent=2) + "\n", encoding="utf-8")

            result = run_script("validate_sources.py", skill, expected=1)
            self.assertIn(
                "performance_observation improvement_band for native-low-bit-weight-quantization "
                "receipt lacks assessment: quant-4bit.json",
                result.stdout,
            )

        with tempfile.TemporaryDirectory() as raw_tmp:
            skill = self._copy_skill(Path(raw_tmp))
            receipt = skill / "assets" / "benchmarks" / "quant-4bit.json"
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["aggregate"]["generation_tps"]["median"] = 999
            receipt.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

            result = run_script("validate_sources.py", skill, expected=1)
            self.assertIn(
                "benchmark assessment digest mismatch for quant-4bit.json",
                result.stdout,
            )
            self.assertNotEqual(
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                next(
                    row["receipt_sha256"]
                    for row in json.loads(
                        (skill / "assets" / "benchmarks" / "receipt_assessments.json").read_text(
                            encoding="utf-8"
                        )
                    )["assessments"]
                    if row["receipt"] == "quant-4bit.json"
                ),
            )


if __name__ == "__main__":
    unittest.main()
