"""Scored golden-scenario harness.

Exercises the skill the way an agent would experience it end to end — inspect ->
route -> weight-key coverage -> staged parity (a seeded bug that MUST be caught) ->
optimization inclusion/exclusion — and asserts an aggregate scorecard. Coverage is
reported honestly: families without a shipped fixture are listed, not silently skipped.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Allow `python3 -m unittest test_scenarios` from the repo root, not just discovery.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_tooling import FIXTURES, SKILL, run_script

SCENARIOS_DIR = FIXTURES / "scenarios"
ALL_FAMILIES = {f["id"] for f in json.loads((SKILL / "assets" / "architectures.yaml").read_text())["families"]}
EXPECTED_COVERED = ALL_FAMILIES


class ScenarioHarnessTests(unittest.TestCase):
    def test_golden_scenarios_score_full_marks(self) -> None:
        scenarios = sorted(SCENARIOS_DIR.glob("*.json"))
        self.assertTrue(scenarios, "no scenarios found")
        score = {"routed": 0, "weights_covered": 0, "parity_bug_caught": 0, "optimization_correct": 0, "total": len(scenarios)}
        covered: set[str] = set()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for path in scenarios:
                case = json.loads(path.read_text())
                name = case["name"]
                inspection = tmp / f"{name}.json"
                run_script("inspect_model.py", FIXTURES / case["fixture"], "--output", inspection)
                data = json.loads(inspection.read_text())

                # 1. Routing: correct family AND runbook.
                self.assertEqual(data["recommended_family"], case["expected_family"], name)
                self.assertEqual(data["recommended_runbook"], case["expected_runbook"], name)
                score["routed"] += 1
                covered.add(data["recommended_family"])

                # 2. Weight-key coverage: the expected source keys are all present.
                keys = {t["key"] for t in data["tensors"]}
                missing = [k for k in case["expected_weight_keys"] if k not in keys]
                self.assertFalse(missing, f"{name}: missing weight keys {missing}")
                score["weights_covered"] += 1

                # 3. Staged parity: a seeded bug MUST be caught (compare_tensors fails).
                bug = case["seeded_parity_bug"]
                src = np.random.default_rng(0).standard_normal(tuple(bug["shape"])).astype(np.float32)
                corrupt = src.copy()
                corrupt.flat[0] += np.float32(bug["delta"])
                s_path, t_path = tmp / f"{name}_src.npz", tmp / f"{name}_bad.npz"
                np.savez(s_path, w=src)
                np.savez(t_path, w=corrupt)
                run_script("compare_tensors.py", s_path, t_path, "--output", tmp / f"{name}_cmp.json", expected=1)
                score["parity_bug_caught"] += 1

                # 4. Optimization inclusion/exclusion for this family.
                rec = tmp / f"{name}_rec.json"
                recommendation_args: list[object] = [inspection, "--output", rec, "--limit", 50]
                if "target_profile" in case:
                    profile = tmp / f"{name}_target_profile.json"
                    profile.write_text(json.dumps(case["target_profile"]), encoding="utf-8")
                    recommendation_args.extend(["--target-profile", profile])
                run_script("recommend_optimizations.py", *recommendation_args)
                report = json.loads(rec.read_text())
                surfaced = {m["id"] for m in report["ready_candidates"] + report["research_candidates"]}
                self.assertIn(case["optimization_must_include"], surfaced, f"{name}: missing must-include")
                self.assertNotIn(case["optimization_must_exclude"], surfaced, f"{name}: leaked must-exclude")
                score["optimization_correct"] += 1

        for key in ("routed", "weights_covered", "parity_bug_caught", "optimization_correct"):
            self.assertEqual(score[key], score["total"], f"{key}: {score[key]}/{score['total']}")

        uncovered = sorted(ALL_FAMILIES - covered)
        print(f"\nScenario scorecard: {score}")
        print(f"Covered families ({len(covered)}/{len(ALL_FAMILIES)}): {sorted(covered)}")
        print(f"NOT yet covered by a golden scenario ({len(uncovered)}): {uncovered}")
        # Guard the covered set so adding a routing fixture without a scenario, or
        # silently losing coverage, fails loudly.
        self.assertEqual(covered, EXPECTED_COVERED)


if __name__ == "__main__":
    unittest.main()
