"""Contracts for the deterministic effective-claim catalog."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_tooling import SKILL, run_script

SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_claim_catalog  # noqa: E402
from _common import SkillError  # noqa: E402


def source_registry(*source_ids: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sources": [{"id": source_id} for source_id in source_ids],
    }


def classified_source(
    source_id: str,
    *,
    support_scope: str = "third_party_pinned",
    claim_types: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": source_id,
        "review_depth": "synthesized",
        "snapshot": "a" * 40,
        "evidence_class": "primary_source_code",
        "support_scope": support_scope,
        "claim_types": claim_types if claim_types is not None else ["performance"],
    }


def source_reported_method(*source_ids: str) -> dict[str, object]:
    return {
        "id": "reported-method",
        "status": "proven-mlx-port",
        "evidence_refs": {"repositories": list(source_ids)},
        "target_constraints": {
            "required_profile_fields": ["hardware.chip", "workloads"],
            "source_ids": list(source_ids),
        },
        "improvement_band": {
            "provenance": "source_reported",
            "claim_status": "eligible-with-profile",
            "range": "1.0x-2.0x",
            "metric": "throughput",
            "evidence_lineage_ids": ["reported-lineage"],
        },
    }


def local_method(
    *,
    receipts: list[str] | None = None,
    lineages: list[str] | None = None,
    source_ids: list[str] | None = None,
    target_constraints: bool = True,
) -> dict[str, object]:
    method: dict[str, object] = {
        "id": "local-method",
        "status": "proven-mlx-port",
        "evidence_refs": {"repositories": ["source-a"]},
        "improvement_band": {
            "provenance": "local_reproduced",
            "claim_status": "eligible",
            "range": "1.2x-1.4x",
            "metric": "decode-tokens-per-sec",
            "receipts": receipts if receipts is not None else ["baseline.json", "candidate.json"],
            "evidence_lineage_ids": lineages if lineages is not None else ["lineage-a"],
            "basis": "Controlled fixture.",
            "applies_when": "The canonical target and workload match.",
        },
    }
    if target_constraints:
        method["target_constraints"] = {
            "required_profile_fields": ["hardware.chip", "software.mlx", "workloads"],
            "source_ids": source_ids if source_ids is not None else ["source-a"],
        }
    return method


def experiment_fingerprint(
    *,
    baseline_receipt: str,
    enabled_methods: list[str],
    tag: str = "default",
    receipt_sha256: str = "a" * 64,
    quality_digest: str = "e" * 64,
    baseline_sha256: str = "b" * 64,
) -> dict[str, object]:
    target_descriptor = {
        "hardware": {"chip": "Apple Test", "fixture_tag": tag},
        "software": {"mlx": "0.30.4", "mlx_lm": "0.31.3"},
    }
    workload_descriptor = {
        "id": "fixture-workload",
        "artifacts": [{"role": "prompt", "path": "inputs/prompt.txt", "sha256": "1" * 64}],
        "parameters": {"max_tokens": 64, "seed": 7},
    }
    invariant = {"primary_metric": "generation_tps", "fixture_tag": tag}
    variant = {"enabled_methods": enabled_methods, "fixture_tag": tag}
    metrics = {
        "prompt_tokens": 32,
        "prompt_tps": 100.0,
        "generation_tokens": 64,
        "generation_tps": 60.0,
        "peak_memory_gb": 1.0,
        "ttft_proxy_s": 0.32,
    }
    aggregate = {
        metric: {"median": value, "min": value, "max": value}
        for metric, value in metrics.items()
    }
    canonical_hash = lambda value: hashlib.sha256(  # noqa: E731 - compact fixture builder
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "candidate_receipt_sha256": receipt_sha256,
        "models": {
            "target": {
                "id": "example/model",
                "revision": "a" * 40,
                "lineage_id": "fixture-lineage",
                "source_id": "example/source",
                "source_revision": "b" * 40,
            }
        },
        "target": {
            "descriptor": target_descriptor,
            "sha256": canonical_hash(target_descriptor),
        },
        "workload": {
            **workload_descriptor,
            "sha256": canonical_hash(workload_descriptor),
        },
        "experiment": {
            "invariant": invariant,
            "invariant_sha256": canonical_hash(invariant),
            "variant": variant,
            "variant_sha256": canonical_hash(variant),
        },
        "primary_metric": "generation_tps",
        "candidate_baseline_binding": {
            "role": "candidate",
            "baseline_receipt": baseline_receipt,
            "baseline_sha256": baseline_sha256,
        },
        "enabled_methods": enabled_methods,
        "aggregate": aggregate,
        "measured_runs": [{
            "run": 1,
            "metrics": metrics,
            "raw_output": {
                "path": f"raw/{receipt_sha256[:12]}.txt",
                "sha256": receipt_sha256,
                "size_bytes": 128,
                "truncated": False,
            },
        }],
        "quality": {
            "status": "pass",
            "artifact": {
                "path": f"quality/{tag}.json",
                "sha256": "d" * 64,
                "size_bytes": 256,
            },
            "result_sha256": "d" * 64,
            "contract_identity": {
                "validator": {"id": "mlx-benchmark-exact-output-parity", "version": 1},
                "metric": {
                    "id": "exact-output-parity",
                    "reference_sha256": quality_digest,
                    "candidate_sha256": quality_digest,
                    "reference_size_bytes": 64,
                    "candidate_size_bytes": 64,
                    "exact_match": True,
                },
                "provenance_mode": "controlled-exact-output-v1",
            },
        },
    }
    return {
        "schema_version": 2,
        "sha256": canonical_hash(payload),
        "payload": payload,
    }


def assessment(
    receipt: str,
    *,
    role: str = "candidate",
    baseline_receipt: str = "baseline.json",
    ratio: float = 1.25,
    classification: str | None = None,
    enabled_methods: list[str] | None = None,
    fingerprint_tag: str = "default",
    receipt_sha256: str = "a" * 64,
    quality_digest: str = "e" * 64,
    baseline_sha256: str = "b" * 64,
) -> dict[str, object]:
    is_candidate = role == "candidate"
    resolved_classification = classification or ("promotion_ready" if is_candidate else "performance_observation")
    resolved_methods = enabled_methods if enabled_methods is not None else (["local-method"] if is_candidate else [])
    gates = {gate: True for gate in generate_claim_catalog.REQUIRED_BENCHMARK_GATES}
    if not is_candidate:
        gates["baseline_compatible"] = False
    return {
        "receipt": receipt,
        "label": Path(receipt).stem,
        "receipt_sha256": receipt_sha256,
        "schema_version": 2,
        "classification": resolved_classification,
        "promotion_ready": resolved_classification == "promotion_ready",
        "experiment_fingerprint": (
            experiment_fingerprint(
                baseline_receipt=baseline_receipt,
                enabled_methods=resolved_methods,
                tag=fingerprint_tag,
                receipt_sha256=receipt_sha256,
                quality_digest=quality_digest,
                baseline_sha256=baseline_sha256,
            )
            if resolved_classification == "promotion_ready"
            else None
        ),
        "enabled_methods": resolved_methods,
        "reasons": [] if is_candidate else ["baseline-role-not-promotable"],
        "gates": gates,
        "baseline": ({
            "label": Path(baseline_receipt).stem,
            "file": baseline_receipt,
            "expected_sha256": baseline_sha256,
            "actual_sha256": baseline_sha256,
            "compatible": True,
        } if is_candidate else None),
        "recomputed_median_ratios": ({
            "decode_tps": ratio,
            "prefill_tps": 1.1,
            "ttft_proxy_inverse": 1.05,
            "peak_memory_inverse": 1.0,
        } if is_candidate else None),
        "primary_metric": "generation_tps",
    }


def raw_ratio_receipt(*, candidate: bool) -> dict[str, object]:
    metrics = {
        "prompt_tokens": 32,
        "prompt_tps": 100.0,
        "generation_tokens": 64,
        "generation_tps": 48.0,
        "peak_memory_gb": 1.0,
        "ttft_proxy_s": 0.336,
    }
    if candidate:
        metrics.update({
            "prompt_tps": 110.0,
            "generation_tps": 60.0,
            "ttft_proxy_s": 0.32,
        })
    return {"runs": [{"run": 1, **metrics}]}


class ClaimCatalogContractTests(unittest.TestCase):
    def test_committed_catalog_has_one_conservative_record_per_improvement_band(self) -> None:
        run_script("generate_claim_catalog.py", "--check")
        guidance = json.loads((SKILL / "assets" / "optimization_guidance.yaml").read_text(encoding="utf-8"))
        catalog = json.loads((SKILL / "assets" / "effective_claims.json").read_text(encoding="utf-8"))
        method_ids = sorted(method["id"] for method in guidance["methods"] if "improvement_band" in method)

        self.assertEqual(catalog["claim_count"], len(method_ids))
        self.assertEqual([claim["method_id"] for claim in catalog["claims"]], method_ids)
        self.assertTrue(all("observed_range" in claim for claim in catalog["claims"]))
        self.assertTrue(all("effective_range" in claim for claim in catalog["claims"]))
        promoted = [
            claim for claim in catalog["claims"]
            if claim["effective_range"] is not None
        ]
        self.assertEqual([claim["method_id"] for claim in promoted], ["bf16-weight-cast"])
        self.assertEqual(promoted[0]["effective_range"], "1.0x-1.8122x")
        self.assertEqual(promoted[0]["promotion_state"], "local-promotion")
        self.assertEqual(promoted[0]["withheld_reasons"], [])

        by_id = {claim["method_id"]: claim for claim in catalog["claims"]}
        for method_id, observed_range in {
            "continuous-batching-serving": "1.0x-4.3x",
            "multimodal-content-prefix-cache": "1.0x-28.0x",
            "qwen3-tts-batch-generation": "1.0x-5.45x",
        }.items():
            with self.subTest(method_id=method_id):
                source_observation = by_id[method_id]
                self.assertEqual(source_observation["observed_range"], observed_range)
                self.assertIsNone(source_observation["profile_eligible_range"])
                self.assertIsNone(source_observation["effective_range"])
                self.assertEqual(source_observation["promotion_state"], "withheld")
                self.assertEqual(source_observation["provenance"], "profile_required")
                self.assertEqual(source_observation["claim_status"], "held")
                self.assertIn("profile-required-provenance", source_observation["withheld_reasons"])
                self.assertIn("claim-status-held", source_observation["withheld_reasons"])

        historical = by_id["native-low-bit-weight-quantization"]
        self.assertEqual(historical["observed_range"], "1.0x-2.4x")
        self.assertIsNone(historical["profile_eligible_range"])
        self.assertIn("performance-observation-not-promotable", historical["withheld_reasons"])

        kernel = by_id["spatial-grid-sample-kernel"]
        self.assertEqual(kernel["observed_range"], "1.0x-10.0x")
        self.assertIsNone(kernel["profile_eligible_range"])
        self.assertIsNone(kernel["effective_range"])

    def test_local_range_is_effective_only_when_every_receipt_and_canonical_gate_passes(self) -> None:
        guidance = {"schema_version": 1, "methods": [local_method()]}
        assessments = {
            "schema_version": 1,
            "assessments": [
                assessment("baseline.json", role="baseline"),
                assessment("candidate.json", baseline_receipt="baseline.json", ratio=1.25),
            ],
        }
        catalog = generate_claim_catalog.build_claim_catalog(
            guidance,
            source_registry("source-a"),
            assessments,
            assessments_available=True,
        )
        claim = catalog["claims"][0]
        self.assertEqual(claim["observed_range"], "1.2x-1.4x")
        self.assertEqual(claim["effective_range"], "1.0x-1.25x")
        self.assertEqual(claim["promotion_state"], "local-promotion")
        self.assertEqual(claim["withheld_reasons"], [])
        self.assertEqual(
            claim["experiment_fingerprint"],
            assessments["assessments"][1]["experiment_fingerprint"],
        )

        one_failed = copy.deepcopy(assessments)
        one_failed["assessments"][1]["classification"] = "performance_observation"
        one_failed["assessments"][1]["promotion_ready"] = False
        held = generate_claim_catalog.build_claim_catalog(
            guidance,
            source_registry("source-a"),
            one_failed,
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(held["effective_range"])
        self.assertIn("receipt-not-promotion-ready:candidate.json", held["withheld_reasons"])

        wrong_link = copy.deepcopy(assessments)
        wrong_link["assessments"][1]["enabled_methods"] = ["other-method"]
        wrong_link["assessments"][1]["experiment_fingerprint"] = experiment_fingerprint(
            baseline_receipt="baseline.json",
            enabled_methods=["other-method"],
        )
        held = generate_claim_catalog.build_claim_catalog(
            guidance,
            source_registry("source-a"),
            wrong_link,
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(held["effective_range"])
        self.assertIn("receipt-method-link-missing:candidate.json", held["withheld_reasons"])

        forged = copy.deepcopy(assessments)
        forged["assessments"][1]["promotion_ready"] = False
        forged["assessments"][1]["reasons"] = ["forged-ready-state"]
        forged["assessments"][1]["gates"]["quality_valid"] = False
        held = generate_claim_catalog.build_claim_catalog(
            guidance,
            source_registry("source-a"),
            forged,
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(held["effective_range"])
        self.assertIn("receipt-promotion-flag-false:candidate.json", held["withheld_reasons"])
        self.assertIn("receipt-has-assessment-reasons:candidate.json", held["withheld_reasons"])
        self.assertIn("receipt-required-gates-failed:candidate.json:quality_valid", held["withheld_reasons"])

        invalid_digest = copy.deepcopy(assessments)
        invalid_digest["assessments"][1]["receipt_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(SkillError, "invalid receipt_sha256"):
            generate_claim_catalog.build_claim_catalog(
                guidance,
                source_registry("source-a"),
                invalid_digest,
                assessments_available=True,
            )

        copied_fingerprint = copy.deepcopy(assessments)
        copied_fingerprint["assessments"][1]["receipt_sha256"] = "e" * 64
        with self.assertRaisesRegex(SkillError, "experiment_fingerprint receipt_sha256 mismatch"):
            generate_claim_catalog.build_claim_catalog(
                guidance,
                source_registry("source-a"),
                copied_fingerprint,
                assessments_available=True,
            )

        unrelated = copy.deepcopy(assessments)
        unrelated["assessments"].append(assessment("unrelated.json", role="baseline"))
        unrelated_guidance = {
            "schema_version": 1,
            "methods": [local_method(receipts=["baseline.json", "candidate.json", "unrelated.json"])],
        }
        held = generate_claim_catalog.build_claim_catalog(
            unrelated_guidance,
            source_registry("source-a"),
            unrelated,
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(held["effective_range"])
        self.assertIn("unrelated-receipt-observation:unrelated.json", held["withheld_reasons"])

    def test_tampered_recomputed_median_ratios_fail_with_intact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_path.write_text(
                json.dumps(raw_ratio_receipt(candidate=False)) + "\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(raw_ratio_receipt(candidate=True)) + "\n",
                encoding="utf-8",
            )
            baseline_sha256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            original = {
                "schema_version": 1,
                "assessments": [
                    assessment(
                        "baseline.json",
                        role="baseline",
                        receipt_sha256=baseline_sha256,
                    ),
                    assessment(
                        "candidate.json",
                        receipt_sha256=candidate_sha256,
                        baseline_sha256=baseline_sha256,
                    ),
                ],
            }
            tampered = copy.deepcopy(original)
            intact_fingerprint = copy.deepcopy(
                tampered["assessments"][1]["experiment_fingerprint"]
            )
            tampered["assessments"][1]["recomputed_median_ratios"]["decode_tps"] = 99.0
            self.assertEqual(
                tampered["assessments"][1]["experiment_fingerprint"],
                intact_fingerprint,
            )

            with self.assertRaisesRegex(
                SkillError,
                "recomputed_median_ratios do not match receipt raw runs",
            ):
                generate_claim_catalog.build_claim_catalog(
                    {"schema_version": 1, "methods": [local_method()]},
                    source_registry("source-a"),
                    tampered,
                    assessments_available=True,
                    receipt_root=root,
                )

    def test_local_promotion_withholds_unsupported_metric_instead_of_trusting_stored_range(self) -> None:
        method = local_method()
        method["improvement_band"]["metric"] = "custom-score"
        assessments = {
            "schema_version": 1,
            "assessments": [
                assessment("baseline.json", role="baseline"),
                assessment("candidate.json"),
            ],
        }
        claim = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [method]},
            source_registry("source-a"),
            assessments,
            assessments_available=True,
        )["claims"][0]
        self.assertEqual(claim["observed_range"], "1.2x-1.4x")
        self.assertIsNone(claim["effective_range"])
        self.assertIn("unsupported-local-metric:custom-score", claim["withheld_reasons"])

    def test_local_candidate_without_improvement_is_not_promoted_as_a_speedup(self) -> None:
        assessments = {
            "schema_version": 1,
            "assessments": [
                assessment("baseline.json", role="baseline"),
                assessment("candidate.json", ratio=0.95),
            ],
        }
        claim = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [local_method()]},
            source_registry("source-a"),
            assessments,
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(claim["effective_range"])
        self.assertEqual(claim["promotion_state"], "withheld")
        self.assertIn("candidate-no-improvement", claim["withheld_reasons"])

    def test_source_reported_claims_remain_held_even_with_classified_pinned_evidence(self) -> None:
        method = source_reported_method("implementation", "paper")
        valid_sources = {
            "schema_version": 1,
            "sources": [
                classified_source("implementation", claim_types=["mlx_implementation"]),
                classified_source("paper", support_scope="paper_only", claim_types=["performance"]),
            ],
        }
        valid = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [method]},
            valid_sources,
            {"schema_version": 1, "assessments": []},
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(valid["profile_eligible_range"])
        self.assertEqual(valid["promotion_state"], "withheld")
        self.assertIn(
            "source-reported-range-requires-local-reproduction",
            valid["withheld_reasons"],
        )

        unclassified = copy.deepcopy(valid_sources)
        del unclassified["sources"][0]["review_depth"]
        del unclassified["sources"][0]["snapshot"]
        del unclassified["sources"][0]["evidence_class"]
        del unclassified["sources"][0]["claim_types"]
        held = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [method]},
            unclassified,
            {"schema_version": 1, "assessments": []},
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(held["profile_eligible_range"])
        self.assertIn("source-claim-not-synthesized:implementation", held["withheld_reasons"])
        self.assertIn("source-claim-not-snapshot-pinned:implementation", held["withheld_reasons"])
        self.assertIn("source-claim-missing-evidence-class:implementation", held["withheld_reasons"])
        self.assertIn("source-claim-missing-claim-types:implementation", held["withheld_reasons"])

        context_only = copy.deepcopy(valid_sources)
        context_only["sources"][0]["support_scope"] = "context_only"
        context_only["sources"][1]["claim_types"] = ["serving_semantics"]
        held = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [method]},
            context_only,
            {"schema_version": 1, "assessments": []},
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(held["profile_eligible_range"])
        self.assertIn("source-claim-invalid-scope:implementation:context_only", held["withheld_reasons"])
        self.assertIn("source-claim-missing-performance-evidence", held["withheld_reasons"])

    def test_one_regressing_candidate_withholds_the_complete_local_claim(self) -> None:
        assessments = {
            "schema_version": 1,
            "assessments": [
                assessment("baseline.json", role="baseline"),
                assessment("candidate-fast.json", ratio=1.25),
                assessment("candidate-regression.json", ratio=0.5),
            ],
        }
        method = local_method(
            receipts=["baseline.json", "candidate-fast.json", "candidate-regression.json"]
        )
        claim = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [method]},
            source_registry("source-a"),
            assessments,
            assessments_available=True,
        )["claims"][0]
        self.assertIsNone(claim["effective_range"])
        self.assertEqual(claim["promotion_state"], "withheld")
        self.assertIn("candidate-regression-present", claim["withheld_reasons"])

    def test_heterogeneous_promotion_receipts_never_collapse_to_a_maximum(self) -> None:
        assessments = {
            "schema_version": 1,
            "assessments": [
                assessment("baseline.json", role="baseline"),
                assessment("candidate-a.json", ratio=1.25, fingerprint_tag="target-a"),
                assessment("candidate-b.json", ratio=9.0, fingerprint_tag="target-b"),
            ],
        }
        method = local_method(
            receipts=["baseline.json", "candidate-a.json", "candidate-b.json"]
        )
        claim = generate_claim_catalog.build_claim_catalog(
            {"schema_version": 1, "methods": [method]},
            source_registry("source-a"),
            assessments,
            assessments_available=True,
        )["claims"][0]

        self.assertIsNone(claim["effective_range"])
        self.assertIsNone(claim["experiment_fingerprint"])
        self.assertEqual(claim["promotion_state"], "withheld")
        self.assertIn(
            "heterogeneous-promotion-candidate-fingerprints",
            claim["withheld_reasons"],
        )
        self.assertNotEqual(claim["effective_range"], "1.0x-9.0x")

    def test_compatible_repetitions_use_the_conservative_full_fingerprint(self) -> None:
        candidate_a = assessment(
            "candidate-a.json",
            ratio=1.25,
            receipt_sha256="c" * 64,
        )
        candidate_b = assessment(
            "candidate-b.json",
            ratio=1.10,
            receipt_sha256="d" * 64,
        )
        method = local_method(
            receipts=["baseline.json", "candidate-a.json", "candidate-b.json"]
        )

        def build(rows: list[dict[str, object]]) -> dict[str, object]:
            return generate_claim_catalog.build_claim_catalog(
                {"schema_version": 1, "methods": [method]},
                source_registry("source-a"),
                {"schema_version": 1, "assessments": rows},
                assessments_available=True,
            )["claims"][0]

        forward = build([assessment("baseline.json", role="baseline"), candidate_a, candidate_b])
        reverse = build([assessment("baseline.json", role="baseline"), candidate_b, candidate_a])
        for claim in (forward, reverse):
            self.assertEqual(claim["effective_range"], "1.0x-1.1x")
            self.assertEqual(claim["promotion_state"], "local-promotion")
            self.assertEqual(
                claim["experiment_fingerprint"],
                candidate_b["experiment_fingerprint"],
            )
            self.assertNotEqual(
                candidate_a["experiment_fingerprint"]["sha256"],
                candidate_b["experiment_fingerprint"]["sha256"],
            )

    def test_repetitions_with_different_quality_or_baseline_identity_are_heterogeneous(self) -> None:
        for field, value in (
            ("quality_digest", "f" * 64),
            ("baseline_sha256", "9" * 64),
        ):
            with self.subTest(field=field):
                assessments = {
                    "schema_version": 1,
                    "assessments": [
                        assessment("baseline.json", role="baseline"),
                        assessment("candidate-a.json", receipt_sha256="c" * 64),
                        assessment(
                            "candidate-b.json",
                            receipt_sha256="d" * 64,
                            **{field: value},
                        ),
                    ],
                }
                claim = generate_claim_catalog.build_claim_catalog(
                    {
                        "schema_version": 1,
                        "methods": [local_method(receipts=[
                            "baseline.json",
                            "candidate-a.json",
                            "candidate-b.json",
                        ])],
                    },
                    source_registry("source-a"),
                    assessments,
                    assessments_available=True,
                )["claims"][0]
                self.assertIsNone(claim["effective_range"])
                self.assertIn(
                    "heterogeneous-promotion-candidate-fingerprints",
                    claim["withheld_reasons"],
                )

    def test_missing_sidecar_or_constraints_preserves_observation_but_withholds_promotion(self) -> None:
        guidance = {"schema_version": 1, "methods": [local_method(target_constraints=False)]}
        catalog = generate_claim_catalog.build_claim_catalog(
            guidance,
            source_registry("source-a"),
            {"schema_version": 1, "assessments": []},
            assessments_available=False,
        )
        claim = catalog["claims"][0]
        self.assertEqual(catalog["receipt_assessment_state"], "unavailable")
        self.assertEqual(claim["observed_range"], "1.2x-1.4x")
        self.assertIsNone(claim["effective_range"])
        self.assertIn("receipt-assessments-unavailable", claim["withheld_reasons"])
        self.assertIn("missing-target-constraints", claim["withheld_reasons"])

    def test_unknown_sources_and_reused_active_lineages_fail_loudly(self) -> None:
        unknown_source = local_method(source_ids=["missing-source"])
        with self.assertRaisesRegex(SkillError, "missing evidence source missing-source"):
            generate_claim_catalog.build_claim_catalog(
                {"schema_version": 1, "methods": [unknown_source]},
                source_registry("source-a"),
                {"schema_version": 1, "assessments": []},
                assessments_available=True,
            )

        first = local_method()
        second = copy.deepcopy(first)
        second["id"] = "second-method"
        with self.assertRaisesRegex(SkillError, "evidence lineage lineage-a is reused"):
            generate_claim_catalog.build_claim_catalog(
                {"schema_version": 1, "methods": [first, second]},
                source_registry("source-a"),
                {"schema_version": 1, "assessments": []},
                assessments_available=True,
            )

    def test_generator_is_deterministic_and_check_mode_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            guidance_path = tmp / "guidance.json"
            sources_path = tmp / "sources.json"
            assessments_path = tmp / "assessments.json"
            output_path = tmp / "effective.json"
            guidance_path.write_text(
                json.dumps({"schema_version": 1, "methods": [local_method()]}),
                encoding="utf-8",
            )
            sources_path.write_text(json.dumps(source_registry("source-a")), encoding="utf-8")
            baseline_path = tmp / "baseline.json"
            candidate_path = tmp / "candidate.json"
            baseline_path.write_text(
                json.dumps(raw_ratio_receipt(candidate=False)) + "\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(raw_ratio_receipt(candidate=True)) + "\n",
                encoding="utf-8",
            )
            baseline_sha256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            assessments_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "assessments": [
                        assessment(
                            "baseline.json",
                            role="baseline",
                            receipt_sha256=baseline_sha256,
                        ),
                        assessment("candidate.json", receipt_sha256=candidate_sha256),
                    ],
                }),
                encoding="utf-8",
            )

            args = [
                "--guidance", guidance_path,
                "--sources", sources_path,
                "--assessments", assessments_path,
                "--output", output_path,
            ]
            run_script("generate_claim_catalog.py", *args)
            first = output_path.read_bytes()
            run_script("generate_claim_catalog.py", *args)
            self.assertEqual(output_path.read_bytes(), first)
            run_script("generate_claim_catalog.py", *args, "--check")

            output_path.write_text("{}\n", encoding="utf-8")
            result = run_script("generate_claim_catalog.py", *args, "--check", expected=1)
            self.assertIn("claim catalog drift", result.stderr)

            candidate_path.write_text('{"fixture":"candidate-tampered"}\n', encoding="utf-8")
            result = run_script("generate_claim_catalog.py", *args, expected=2)
            self.assertIn("receipt_sha256 does not match the receipt artifact", result.stderr)


if __name__ == "__main__":
    unittest.main()
