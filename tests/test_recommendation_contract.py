"""Safety and evidence contracts for model-specific optimization advice."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_tooling import FIXTURES, SKILL, run_script, trusted_inspection_fixture

SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import COMPOUND_PROMOTION_REQUIRED_GATES, SkillError, compose_stack_band  # noqa: E402
from generate_claim_catalog import REQUIRED_BENCHMARK_GATES  # noqa: E402
from make_port_plan import load_recommendation_report, recommendation_lines  # noqa: E402
from recommend_optimizations import (  # noqa: E402
    MAX_KNOWLEDGE_GRAPH_BYTES,
    load_receipt_assessments,
    target_claim_holds,
    trusted_inspection_sha256,
    validate_trusted_inspection,
)

OPT_IN_PROMPT = "This is an experimental approach. Do you want to try it?"


def stack_experiment_fingerprint(
    tag: str = "default",
    *,
    receipt_sha256: str = "a" * 64,
    quality_digest: str = "e" * 64,
) -> dict[str, object]:
    canonical_hash = lambda value: hashlib.sha256(  # noqa: E731 - compact fixture builder
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    target_descriptor = {
        "hardware": {"chip": "Apple Test", "model": "MacTest,1", "fixture_tag": tag},
        "software": {"mlx": "0.30.4", "mlx_lm": "0.31.3"},
    }
    workload_descriptor = {
        "id": "stack-fixture",
        "artifacts": [{"role": "prompt", "path": "prompt.txt", "sha256": "1" * 64}],
        "parameters": {"max_tokens": 64},
    }
    invariant = {"primary_metric": "generation_tps"}
    variant = {"enabled_methods": ["a", "b"]}
    metrics = {
        "prompt_tokens": 32,
        "prompt_tps": 100.0,
        "generation_tokens": 64,
        "generation_tps": 60.0,
        "peak_memory_gb": 1.0,
        "ttft_proxy_s": 0.32,
    }
    payload = {
        "candidate_receipt_sha256": receipt_sha256,
        "models": {
            "target": {
                "id": "example/model",
                "revision": "a" * 40,
                "lineage_id": "stack-fixture-lineage",
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
            "baseline_receipt": "baseline.json",
            "baseline_sha256": "b" * 64,
        },
        "enabled_methods": ["a", "b"],
        "aggregate": {
            metric: {"median": value, "min": value, "max": value}
            for metric, value in metrics.items()
        },
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


def exact_target_profile(fingerprint: dict[str, object]) -> dict[str, object]:
    payload = fingerprint["payload"]
    assert isinstance(payload, dict)
    target = payload["target"]
    assert isinstance(target, dict) and isinstance(target["descriptor"], dict)
    descriptor = target["descriptor"]
    return {
        "schema_version": 1,
        "hardware": copy.deepcopy(descriptor["hardware"]),
        "software": copy.deepcopy(descriptor["software"]),
        "capabilities": [],
        "workloads": ["server"],
        "models": copy.deepcopy(payload["models"]),
        "target": copy.deepcopy(target),
        "workload": copy.deepcopy(payload["workload"]),
        "experiment_fingerprint": copy.deepcopy(fingerprint),
    }


def write_inspection(
    path: Path,
    families: list[str],
    *,
    model_type: str = "synthetic",
    traits: list[str] | None = None,
    blockers: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(trusted_inspection_fixture(
            families,
            model_type=model_type,
            traits=traits,
            blockers=blockers,
        )),
        encoding="utf-8",
    )


def run_recommender(
    tmp: Path,
    families: list[str],
    *,
    model_type: str = "synthetic",
    traits: list[str] | None = None,
    target_profile: dict[str, object] | None = None,
    blockers: list[str] | None = None,
    knowledge_graph: Path | None = None,
) -> tuple[dict[str, object], str]:
    inspection = tmp / f"inspection-{len(list(tmp.glob('inspection-*.json')))}.json"
    output = tmp / f"recommendations-{len(list(tmp.glob('recommendations-*.json')))}.json"
    markdown = tmp / f"recommendations-{len(list(tmp.glob('recommendations-*.md')))}.md"
    write_inspection(
        inspection,
        families,
        model_type=model_type,
        traits=traits,
        blockers=blockers,
    )
    args: list[object] = [inspection, "--output", output, "--markdown", markdown, "--limit", 100]
    if target_profile is not None:
        profile_path = tmp / f"target-profile-{len(list(tmp.glob('target-profile-*.json')))}.json"
        profile_path.write_text(json.dumps(target_profile), encoding="utf-8")
        args.extend(["--target-profile", profile_path])
    if blockers:
        args.extend(["--family", families[0]])
    if knowledge_graph is not None:
        args.extend(["--knowledge-graph", knowledge_graph])
    run_script("recommend_optimizations.py", *args)
    return json.loads(output.read_text(encoding="utf-8")), markdown.read_text(encoding="utf-8")


def candidate_ids(report: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for key in ("ready_candidates", "research_candidates", "held_candidates", "notable_exclusions"):
        for candidate in report.get(key, []):
            ids.add(str(candidate["id"]))
    return ids


def candidate_by_id(report: dict[str, object], method_id: str) -> dict[str, object] | None:
    for key in ("ready_candidates", "research_candidates", "held_candidates", "notable_exclusions"):
        for candidate in report.get(key, []):
            if candidate.get("id") == method_id:
                return candidate
    return None


class RecommendationContractTests(unittest.TestCase):
    def test_graph_advisory_surfaces_provenance_without_becoming_a_sixth_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            graph_path = tmp / "knowledge_graph.json"
            nodes = [
                {
                    "id": "candidate:paper:fixture-claim",
                    "kind": "source_candidate",
                    "label": "Fixture claims 999x speedup",
                    "locator": "https://example.com/unreviewed-paper",
                    "read_state": "unread_candidate",
                    "review_status": "candidate-unreviewed",
                },
                {
                    "id": "source:fixture-lineage",
                    "kind": "source",
                    "label": "Known fixture source",
                    "locator": "https://example.com/known-source",
                    "read_state": "indexed_only",
                    "review_depth": "indexed",
                },
                {
                    "id": "source:fixture-evidence",
                    "kind": "source",
                    "label": "Fixture evidence",
                    "locator": "https://example.com/evidence",
                    "read_state": "already_read",
                    "review_depth": "screened",
                },
                {
                    "id": "approach:fixture-review-only",
                    "kind": "approach",
                    "label": "Fixture approach",
                    "status": "research-candidate",
                    "decision_state": "method_registry",
                    "applies_to": ["dense-decoder-transformer"],
                },
                {
                    "id": "outcome:fixture-review-only",
                    "kind": "model_outcome",
                    "label": "Fixture outcome",
                    "status": "source_backed_working",
                    "decision_state": "outcome_registry",
                    "families": ["dense-decoder-transformer"],
                },
            ]
            edges = [
                {
                    "source": "candidate:paper:fixture-claim",
                    "target": "approach:fixture-review-only",
                    "relation": "candidate_relevant_to",
                    "score": 999,
                },
                {
                    "source": "source:fixture-evidence",
                    "target": "approach:fixture-review-only",
                    "relation": "evidence_for",
                },
                {
                    "source": "source:fixture-evidence",
                    "target": "outcome:fixture-review-only",
                    "relation": "evidence_for_outcome",
                },
                {
                    "source": "candidate:paper:fixture-claim",
                    "target": "source:fixture-lineage",
                    "relation": "candidate_version_of",
                },
            ]
            graph_path.write_text(json.dumps({
                "schema_version": 1,
                "generated_at": "2026-07-10T00:00:00+00:00",
                "run_id": "fixture-graph",
                "policy": {
                    "review_only": True,
                    "auto_promote_sources": False,
                    "auto_modify_recommendations": False,
                },
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": nodes,
                "edges": edges,
            }), encoding="utf-8")

            report, markdown = run_recommender(
                tmp,
                ["dense-decoder-transformer"],
                knowledge_graph=graph_path,
            )
            advisory = report["research_advisory"]
            self.assertEqual(advisory["status"], "available")
            self.assertTrue(advisory["distinct_from_advisor_buckets"])
            self.assertEqual(advisory["selected_count"], 4)
            self.assertEqual(
                {item["relation"] for item in advisory["items"]},
                {
                    "candidate_relevant_to",
                    "evidence_for",
                    "evidence_for_outcome",
                    "candidate_version_of",
                },
            )
            for item in advisory["items"]:
                self.assertFalse(item["execution_allowed"])
                self.assertFalse(item["numeric_claims_included"])
                self.assertNotIn("expected_effect", item)
                self.assertNotIn("improvement_band", item)
                self.assertIn("source_node_id", item["graph_provenance"])
                self.assertIn("target_node_id", item["graph_provenance"])
                self.assertNotIn("source_label", item)
                self.assertNotIn("target_label", item)
            self.assertNotIn("999x speedup", json.dumps(advisory))
            self.assertNotIn("score", json.dumps(advisory))
            self.assertNotIn("unreviewed-research-signal", report["advisor_buckets"])
            self.assertIn("Unreviewed research signals (experimental/review queue)", markdown)
            self.assertIn("never supply numeric claims", markdown)

    def test_graph_advisory_fails_closed_without_crashing_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            malformed = tmp / "malformed-graph.json"
            malformed.write_text('{"schema_version": 1, "nodes": [', encoding="utf-8")
            report, markdown = run_recommender(
                tmp,
                ["dense-decoder-transformer"],
                knowledge_graph=malformed,
            )
            self.assertEqual(report["research_advisory"]["status"], "unavailable")
            self.assertIn("not valid JSON", report["research_advisory"]["reason"])
            self.assertTrue(report["ready_candidates"])
            self.assertIn("Graph unavailable", markdown)

            oversized = tmp / "oversized-graph.json"
            oversized.write_bytes(b"{" + b" " * MAX_KNOWLEDGE_GRAPH_BYTES + b"}")
            oversized_report, _ = run_recommender(
                tmp,
                ["dense-decoder-transformer"],
                knowledge_graph=oversized,
            )
            self.assertEqual(oversized_report["research_advisory"]["status"], "unavailable")
            self.assertIn("advisory read limit", oversized_report["research_advisory"]["reason"])

    def test_family_override_cannot_bypass_missing_inspection_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            inspection.write_text(
                json.dumps({"schema_version": 1, "recommendation_blockers": []}),
                encoding="utf-8",
            )
            for script, extra in (
                ("recommend_optimizations.py", []),
                ("make_port_plan.py", ["--output", tmp / "PORT_PLAN.md"]),
            ):
                with self.subTest(script=script):
                    result = run_script(
                        script,
                        inspection,
                        "--family",
                        "dense-decoder-transformer",
                        *extra,
                        expected=2,
                    )
                    self.assertIn("Untrusted inspection report", result.stderr)

    def test_port_plan_family_override_must_be_an_inspected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            inspection.write_text(
                json.dumps(trusted_inspection_fixture(["dense-decoder-transformer"])),
                encoding="utf-8",
            )
            result = run_script(
                "make_port_plan.py",
                inspection,
                "--family",
                "not-an-inspected-family",
                "--output",
                tmp / "PORT_PLAN.md",
                expected=2,
            )
            self.assertIn("not present in the trusted inspection candidates", result.stderr)

    def test_real_inspector_report_satisfies_downstream_trust_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            model = Path(raw_tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "license": "apache-2.0",
                }),
                encoding="utf-8",
            )
            (model / "LICENSE").write_text(
                "Apache License Version 2.0\n"
                "Copyright 2026 Test Author\n"
                "Licensed under the Apache License, Version 2.0; you may not use this file "
                "except in compliance with the License.\n",
                encoding="utf-8",
            )
            report = json.loads(run_script("inspect_model.py", model).stdout)

            self.assertIs(validate_trusted_inspection(report), report)
            self.assertEqual(report["artifact_identity"]["status"], "verified")
            self.assertEqual(report["license"]["status"], "acceptable-evidence")
            self.assertTrue(report["recommendation_blockers"])

    def test_actionable_port_plan_reverifies_artifact_and_rejects_coherent_route_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            model = FIXTURES / "models" / "decoder"
            inspection = tmp / "inspection.json"
            plan = tmp / "PORT_PLAN.md"
            run_script("inspect_model.py", model, "--output", inspection)

            missing = run_script(
                "make_port_plan.py",
                inspection,
                "--output",
                plan,
                expected=2,
            )
            self.assertIn("require --artifact-root", missing.stderr)

            forged = json.loads(inspection.read_text(encoding="utf-8"))
            dense = next(
                candidate
                for candidate in forged["architecture_candidates"]
                if candidate["family"] == "dense-decoder-transformer"
            )
            encoder = next(
                candidate
                for candidate in forged["architecture_candidates"]
                if candidate["family"] == "encoder-transformer"
            )
            dense_runbook = dense["runbook"]
            encoder_runbook = encoder["runbook"]
            dense["family"], dense["runbook"] = "encoder-transformer", encoder_runbook
            encoder["family"], encoder["runbook"] = "dense-decoder-transformer", dense_runbook
            forged["routing_decision"].update({
                "winner_family": "encoder-transformer",
                "winner_score": dense["score"],
                "runner_up_family": "dense-decoder-transformer",
                "runner_up_score": encoder["score"],
                "winner_margin": round(dense["score"] - encoder["score"], 2),
            })
            forged["recommended_family"] = "encoder-transformer"
            forged["recommended_runbook"] = encoder_runbook
            forged["recommended_families"] = ["encoder-transformer"]
            forged["recommended_runbooks"] = [encoder_runbook]
            inspection.write_text(json.dumps(forged), encoding="utf-8")
            self.assertIs(validate_trusted_inspection(forged), forged)

            mismatch = run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                model,
                "--output",
                plan,
                expected=2,
            )
            self.assertIn("does not match --artifact-root", mismatch.stderr)
            self.assertFalse(plan.exists())

    def test_trusted_inspection_contract_rejects_field_and_state_mutations(self) -> None:
        valid = trusted_inspection_fixture(["dense-decoder-transformer"], model_type="llama")
        self.assertIs(validate_trusted_inspection(valid), valid)

        for field in (
            "inspection_mode",
            "source",
            "artifact_identity",
            "file_summary",
            "tensor_summary",
            "source_format_summary",
            "routing_decision",
            "recommendation_blockers",
            "license",
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(valid)
                mutated.pop(field)
                with self.assertRaisesRegex(SkillError, "Untrusted inspection report"):
                    validate_trusted_inspection(mutated)

        bad_schema = copy.deepcopy(valid)
        bad_schema["schema_version"] = 2
        with self.assertRaisesRegex(SkillError, "schema_version"):
            validate_trusted_inspection(bad_schema)

        hidden_integrity_failure = copy.deepcopy(valid)
        hidden_integrity_failure["tensor_summary"]["integrity_ok"] = False
        with self.assertRaisesRegex(SkillError, "requires at least one blocker"):
            validate_trusted_inspection(hidden_integrity_failure)

        forged_identity = copy.deepcopy(valid)
        forged_identity["artifact_identity"]["fingerprint"] = "sha256:" + ("0" * 64)
        with self.assertRaisesRegex(SkillError, "artifact identity is internally inconsistent"):
            validate_trusted_inspection(forged_identity)

        hidden_incomplete_identity = copy.deepcopy(valid)
        hidden_incomplete_identity["artifact_identity"].update({
            "status": "incomplete",
            "immutable": False,
            "fingerprint": None,
            "errors": ["inventory truncated"],
        })
        with self.assertRaisesRegex(SkillError, "requires at least one blocker"):
            validate_trusted_inspection(hidden_incomplete_identity)

        hidden_license_review = copy.deepcopy(valid)
        hidden_license_review["license"].update({
            "status": "review-required",
            "requires_review": True,
            "accepted_evidence": [],
            "reasons": ["missing acceptable evidence"],
        })
        with self.assertRaisesRegex(SkillError, "requires at least one blocker"):
            validate_trusted_inspection(hidden_license_review)

        forged_routing = copy.deepcopy(valid)
        forged_routing["routing_decision"]["winner_family"] = "moe-decoder-transformer"
        with self.assertRaisesRegex(SkillError, "routing winner is not an architecture candidate"):
            validate_trusted_inspection(forged_routing)

        unpinned_remote = copy.deepcopy(valid)
        unpinned_remote["source"].update({
            "kind": "huggingface",
            "input": "example/model",
            "revision": "main",
        })
        with self.assertRaisesRegex(SkillError, "must be a pinned commit"):
            validate_trusted_inspection(unpinned_remote)

        forged_blocked_route = copy.deepcopy(valid)
        forged_blocked_route["recommendation_blockers"] = ["integrity failure"]
        with self.assertRaisesRegex(SkillError, "blocked reports must not contain recommended families"):
            validate_trusted_inspection(forged_blocked_route)

    def test_port_plan_requires_complete_recommender_contract_bound_to_exact_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection_path = tmp / "inspection.json"
            path = Path(raw_tmp) / "recommendations.json"
            run_script(
                "inspect_model.py",
                FIXTURES / "models" / "decoder",
                "--output",
                inspection_path,
            )
            run_script(
                "recommend_optimizations.py",
                inspection_path,
                "--output",
                path,
                "--limit",
                100,
            )
            inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                load_recommendation_report(
                    str(path),
                    inspection,
                    ["dense-decoder-transformer"],
                ),
                report,
            )
            rendered = "\n".join(recommendation_lines(report))
            if report["advisor_buckets"]["experimental-approach"]:
                self.assertIn(OPT_IN_PROMPT, rendered)
                self.assertIn("Execution remains held until explicit consent", rendered)

            forged = copy.deepcopy(report)
            forged["inspection_sha256"] = "0" * 64
            path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(SkillError, "not bound to this exact inspection"):
                load_recommendation_report(
                    str(path),
                    inspection,
                    ["dense-decoder-transformer"],
                )

            incomplete = copy.deepcopy(report)
            incomplete.pop("match_context")
            path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(SkillError, "missing canonical fields"):
                load_recommendation_report(
                    str(path),
                    inspection,
                    ["dense-decoder-transformer"],
                )

            if report["advisor_buckets"]["experimental-approach"]:
                unsafe_experiment = copy.deepcopy(report)
                unsafe_experiment["advisor_buckets"]["experimental-approach"][0]["execution_allowed"] = True
                path.write_text(json.dumps(unsafe_experiment), encoding="utf-8")
                with self.assertRaisesRegex(SkillError, "violates opt-in policy"):
                    load_recommendation_report(
                        str(path),
                        inspection,
                        ["dense-decoder-transformer"],
                    )

    def test_blocked_port_plan_is_remediation_only_even_with_family_override(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            plan = tmp / "PORT_PLAN.md"
            inspection.write_text(json.dumps(trusted_inspection_fixture(
                ["dense-decoder-transformer"],
                blockers=["artifact integrity validation failed"],
            )), encoding="utf-8")

            run_script(
                "make_port_plan.py",
                inspection,
                "--family",
                "dense-decoder-transformer",
                "--output",
                plan,
            )
            text = plan.read_text(encoding="utf-8")
            self.assertIn("non-actionable remediation only", text)
            self.assertIn("artifact integrity validation failed", text)
            self.assertIn("family override does not clear intake blockers", text.lower())
            self.assertNotIn("## Implementation phases", text)
            self.assertNotIn("## Optimization advice", text)
            self.assertNotIn("Proven/native optimization candidates", text)

    def test_port_plan_embeds_only_matching_canonical_recommender_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            recommendations = tmp / "recommendations.json"
            plan = tmp / "PORT_PLAN.md"
            model = FIXTURES / "models" / "decoder"
            run_script("inspect_model.py", model, "--output", inspection)
            run_script(
                "recommend_optimizations.py",
                inspection,
                "--output",
                recommendations,
                "--limit",
                1,
            )
            report = json.loads(recommendations.read_text(encoding="utf-8"))
            run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                model,
                "--recommendations",
                recommendations,
                "--output",
                plan,
            )
            text = plan.read_text(encoding="utf-8")
            surfaced = {
                str(item["id"])
                for bucket in report["advisor_buckets"].values()
                for item in bucket
            }
            self.assertTrue(surfaced)
            for method_id in surfaced - {
                str(item["id"])
                for item in report["advisor_buckets"]["rejected-do-not-use"]
            }:
                self.assertIn(f"`{method_id}`", text)
            if report["advisor_buckets"]["experimental-approach"]:
                experimental = report["advisor_buckets"]["experimental-approach"][0]
                self.assertIn(experimental["opt_in_prompt"], text)
                self.assertIn("Execution remains held until explicit consent", text)
            self.assertNotIn("cuda-graphs-decode-capture", text)
            self.assertNotIn("Proven/native optimization candidates", text)

            forged_prose = copy.deepcopy(report)
            forged_candidate = next(
                item
                for bucket in forged_prose["advisor_buckets"].values()
                for item in bucket
            )
            forged_candidate["expected_effect"] = "FORGED: 9999x throughput with no quality loss."
            recommendations.write_text(json.dumps(forged_prose), encoding="utf-8")
            result = run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                model,
                "--recommendations",
                recommendations,
                "--output",
                plan,
                expected=2,
            )
            self.assertIn("does not exactly match canonical recomputation", result.stderr)
            self.assertNotIn("9999x", plan.read_text(encoding="utf-8"))

            stale = copy.deepcopy(report)
            stale["inspection_sha256"] = "0" * 64
            recommendations.write_text(json.dumps(stale), encoding="utf-8")
            result = run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                model,
                "--recommendations",
                recommendations,
                "--output",
                plan,
                expected=2,
            )
            self.assertIn("not bound to this exact inspection", result.stderr)

            stale = copy.deepcopy(report)
            stale["families"] = ["moe-decoder-transformer"]
            recommendations.write_text(json.dumps(stale), encoding="utf-8")
            result = run_script(
                "make_port_plan.py",
                inspection,
                "--artifact-root",
                model,
                "--recommendations",
                recommendations,
                "--output",
                plan,
                expected=2,
            )
            self.assertIn("families do not match", result.stderr)

    def test_custom_assessment_sidecar_requires_real_colocated_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assessment_path = Path(tmp) / "receipt_assessments.json"
            assessment_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "assessments": [{
                        "receipt": "does-not-exist.json",
                        "label": "forged-stack",
                        "receipt_sha256": "a" * 64,
                        "classification": "promotion_ready",
                        "promotion_ready": True,
                        "enabled_methods": ["a", "b"],
                        "reasons": [],
                        "gates": {
                            gate: True for gate in COMPOUND_PROMOTION_REQUIRED_GATES
                        },
                        "primary_metric": "generation_tps",
                        "recomputed_median_ratios": {"decode_tps": 9.0},
                    }],
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SkillError, "not backed by the colocated benchmark receipts"):
                load_receipt_assessments(str(assessment_path))

    def test_compound_promotion_gates_match_claim_catalog_gates(self) -> None:
        self.assertEqual(
            set(COMPOUND_PROMOTION_REQUIRED_GATES),
            REQUIRED_BENCHMARK_GATES,
        )

    def test_duplicate_json_keys_are_detected_before_registry_loading(self) -> None:
        from validate_sources import find_duplicate_json_keys

        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "duplicate.json"
            path.write_text('{"band": {"lineage": "a", "lineage": "b"}}', encoding="utf-8")
            self.assertEqual(find_duplicate_json_keys(path), ["lineage"])

    def test_matching_is_exact_and_checks_every_recommended_family(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            codec, _ = run_recommender(tmp, ["neural-audio-codec"])
            self.assertNotIn("prompt-lookup-ngram-speculation", candidate_ids(codec))

            hybrid, _ = run_recommender(
                tmp,
                ["ssm-recurrent-hybrid", "dense-decoder-transformer"],
                model_type="falcon_h1",
                traits=["mamba2-state-space", "full-attention", "kv-cache"],
            )
            self.assertIn("fast-sdpa", candidate_ids(hybrid))
            self.assertEqual(
                hybrid["families"],
                ["ssm-recurrent-hybrid", "dense-decoder-transformer"],
            )

    def test_taxonomy_buckets_and_experimental_opt_in_are_enforced_in_both_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            report, markdown = run_recommender(Path(raw_tmp), ["dense-decoder-transformer"], model_type="llama")
            buckets = report["advisor_buckets"]
            self.assertEqual(
                set(buckets),
                {
                    "validated-locally",
                    "validated-source-theory",
                    "benchmark-required",
                    "experimental-approach",
                    "rejected-do-not-use",
                },
            )
            self.assertTrue(buckets["experimental-approach"])
            for candidate in buckets["experimental-approach"]:
                self.assertTrue(candidate["requires_user_opt_in"])
                self.assertFalse(candidate["execution_allowed"])
                self.assertEqual(candidate["opt_in_prompt"], OPT_IN_PROMPT)
            self.assertIn("## Experimental approaches", markdown)
            self.assertIn(OPT_IN_PROMPT, markdown)

    def test_stack_steps_keep_advisor_contract_and_blocked_reports_forbid_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            report, markdown = run_recommender(
                tmp,
                ["vision-language-omni"],
                target_profile={
                    "schema_version": 1,
                    "hardware": {},
                    "software": {},
                    "capabilities": [],
                    "workloads": ["repeated-media"],
                },
            )
            privacy = next(
                step
                for step in report["planning_stack"]["steps"]
                if step["method"] == "cache-privacy-and-isolation"
            )
            self.assertEqual(privacy["status"], "research-candidate")
            self.assertEqual(privacy["advisor_bucket"], "experimental-approach")
            self.assertTrue(privacy["requires_user_opt_in"])
            self.assertEqual(privacy["opt_in_prompt"], OPT_IN_PROMPT)
            self.assertFalse(privacy["execution_allowed"])
            self.assertIn("experimental-approach; opt-in required", markdown)

            unscoped, _ = run_recommender(tmp, ["vision-language-omni"])
            self.assertNotIn("planning_stack", unscoped)

            blocked, _ = run_recommender(
                tmp,
                ["dense-decoder-transformer"],
                model_type="llama",
                blockers=["high: remote-code auto_map present"],
            )
            self.assertTrue(blocked["blocked"])
            self.assertTrue(blocked["held_candidates"])
            self.assertTrue(
                all(not candidate["execution_allowed"] for candidate in blocked["held_candidates"])
            )
            self.assertTrue(
                all(
                    not step["execution_allowed"]
                    for step in blocked["held_planning_stack"]["steps"]
                )
            )

    def test_unknown_family_inputs_and_overrides_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            write_inspection(inspection, ["not-a-real-family"])
            result = run_script("recommend_optimizations.py", inspection, expected=2)
            self.assertIn("unknown architecture family ids: not-a-real-family", result.stderr)

            write_inspection(inspection, ["dense-decoder-transformer"])
            result = run_script(
                "recommend_optimizations.py",
                inspection,
                "--family",
                "also-not-a-family",
                expected=2,
            )
            self.assertIn("unknown architecture family ids: also-not-a-family", result.stderr)

            result = run_script(
                "recommend_optimizations.py",
                inspection,
                "--objective",
                "not-a-real-objective",
                expected=2,
            )
            self.assertIn("unknown objective ids: not-a-real-objective", result.stderr)

    def test_target_profile_rejects_malformed_experiment_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            write_inspection(inspection, ["dense-decoder-transformer"], model_type="llama")
            profile = tmp / "target-profile.json"
            profile.write_text(json.dumps({
                "schema_version": 1,
                "hardware": {},
                "software": {},
                "capabilities": [],
                "workloads": [],
                "experiment_fingerprint_sha256": "not-a-digest",
            }), encoding="utf-8")

            result = run_script(
                "recommend_optimizations.py",
                inspection,
                "--target-profile",
                profile,
                expected=2,
            )
            self.assertIn(
                "experiment_fingerprint_sha256 must be a lowercase 64-hex digest",
                result.stderr,
            )

    def test_missing_target_profile_holds_version_sensitive_numbers_and_stack_products(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            report, markdown = run_recommender(Path(raw_tmp), ["dense-decoder-transformer"], model_type="llama")
            self.assertEqual(report["target_profile_status"], "missing")
            quantization = candidate_by_id(report, "native-low-bit-weight-quantization")
            self.assertIsNotNone(quantization)
            self.assertEqual(quantization["improvement_band"]["provenance"], "profile_required")
            self.assertIsNone(quantization["improvement_band"].get("range"))
            self.assertTrue(
                any("Effective claim catalog" in reason for reason in quantization["claim_holds"])
            )
            batching = candidate_by_id(report, "continuous-batching-serving")
            self.assertIsNotNone(batching)
            self.assertTrue(any("Effective claim catalog" in reason for reason in batching["claim_holds"]))
            self.assertIn("planning_stack", report)
            self.assertNotIn("recommended_stack", report)
            self.assertNotIn("measured", report["planning_stack"]["compound"])
            self.assertIsNone(report["planning_stack"]["compound"]["hypothesis_ceiling"]["ceiling"])
            self.assertIn("Rejected observed configurations", markdown)
            self.assertIn("observed `0.33x`", markdown)
            for unsupported in ("2.64x", "23.8x", "784.0x"):
                self.assertNotIn(unsupported, markdown)

    def test_effective_claim_catalog_is_current_complete_and_the_only_numeric_authority(self) -> None:
        import recommend_optimizations

        raw_method = {
            "id": "a",
            "status": "proven-mlx-port",
            "target_constraints": {
                "required_profile_fields": [],
                "exact_software": {},
                "workloads_any": [],
            },
            "expected_effect": "Raw guidance claims 99.0x.",
            "improvement_band": {
                "provenance": "source_reported",
                "claim_status": "eligible-with-profile",
                "range": "1.0x-99.0x",
                "metric": "decode-tokens-per-sec",
                "basis": "raw",
                "applies_when": "raw",
                "evidence_lineage_ids": ["lineage-a"],
            },
        }
        claim = {
            "method_id": "a",
            "method_status": "proven-mlx-port",
            "provenance": "source_reported",
            "claim_status": "eligible-with-profile",
            "metric": "decode-tokens-per-sec",
            "observed_range": "1.0x-99.0x",
            "profile_eligible_range": "1.0x-2.0x",
            "effective_range": None,
            "promotion_state": "profile-eligible",
            "evidence_lineage_ids": ["lineage-a"],
            "withheld_reasons": [],
        }
        profile_candidate, profile_holds = recommend_optimizations.prepare_effective_claim(
            raw_method,
            claim,
        )
        self.assertEqual(profile_candidate["improvement_band"]["range"], "1.0x-2.0x")
        self.assertNotIn("99.0x", json.dumps(profile_candidate))
        self.assertEqual(profile_holds, [])

        local_claim = copy.deepcopy(claim)
        local_claim.update({
            "provenance": "local_reproduced",
            "profile_eligible_range": None,
            "effective_range": "1.0x-1.25x",
            "promotion_state": "local-promotion",
            "experiment_fingerprint": stack_experiment_fingerprint("local-claim"),
        })
        local_candidate, local_holds = recommend_optimizations.prepare_effective_claim(
            raw_method,
            local_claim,
        )
        self.assertEqual(local_candidate["improvement_band"]["range"], "1.0x-1.25x")
        fingerprint = local_candidate["improvement_band"]["experiment_fingerprint"]
        self.assertEqual(
            local_candidate["effective_claim_state"]["experiment_fingerprint"],
            fingerprint,
        )
        self.assertNotIn("99.0x", json.dumps(local_candidate))
        self.assertEqual(local_holds, [])
        missing_profile = target_claim_holds(local_candidate, None)
        self.assertTrue(any("full canonical" in reason for reason in missing_profile))
        wrong_profile = target_claim_holds(
            local_candidate,
            {
                "schema_version": 1,
                "hardware": {},
                "software": {},
                "capabilities": [],
                "workloads": [],
                "experiment_fingerprint_sha256": "0" * 64,
            },
        )
        self.assertTrue(any("experiment_fingerprint does not exactly match" in reason for reason in wrong_profile))
        self.assertTrue(any("generic profile" in reason for reason in wrong_profile))
        copied_digest_only = target_claim_holds(
            local_candidate,
            {
                "schema_version": 1,
                "hardware": {},
                "software": {},
                "capabilities": [],
                "workloads": [],
                "experiment_fingerprint_sha256": fingerprint["sha256"],
            },
        )
        self.assertTrue(any(
            "experiment_fingerprint does not exactly match" in reason
            for reason in copied_digest_only
        ))
        self.assertTrue(any("generic profile" in reason for reason in copied_digest_only))
        exact_profile = target_claim_holds(
            local_candidate,
            exact_target_profile(fingerprint),
        )
        self.assertEqual(exact_profile, [])

        copied_fingerprint_wrong_target = exact_target_profile(fingerprint)
        copied_fingerprint_wrong_target["hardware"] = {"chip": "Apple Other"}
        self.assertTrue(any(
            "hardware does not exactly match" in reason
            for reason in target_claim_holds(local_candidate, copied_fingerprint_wrong_target)
        ))

        copied_fingerprint_wrong_workload = exact_target_profile(fingerprint)
        copied_fingerprint_wrong_workload["workload"] = {"id": "lookalike"}
        self.assertTrue(any(
            "workload does not exactly match" in reason
            for reason in target_claim_holds(local_candidate, copied_fingerprint_wrong_workload)
        ))

        withheld_claim = copy.deepcopy(claim)
        withheld_claim.update({
            "profile_eligible_range": None,
            "promotion_state": "withheld",
            "withheld_reasons": ["claim-status-held"],
        })
        withheld_candidate, withheld_holds = recommend_optimizations.prepare_effective_claim(
            raw_method,
            withheld_claim,
        )
        self.assertIsNone(withheld_candidate["improvement_band"]["range"])
        self.assertNotIn("99.0x", withheld_candidate["expected_effect"])
        self.assertTrue(any("claim-status-held" in reason for reason in withheld_holds))

        second_method = copy.deepcopy(raw_method)
        second_method["id"] = "b"
        second_method["improvement_band"]["evidence_lineage_ids"] = ["lineage-b"]
        second_claim = copy.deepcopy(claim)
        second_claim["method_id"] = "b"
        second_claim["evidence_lineage_ids"] = ["lineage-b"]
        second_candidate, _ = recommend_optimizations.prepare_effective_claim(
            second_method,
            second_claim,
        )
        stack = {
            "id": "catalog-only",
            "primary_metric": "decode-tokens-per-sec",
            "steps": [
                {"method": "a", "lossiness": "lossless", "gate": "a"},
                {"method": "b", "lossiness": "lossless", "gate": "b"},
            ],
            "composition_notes": [
                {"pair": ["a", "b"], "validity": "validated-composable", "why": "test"}
            ],
            "compound": {"measured_together": False, "receipts": []},
        }
        raw_composed = compose_stack_band(stack, [raw_method, second_method])
        self.assertIsNone(raw_composed["hypothesis_ceiling"]["ceiling"])
        self.assertTrue(all(step["band"] is None for step in raw_composed["per_step"]))
        self.assertTrue(
            any("effective_claims numeric authority" in reason for reason in raw_composed["withheld_reasons"])
        )
        composed = compose_stack_band(stack, [profile_candidate, second_candidate])
        self.assertEqual(composed["hypothesis_ceiling"]["ceiling"], "4.0x")

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            inspection = tmp / "inspection.json"
            write_inspection(inspection, ["dense-decoder-transformer"], model_type="llama")

            current = json.loads(run_script("recommend_optimizations.py", inspection).stdout)
            self.assertEqual(current["effective_claim_catalog_status"], "current")

            missing = tmp / "missing-effective-claims.json"
            result = run_script(
                "recommend_optimizations.py",
                inspection,
                "--effective-claims",
                missing,
                expected=2,
            )
            self.assertIn("Effective claim catalog not found", result.stderr)

            catalog_path = SKILL / "assets" / "effective_claims.json"
            stale_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            stale_catalog["claims"].pop()
            stale_catalog["claim_count"] -= 1
            stale_path = tmp / "stale-effective-claims.json"
            stale_path.write_text(json.dumps(stale_catalog), encoding="utf-8")
            result = run_script(
                "recommend_optimizations.py",
                inspection,
                "--effective-claims",
                stale_path,
                expected=2,
            )
            self.assertIn("must contain exactly one record", result.stderr)

            stale_guidance = json.loads(
                (SKILL / "assets" / "optimization_guidance.yaml").read_text(encoding="utf-8")
            )
            next(
                method
                for method in stale_guidance["methods"]
                if method["id"] == "continuous-batching-serving"
            )["improvement_band"]["range"] = "1.0x-99.0x"
            guidance_path = tmp / "stale-guidance.json"
            guidance_path.write_text(json.dumps(stale_guidance), encoding="utf-8")
            result = run_script(
                "recommend_optimizations.py",
                inspection,
                "--guidance",
                guidance_path,
                expected=2,
            )
            self.assertIn("Effective claim catalog is stale", result.stderr)

    def test_compound_numbers_require_validated_pairs_and_unique_evidence_lineages(self) -> None:
        methods = [
            {
                "id": "a",
                "improvement_band": {
                    "provenance": "source_reported",
                    "range": "1.0x-2.0x",
                    "metric": "throughput",
                    "evidence_lineage_ids": ["lineage-a"],
                    "numeric_authority": "effective_claims",
                },
            },
            {
                "id": "b",
                "improvement_band": {
                    "provenance": "source_reported",
                    "range": "1.0x-3.0x",
                    "metric": "throughput",
                    "evidence_lineage_ids": ["lineage-b"],
                    "numeric_authority": "effective_claims",
                },
            },
        ]

        def stack(validity: str) -> dict[str, object]:
            return {
                "id": f"pair-{validity}",
                "primary_metric": "throughput",
                "steps": [
                    {"method": "a", "lossiness": "lossless", "gate": "a"},
                    {"method": "b", "lossiness": "lossless", "gate": "b"},
                ],
                "composition_notes": [{"pair": ["a", "b"], "validity": validity, "why": "test"}],
                "compound": {"measured_together": False, "receipts": []},
            }

        unknown = compose_stack_band(stack("unknown"), methods)
        self.assertIsNone(unknown["hypothesis_ceiling"]["ceiling"])
        self.assertEqual(unknown["composition_status"], "withheld")

        mutually_exclusive = compose_stack_band(stack("mutually-exclusive"), methods)
        self.assertIsNone(mutually_exclusive["hypothesis_ceiling"]["ceiling"])

        duplicate_methods = json.loads(json.dumps(methods))
        duplicate_methods[1]["improvement_band"]["evidence_lineage_ids"] = ["lineage-a"]
        duplicate = compose_stack_band(stack("validated-composable"), duplicate_methods)
        self.assertIsNone(duplicate["hypothesis_ceiling"]["ceiling"])
        self.assertTrue(any("duplicate evidence lineage" in reason for reason in duplicate["withheld_reasons"]))

        composable = compose_stack_band(stack("validated-composable"), methods)
        self.assertEqual(composable["composition_status"], "numeric-hypothesis")
        self.assertEqual(composable["hypothesis_ceiling"]["ceiling"], "6.0x")

    def test_compose_rejects_duplicate_steps_self_pairs_and_duplicate_pair_notes(self) -> None:
        methods = [
            {
                "id": method_id,
                "improvement_band": {
                    "provenance": "source_reported",
                    "range": band,
                    "metric": "throughput",
                    "evidence_lineage_ids": [f"lineage-{method_id}"],
                },
            }
            for method_id, band in (("a", "1.0x-2.0x"), ("b", "1.0x-3.0x"))
        ]
        base = {
            "id": "structural-guard",
            "primary_metric": "throughput",
            "steps": [
                {"method": "a", "lossiness": "lossless", "gate": "a"},
                {"method": "b", "lossiness": "lossless", "gate": "b"},
            ],
            "composition_notes": [
                {"pair": ["a", "b"], "validity": "validated-composable", "why": "test"}
            ],
            "compound": {"measured_together": False, "receipts": []},
        }

        duplicate_step = copy.deepcopy(base)
        duplicate_step["steps"].append(copy.deepcopy(duplicate_step["steps"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate step method a"):
            compose_stack_band(duplicate_step, methods)

        self_pair = copy.deepcopy(base)
        self_pair["composition_notes"] = [
            {"pair": ["a", "a"], "validity": "mutually-exclusive", "why": "test"}
        ]
        with self.assertRaisesRegex(ValueError, "self composition pair a"):
            compose_stack_band(self_pair, methods)

        duplicate_pair = copy.deepcopy(base)
        duplicate_pair["composition_notes"].append(
            {"pair": ["b", "a"], "validity": "known-conflicting", "why": "test"}
        )
        with self.assertRaisesRegex(ValueError, r"duplicate composition pair a \+ b"):
            compose_stack_band(duplicate_pair, methods)

    def test_measured_compound_requires_external_promotion_and_exact_ordered_coverage(self) -> None:
        methods = [
            {
                "id": method_id,
                "improvement_band": {
                    "provenance": "source_reported",
                    "range": band,
                    "metric": "decode-tokens-per-sec",
                    "evidence_lineage_ids": [f"lineage-{method_id}"],
                },
            }
            for method_id, band in (("a", "1.0x-2.0x"), ("b", "1.0x-3.0x"))
        ]
        stack = {
            "id": "measured-guard",
            "primary_metric": "decode-tokens-per-sec",
            "steps": [
                {"method": "a", "lossiness": "lossless", "gate": "a"},
                {"method": "b", "lossiness": "lossless", "gate": "b"},
            ],
            "composition_notes": [
                {"pair": ["a", "b"], "validity": "validated-composable", "why": "test"}
            ],
            "compound": {
                "measured_together": True,
                "receipts": [
                    {
                        "label": "candidate",
                        "file": "candidate.json",
                        "metric": "decode-tokens-per-sec",
                        "measured_ratio": "99.0x",
                        "measured_on": {"chip": "forged-inline-machine"},
                        "basis": "Forged inline basis claims 99.0x.",
                    }
                ],
            },
        }

        observation = {
            "schema_version": 1,
            "assessments": [
                {
                    "receipt": "candidate.json",
                    "label": "candidate",
                    "receipt_sha256": "a" * 64,
                    "classification": "performance_observation",
                    "enabled_methods": ["a", "b"],
                    "reasons": ["quality gate is incomplete"],
                    "gates": {"quality": False},
                    "primary_metric": "generation_tps",
                    "recomputed_median_ratios": {"decode_tps": 1.5},
                }
            ],
        }
        observed = compose_stack_band(stack, methods, observation)
        self.assertNotIn("measured", observed)
        self.assertEqual(
            observed["measured_evidence"][0]["classification"],
            "performance_observation",
        )

        promotion = copy.deepcopy(observation)
        promotion["assessments"][0]["classification"] = "promotion_ready"
        promotion["assessments"][0]["promotion_ready"] = True
        promotion["assessments"][0]["reasons"] = []
        promotion["assessments"][0]["gates"] = {
            gate: True
            for gate in (
                "aggregates_recomputed",
                "model_lineage_pinned",
                "target_hash_valid",
                "workload_hash_valid",
                "raw_outputs_valid",
                "quality_valid",
                "stability_passed",
                "stability_threshold_valid",
                "rollback_defined",
                "baseline_compatible",
                "enabled_methods_valid",
                "improvement_beyond_noise",
                "runner_valid",
                "quality_binding_valid",
                "experiment_descriptor_valid",
                "experiment_compatible",
                "source_identity_pinned",
                "baseline_source_identity_match",
                "execution_attested",
            )
        }
        promotion["assessments"][0]["experiment_fingerprint"] = stack_experiment_fingerprint()
        unverified = compose_stack_band(stack, methods, promotion)
        self.assertNotIn("measured", unverified)
        self.assertEqual(unverified["measured_evidence"][0]["classification"], "incomplete")
        self.assertTrue(
            any(
                "verified receipt" in reason
                for reason in unverified["measured_evidence"][0]["reasons"]
            )
        )

        verified_receipts = {
            "candidate.json": copy.deepcopy(promotion["assessments"][0])
        }
        promoted = compose_stack_band(
            stack,
            methods,
            promotion,
            verified_receipts=verified_receipts,
        )
        self.assertEqual(promoted["measured"]["ratio"], "1.5x")
        self.assertNotEqual(promoted["measured"]["ratio"], "99.0x")
        self.assertNotIn("99.0x", json.dumps(promoted["measured"]))
        self.assertEqual(promoted["measured"]["provenance"], "local_reproduced")
        self.assertEqual(promoted["measured"]["measured_on"]["chip"], "Apple Test")
        self.assertEqual(
            promoted["measured"]["measured_on"]["experiment_fingerprint_sha256"],
            promotion["assessments"][0]["experiment_fingerprint"]["sha256"],
        )
        self.assertNotIn("forged-inline-machine", json.dumps(promoted["measured"]))

        copied_fingerprint = copy.deepcopy(promotion)
        copied_fingerprint["assessments"][0]["receipt_sha256"] = "e" * 64
        copied_verified = {
            "candidate.json": copy.deepcopy(copied_fingerprint["assessments"][0])
        }
        copied_result = compose_stack_band(
            stack,
            methods,
            copied_fingerprint,
            verified_receipts=copied_verified,
        )
        self.assertNotIn("measured", copied_result)
        self.assertTrue(any(
            "experiment_fingerprint receipt_sha256 does not match" in reason
            for reason in copied_result["measured_evidence"][0]["reasons"]
        ))

        heterogeneous_stack = copy.deepcopy(stack)
        heterogeneous_stack["compound"]["receipts"] = [
            {"label": "candidate-a", "file": "candidate-a.json", "measured_on": {"chip": "forged-a"}},
            {"label": "candidate-b", "file": "candidate-b.json", "measured_on": {"chip": "forged-b"}},
        ]
        heterogeneous_assessments = {"schema_version": 1, "assessments": []}
        heterogeneous_verified: dict[str, dict[str, object]] = {}
        for label, ratio in (("candidate-a", 1.5), ("candidate-b", 9.0)):
            row = copy.deepcopy(promotion["assessments"][0])
            row["receipt"] = f"{label}.json"
            row["label"] = label
            row["recomputed_median_ratios"]["decode_tps"] = ratio
            row["experiment_fingerprint"] = stack_experiment_fingerprint(label)
            heterogeneous_assessments["assessments"].append(row)
            heterogeneous_verified[f"{label}.json"] = copy.deepcopy(row)
        heterogeneous = compose_stack_band(
            heterogeneous_stack,
            methods,
            heterogeneous_assessments,
            verified_receipts=heterogeneous_verified,
        )
        self.assertNotIn("measured", heterogeneous)
        self.assertTrue(
            all(
                "heterogeneous verified experiment fingerprints" in evidence["reasons"]
                for evidence in heterogeneous["measured_evidence"]
            )
        )
        self.assertNotIn("9.0x", json.dumps(heterogeneous.get("measured")))

        compatible_stack = copy.deepcopy(stack)
        compatible_stack["compound"]["receipts"] = [
            {"label": "candidate-a", "file": "candidate-a.json"},
            {"label": "candidate-b", "file": "candidate-b.json"},
        ]
        compatible_assessments = {"schema_version": 1, "assessments": []}
        compatible_verified: dict[str, dict[str, object]] = {}
        selected_fingerprint: dict[str, object] | None = None
        for label, ratio, receipt_sha256 in (
            ("candidate-a", 1.5, "c" * 64),
            ("candidate-b", 1.1, "d" * 64),
        ):
            row = copy.deepcopy(promotion["assessments"][0])
            row["receipt"] = f"{label}.json"
            row["label"] = label
            row["receipt_sha256"] = receipt_sha256
            row["recomputed_median_ratios"]["decode_tps"] = ratio
            row["experiment_fingerprint"] = stack_experiment_fingerprint(
                receipt_sha256=receipt_sha256
            )
            compatible_assessments["assessments"].append(row)
            compatible_verified[f"{label}.json"] = copy.deepcopy(row)
            if label == "candidate-b":
                selected_fingerprint = row["experiment_fingerprint"]
        compatible = compose_stack_band(
            compatible_stack,
            methods,
            compatible_assessments,
            verified_receipts=compatible_verified,
        )
        self.assertEqual(compatible["measured"]["ratio"], "1.1x")
        self.assertEqual(
            compatible["measured"]["experiment_fingerprint"],
            selected_fingerprint,
        )
        self.assertEqual(compatible["measured"]["receipt"], "candidate-b.json")

        reversed_stack = copy.deepcopy(compatible_stack)
        reversed_stack["compound"]["receipts"].reverse()
        reversed_compatible = compose_stack_band(
            reversed_stack,
            methods,
            {"schema_version": 1, "assessments": list(reversed(compatible_assessments["assessments"]))},
            verified_receipts=compatible_verified,
        )
        self.assertEqual(reversed_compatible["measured"], compatible["measured"])

        rejected_mix = copy.deepcopy(compatible_assessments)
        rejected_mix["assessments"][0]["classification"] = "rejected"
        rejected_mix["assessments"][0]["promotion_ready"] = False
        rejected_verified = copy.deepcopy(compatible_verified)
        rejected_verified["candidate-a.json"] = copy.deepcopy(rejected_mix["assessments"][0])
        mixed = compose_stack_band(
            compatible_stack,
            methods,
            rejected_mix,
            verified_receipts=rejected_verified,
        )
        self.assertNotIn("measured", mixed)
        self.assertTrue(any(row["classification"] == "rejected" for row in mixed["measured_evidence"]))

        missing_stack = copy.deepcopy(compatible_stack)
        missing_stack["compound"]["receipts"].append(
            {"label": "candidate-missing", "file": "candidate-missing.json"}
        )
        missing = compose_stack_band(
            missing_stack,
            methods,
            compatible_assessments,
            verified_receipts=compatible_verified,
        )
        self.assertNotIn("measured", missing)
        self.assertTrue(any(
            "missing external receipt assessment" in row["reasons"]
            for row in missing["measured_evidence"]
        ))

        unsupported_stack = copy.deepcopy(stack)
        unsupported_stack["primary_metric"] = "batch-throughput"
        unsupported = compose_stack_band(unsupported_stack, methods, promotion)
        self.assertNotIn("measured", unsupported)
        self.assertEqual(unsupported["measured_evidence"][0]["classification"], "incomplete")
        self.assertTrue(
            any(
                "no controlled assessment ratio mapping" in reason
                for reason in unsupported["measured_evidence"][0]["reasons"]
            )
        )

        wrong_order = copy.deepcopy(promotion)
        wrong_order["assessments"][0]["enabled_methods"] = ["b", "a"]
        incomplete = compose_stack_band(stack, methods, wrong_order)
        self.assertNotIn("measured", incomplete)
        self.assertEqual(incomplete["measured_evidence"][0]["classification"], "incomplete")
        self.assertEqual(
            incomplete["measured_evidence"][0]["original_classification"],
            "promotion_ready",
        )
        self.assertTrue(
            any(
                "ordered enabled_methods do not exactly match" in reason
                for reason in incomplete["measured_evidence"][0]["reasons"]
            )
        )

        forged = copy.deepcopy(promotion)
        forged["assessments"][0]["promotion_ready"] = False
        forged["assessments"][0]["receipt_sha256"] = "not-a-digest"
        forged["assessments"][0]["reasons"] = ["quality review is still open"]
        forged["assessments"][0]["gates"]["quality_valid"] = False
        forged["assessments"][0]["gates"]["stability_threshold_valid"] = False
        forged["assessments"][0]["gates"]["runner_valid"] = False
        forged["assessments"][0]["gates"]["experiment_compatible"] = False
        forged["assessments"][0]["gates"]["source_identity_pinned"] = False
        rejected_forgery = compose_stack_band(stack, methods, forged)
        self.assertNotIn("measured", rejected_forgery)
        forged_evidence = rejected_forgery["measured_evidence"][0]
        self.assertEqual(forged_evidence["classification"], "incomplete")
        self.assertEqual(forged_evidence["original_classification"], "promotion_ready")
        self.assertTrue(
            any("promotion_ready boolean is not true" in reason for reason in forged_evidence["reasons"])
        )
        self.assertTrue(any("receipt_sha256" in reason for reason in forged_evidence["reasons"]))
        self.assertTrue(any("quality_valid" in reason for reason in forged_evidence["reasons"]))
        self.assertTrue(
            any("stability_threshold_valid" in reason for reason in forged_evidence["reasons"])
        )
        self.assertTrue(any("runner_valid" in reason for reason in forged_evidence["reasons"]))
        self.assertTrue(
            any("experiment_compatible" in reason for reason in forged_evidence["reasons"])
        )
        self.assertTrue(
            any("source_identity_pinned" in reason for reason in forged_evidence["reasons"])
        )

        metric_mismatch = copy.deepcopy(promotion)
        metric_mismatch["assessments"][0]["primary_metric"] = "prompt_tps"
        mismatched = compose_stack_band(stack, methods, metric_mismatch)
        self.assertNotIn("measured", mismatched)
        self.assertEqual(mismatched["measured_evidence"][0]["classification"], "incomplete")
        self.assertTrue(
            any("primary_metric" in reason for reason in mismatched["measured_evidence"][0]["reasons"])
        )

        missing_ratio = copy.deepcopy(promotion)
        missing_ratio["assessments"][0]["recomputed_median_ratios"] = {}
        incomplete_ratio = compose_stack_band(stack, methods, missing_ratio)
        self.assertNotIn("measured", incomplete_ratio)
        self.assertEqual(
            incomplete_ratio["measured_evidence"][0]["classification"],
            "incomplete",
        )

        regression = copy.deepcopy(promotion)
        regression["assessments"][0]["recomputed_median_ratios"]["decode_tps"] = 0.9
        rejected_regression = compose_stack_band(stack, methods, regression)
        self.assertNotIn("measured", rejected_regression)
        self.assertEqual(
            rejected_regression["measured_evidence"][0]["classification"],
            "rejected",
        )
        self.assertTrue(
            any("does not exceed 1.0" in reason for reason in rejected_regression["measured_evidence"][0]["reasons"])
        )

    def test_source_validator_rejects_duplicate_stack_structure_and_accepts_mutual_exclusion(self) -> None:
        mutations = (
            (
                "duplicate-step",
                lambda stack: stack["steps"].append(copy.deepcopy(stack["steps"][0])),
                "has duplicate step method",
                1,
            ),
            (
                "self-pair",
                lambda stack: stack["composition_notes"].append({
                    "pair": [stack["steps"][0]["method"], stack["steps"][0]["method"]],
                    "validity": "mutually-exclusive",
                    "why": "invalid self pair",
                }),
                "has self composition pair",
                1,
            ),
            (
                "duplicate-pair",
                lambda stack: stack["composition_notes"].append({
                    **copy.deepcopy(stack["composition_notes"][0]),
                    "pair": list(reversed(stack["composition_notes"][0]["pair"])),
                }),
                "has duplicate composition pair",
                1,
            ),
            (
                "mutually-exclusive",
                lambda stack: stack["composition_notes"][0].update({
                    "validity": "mutually-exclusive",
                }),
                '"ok": true',
                0,
            ),
        )
        for name, mutate, expected, exit_code in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw_tmp:
                skill = Path(raw_tmp) / "skill"
                shutil.copytree(SKILL / "assets", skill / "assets")
                stacks_path = skill / "assets" / "optimization_stacks.yaml"
                stacks = json.loads(stacks_path.read_text(encoding="utf-8"))
                mutate(stacks["stacks"][0])
                stacks_path.write_text(json.dumps(stacks, indent=2) + "\n", encoding="utf-8")
                result = run_script("validate_sources.py", skill, expected=exit_code)
                self.assertIn(expected, result.stdout)

    def test_negative_dense_observation_vlm_dedup_and_qwen_tts_scope_are_encoded(self) -> None:
        stacks = json.loads((SKILL / "assets" / "optimization_stacks.yaml").read_text(encoding="utf-8"))
        by_stack = {stack["id"]: stack for stack in stacks["stacks"]}
        dense = by_stack["dense-decoder-inference"]
        self.assertEqual(dense["status"], "planning-only")
        self.assertFalse(dense["compound"]["measured_together"])
        rejected = dense["observed_configurations"][0]
        self.assertEqual(rejected["decision"], "rejected")
        self.assertEqual(rejected["measured_ratio"], "0.33x")

        vlm_methods = [step["method"] for step in by_stack["vlm-repeated-media"]["steps"]]
        self.assertNotIn("content-prefix-cache-vlm", vlm_methods)
        self.assertEqual(len(vlm_methods), len(set(vlm_methods)))

        guidance = json.loads((SKILL / "assets" / "optimization_guidance.yaml").read_text(encoding="utf-8"))
        by_method = {method["id"]: method for method in guidance["methods"]}
        self.assertEqual(by_method["content-prefix-cache-vlm"]["status"], "rejected-or-superseded")
        qwen = by_method["qwen3-tts-batch-generation"]["match"]
        self.assertEqual(qwen["model_types"], ["qwen3_tts"])
        self.assertEqual(qwen["capabilities_all"], ["api:batch-generate"])
        self.assertEqual(qwen["workloads_any"], ["concurrent-tts-short-prompts"])

    def test_compound_stack_documentation_matches_registered_steps_and_authority(self) -> None:
        stacks = json.loads((SKILL / "assets" / "optimization_stacks.yaml").read_text(encoding="utf-8"))
        documentation = (SKILL / "references" / "compound-stacks.md").read_text(encoding="utf-8")
        for index, stack in enumerate(stacks["stacks"]):
            start = documentation.index(f"`{stack['id']}`")
            end = (
                documentation.index(f"`{stacks['stacks'][index + 1]['id']}`", start)
                if index + 1 < len(stacks["stacks"])
                else documentation.index("## Derived band rule", start)
            )
            section = documentation[start:end]
            with self.subTest(stack=stack["id"]):
                for step in stack["steps"]:
                    self.assertIn(f"`{step['method']}`", section)
        self.assertNotIn("`content-prefix-cache-vlm`", documentation)
        self.assertIn("`assets/effective_claims.json` is the only numeric authority", documentation)
        self.assertIn("nonempty, unique evidence", documentation)
        self.assertIn("generated receipt assessment is promotion-ready", documentation)

    def test_qwen_tts_observation_stays_held_even_for_the_broad_matching_profile(self) -> None:
        target = {
            "schema_version": 1,
            "hardware": {"chip": "Apple M4 Pro", "memory_gb": 48},
            "software": {"mlx_audio": "0.4.4"},
            "capabilities": ["api:batch-generate"],
            "workloads": ["concurrent-tts-short-prompts"],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            generic, _ = run_recommender(
                tmp,
                ["autoregressive-audio-lm"],
                model_type="bark",
                target_profile=target,
            )
            self.assertNotIn("qwen3-tts-batch-generation", candidate_ids(generic))

            qwen, _ = run_recommender(
                tmp,
                ["autoregressive-audio-lm"],
                model_type="qwen3_tts",
                target_profile=target,
            )
            candidate = candidate_by_id(qwen, "qwen3-tts-batch-generation")
            self.assertIsNotNone(candidate)
            self.assertIsNone(candidate["improvement_band"]["range"])
            self.assertTrue(any("Effective claim catalog" in reason for reason in candidate["claim_holds"]))
            self.assertNotIn("5.45x", candidate["expected_effect"])

    def test_source_benchmark_ranges_never_escape_through_arbitrary_target_profiles(self) -> None:
        cases = [
            {
                "method_id": "continuous-batching-serving",
                "families": ["dense-decoder-transformer"],
                "model_type": "llama",
                "profile": {
                    "schema_version": 1,
                    "hardware": {"chip": "Apple M1", "memory_gb": 8},
                    "software": {
                        "vllm_mlx_revision": "a48c86c1a41900f7d26658471b5f67e5fdd35445"
                    },
                    "capabilities": [],
                    "workloads": ["concurrent-serving"],
                },
                "forbidden": "4.3x",
            },
            {
                "method_id": "multimodal-content-prefix-cache",
                "families": ["vision-language-omni"],
                "model_type": "llava",
                "profile": {
                    "schema_version": 1,
                    "hardware": {"chip": "Apple M1", "memory_gb": 8},
                    "software": {
                        "vllm_mlx_revision": "a48c86c1a41900f7d26658471b5f67e5fdd35445"
                    },
                    "capabilities": [],
                    "workloads": ["repeated-media"],
                },
                "forbidden": "28.0x",
            },
            {
                "method_id": "qwen3-tts-batch-generation",
                "families": ["autoregressive-audio-lm"],
                "model_type": "qwen3_tts",
                "profile": {
                    "schema_version": 1,
                    "hardware": {"chip": "Apple M1", "memory_gb": 8},
                    "software": {"mlx_audio": "0.4.4"},
                    "capabilities": ["api:batch-generate"],
                    "workloads": ["concurrent-tts-short-prompts"],
                },
                "forbidden": "5.45x",
            },
        ]
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            for case in cases:
                with self.subTest(method_id=case["method_id"]):
                    report, markdown = run_recommender(
                        tmp,
                        case["families"],
                        model_type=case["model_type"],
                        target_profile=case["profile"],
                    )
                    candidate = candidate_by_id(report, case["method_id"])
                    self.assertIsNotNone(candidate)
                    self.assertIsNone(candidate["improvement_band"]["range"])
                    self.assertTrue(candidate["claim_holds"])
                    self.assertNotIn(case["forbidden"], markdown)

    def test_activation_quantization_and_unsupported_speculative_outcomes_are_consistent(self) -> None:
        techniques = json.loads((SKILL / "assets" / "techniques.yaml").read_text(encoding="utf-8"))
        activation = next(item for item in techniques["techniques"] if item["id"] == "activation-quant")
        self.assertEqual(activation["status"], "native-mlx")
        self.assertIn("mlx-doc-core-quantize", activation["evidence"])
        self.assertIn("quantize_input", activation["use_when"])
        self.assertIn("generic", activation["avoid_when"].lower())

        outcomes = json.loads((SKILL / "assets" / "model_outcomes.json").read_text(encoding="utf-8"))
        by_id = {record["id"]: record for record in outcomes["records"]}
        for record_id in (
            "vlm-mlx-vlm-working-route",
            "asr-whisper-working-adjacent-route",
            "diffusion-and-flow-partial-routes",
            "audio-tts-qwen3-and-vocoder-routes",
        ):
            with self.subTest(record=record_id):
                speculative = by_id[record_id]["potential_speedup"]["speculative_decoding"]
                self.assertEqual(speculative["provenance"], "profile_required")
                self.assertIsNone(speculative.get("range"))

    def test_model_outcomes_hold_superseded_observations_and_require_controlled_provenance(self) -> None:
        outcomes = json.loads((SKILL / "assets" / "model_outcomes.json").read_text(encoding="utf-8"))
        self.assertEqual(outcomes["schema_version"], 2)
        self.assertTrue(
            any("promotion-ready" in policy for policy in outcomes["claim_policy"])
        )
        by_id = {record["id"]: record for record in outcomes["records"]}
        speculative = by_id["decoder-mlx-lm-working-route"]["potential_speedup"]["speculative_decoding"]
        self.assertEqual(speculative["provenance"], "performance_observation")
        self.assertIsNone(speculative["range"])
        self.assertEqual(speculative["observed_source_range"], "1.0x-1.3x")

        for record_id in (
            "asr-whisper-working-adjacent-route",
            "encoder-embedding-top-model-gap",
            "diffusion-and-flow-partial-routes",
            "time-series-and-structured-gap",
        ):
            with self.subTest(record=record_id):
                overall = by_id[record_id]["potential_speedup"]["overall"]
                self.assertEqual(overall["range"], "1.0x-1.0x")
                self.assertEqual(overall["provenance"], "not_a_speedup")

        with tempfile.TemporaryDirectory() as raw_tmp:
            skill = Path(raw_tmp) / "skill"
            shutil.copytree(SKILL / "assets", skill / "assets")
            outcomes_path = skill / "assets" / "model_outcomes.json"
            invalid = json.loads(outcomes_path.read_text(encoding="utf-8"))
            invalid["records"][0]["potential_speedup"]["overall"]["provenance"] = "uncontrolled"
            outcomes_path.write_text(json.dumps(invalid, indent=2) + "\n", encoding="utf-8")
            result = run_script("validate_sources.py", skill, expected=1)
            self.assertIn("has invalid provenance uncontrolled", result.stdout)

    def test_historical_speed_receipts_are_observations_not_promoted_local_claims(self) -> None:
        taxonomy = json.loads((SKILL / "assets" / "recommendation-taxonomy.yaml").read_text(encoding="utf-8"))
        self.assertIn("performance_observation", taxonomy["improvement_band_policy"])
        self.assertEqual(
            taxonomy["claim_provenance_to_advisor_bucket"]["performance_observation"],
            "benchmark-required",
        )

        guidance = json.loads((SKILL / "assets" / "optimization_guidance.yaml").read_text(encoding="utf-8"))
        by_method = {method["id"]: method for method in guidance["methods"]}
        for method_id in (
            "native-low-bit-weight-quantization",
            "uniform-kv-quantization",
            "prompt-prefix-cache",
            "draft-model-speculation",
        ):
            with self.subTest(method=method_id):
                band = by_method[method_id]["improvement_band"]
                self.assertEqual(band["provenance"], "performance_observation")
                self.assertEqual(band["claim_status"], "held")
                self.assertTrue(band["receipts"])

        kernel_band = by_method["spatial-grid-sample-kernel"]["improvement_band"]
        self.assertEqual(kernel_band["provenance"], "profile_required")
        self.assertIsNone(kernel_band["range"])
        self.assertEqual(kernel_band["observed_source_range"], "1.0x-10.0x")

        with tempfile.TemporaryDirectory() as raw_tmp:
            report, markdown = run_recommender(Path(raw_tmp), ["diffusion-flow"])
            kernel = candidate_by_id(report, "spatial-grid-sample-kernel")
            self.assertIsNotNone(kernel)
            self.assertTrue(kernel["claim_holds"])
            self.assertNotIn("10x", kernel["expected_effect"])
            self.assertNotIn("10x", markdown)


if __name__ == "__main__":
    unittest.main()
