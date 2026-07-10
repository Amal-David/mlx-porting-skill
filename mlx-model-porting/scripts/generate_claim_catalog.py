#!/usr/bin/env python3
"""Generate the conservative, deterministic catalog of effective numeric claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from _common import (
    SkillError,
    atomic_write_text,
    experiment_identity_from_fingerprint,
    load_structured,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_GUIDANCE = SKILL_ROOT / "assets" / "optimization_guidance.yaml"
DEFAULT_SOURCES = SKILL_ROOT / "assets" / "sources.yaml"
DEFAULT_ASSESSMENTS = SKILL_ROOT / "assets" / "benchmarks" / "receipt_assessments.json"
DEFAULT_OUTPUT = SKILL_ROOT / "assets" / "effective_claims.json"
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
ASSESSMENT_CLASSIFICATIONS = {"promotion_ready", "performance_observation", "rejected"}
INACTIVE_CLAIM_STATUSES = {"held", "superseded"}
INACTIVE_METHOD_STATUSES = {"rejected-or-superseded"}
REQUIRED_BENCHMARK_GATES = {
    "aggregates_recomputed",
    "baseline_source_identity_match",
    "baseline_compatible",
    "enabled_methods_valid",
    "execution_attested",
    "experiment_compatible",
    "experiment_descriptor_valid",
    "improvement_beyond_noise",
    "model_lineage_pinned",
    "quality_valid",
    "quality_binding_valid",
    "raw_outputs_valid",
    "rollback_defined",
    "runner_valid",
    "source_identity_pinned",
    "stability_passed",
    "stability_threshold_valid",
    "target_hash_valid",
    "workload_hash_valid",
}
LOCAL_METRIC_RATIOS = {
    "decode-tokens-per-sec": ("decode_tps", "generation_tps"),
    "generation-tokens-per-sec": ("decode_tps", "generation_tps"),
    "prefill-tokens-per-sec": ("prefill_tps", "prompt_tps"),
    "prompt-tokens-per-sec": ("prefill_tps", "prompt_tps"),
    "ttft-proxy": ("ttft_proxy_inverse", "ttft_proxy_s"),
    "peak-memory": ("peak_memory_inverse", "peak_memory_gb"),
    "wall-time": ("wall_seconds_inverse", "wall_seconds"),
    "kernel-latency": ("wall_seconds_inverse", "wall_seconds"),
}
MLX_BENCHMARK_METRICS = (
    "prompt_tokens",
    "prompt_tps",
    "generation_tokens",
    "generation_tps",
    "peak_memory_gb",
    "ttft_proxy_s",
)
EXTERNAL_BENCHMARK_METRICS = ("wall_seconds",)
BENCHMARK_METRIC_SETS = {
    frozenset(MLX_BENCHMARK_METRICS),
    frozenset(EXTERNAL_BENCHMARK_METRICS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or check the effective numeric-claim catalog")
    parser.add_argument("--guidance", default=str(DEFAULT_GUIDANCE))
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument("--assessments", default=str(DEFAULT_ASSESSMENTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true", help="Fail when the committed catalog has drifted")
    return parser.parse_args()


def _string_list(value: Any, *, field: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise SkillError(f"{field} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SkillError(f"{field} must contain non-empty strings")
        normalized.append(item.strip())
    if not allow_empty and not normalized:
        raise SkillError(f"{field} must not be empty")
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    if duplicates:
        raise SkillError(f"{field} contains duplicate values: {', '.join(duplicates)}")
    return normalized


def _canonical(value: Any) -> Any:
    """Return a JSON-compatible copy with mapping keys in deterministic order."""
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def experiment_fingerprint_valid(value: Any) -> bool:
    """Accept only the canonical promotion fingerprint emitted by receipt validation."""
    if not isinstance(value, dict) or set(value) != {"schema_version", "sha256", "payload"}:
        return False
    if value.get("schema_version") != 2:
        return False
    digest = value.get("sha256")
    payload = value.get("payload")
    if not isinstance(digest, str) or HEX_DIGEST_RE.fullmatch(digest) is None:
        return False
    if not isinstance(payload, dict) or set(payload) != {
        "candidate_receipt_sha256",
        "models",
        "target",
        "workload",
        "experiment",
        "primary_metric",
        "candidate_baseline_binding",
        "enabled_methods",
        "aggregate",
        "measured_runs",
        "quality",
    }:
        return False

    candidate_receipt_sha256 = payload.get("candidate_receipt_sha256")
    if (
        not isinstance(candidate_receipt_sha256, str)
        or HEX_DIGEST_RE.fullmatch(candidate_receipt_sha256) is None
    ):
        return False

    models = payload.get("models")
    if not isinstance(models, dict) or "target" not in models:
        return False
    model_fields = {"id", "revision", "lineage_id", "source_id", "source_revision"}
    if any(
        not isinstance(model, dict)
        or set(model) != model_fields
        or any(not isinstance(model.get(field), str) or not model[field] for field in model_fields)
        for model in models.values()
    ):
        return False

    target = payload.get("target")
    workload = payload.get("workload")
    experiment = payload.get("experiment")
    binding = payload.get("candidate_baseline_binding")
    enabled_methods = payload.get("enabled_methods")
    if (
        not isinstance(target, dict)
        or set(target) != {"descriptor", "sha256"}
        or not isinstance(target.get("descriptor"), dict)
        or not isinstance(target.get("sha256"), str)
        or HEX_DIGEST_RE.fullmatch(target["sha256"]) is None
    ):
        return False

    measured_runs = payload.get("measured_runs")
    aggregate = payload.get("aggregate")
    if not isinstance(measured_runs, list) or not measured_runs or not isinstance(aggregate, dict):
        return False
    metric_names = tuple(sorted(aggregate))
    if frozenset(metric_names) not in BENCHMARK_METRIC_SETS:
        return False
    metric_values: dict[str, list[float | int]] = {metric: [] for metric in metric_names}
    seen_run_ids: set[int] = set()
    for run in measured_runs:
        if not isinstance(run, dict) or set(run) != {"run", "metrics", "raw_output"}:
            return False
        run_id = run.get("run")
        metrics = run.get("metrics")
        raw_output = run.get("raw_output")
        if (
            isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id < 1
            or run_id in seen_run_ids
            or not isinstance(metrics, dict)
            or set(metrics) != set(metric_names)
            or not isinstance(raw_output, dict)
            or set(raw_output) != {"path", "sha256", "size_bytes", "truncated"}
            or not isinstance(raw_output.get("path"), str)
            or not raw_output["path"]
            or not isinstance(raw_output.get("sha256"), str)
            or HEX_DIGEST_RE.fullmatch(raw_output["sha256"]) is None
            or isinstance(raw_output.get("size_bytes"), bool)
            or not isinstance(raw_output.get("size_bytes"), int)
            or raw_output["size_bytes"] < 0
            or raw_output.get("truncated") is not False
        ):
            return False
        seen_run_ids.add(run_id)
        for metric in metric_names:
            metric_value = metrics.get(metric)
            if (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(float(metric_value))
            ):
                return False
            metric_values[metric].append(metric_value)
    for metric, values in metric_values.items():
        summary = aggregate.get(metric)
        expected = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if not isinstance(summary, dict) or set(summary) != set(expected):
            return False
        if any(
            isinstance(summary.get(key), bool)
            or not isinstance(summary.get(key), (int, float))
            or not math.isclose(
                float(summary[key]),
                float(expected_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key, expected_value in expected.items()
        ):
            return False

    quality = payload.get("quality")
    if (
        not isinstance(quality, dict)
        or set(quality) != {"status", "artifact", "result_sha256", "contract_identity"}
        or quality.get("status") != "pass"
        or not isinstance(quality.get("artifact"), dict)
        or set(quality["artifact"]) != {"path", "sha256", "size_bytes"}
        or not isinstance(quality["artifact"].get("path"), str)
        or not quality["artifact"]["path"]
        or not isinstance(quality["artifact"].get("sha256"), str)
        or HEX_DIGEST_RE.fullmatch(quality["artifact"]["sha256"]) is None
        or isinstance(quality["artifact"].get("size_bytes"), bool)
        or not isinstance(quality["artifact"].get("size_bytes"), int)
        or quality["artifact"]["size_bytes"] < 0
        or quality.get("result_sha256") != quality["artifact"]["sha256"]
        or experiment_identity_from_fingerprint(value) is None
    ):
        return False
    if (
        not isinstance(workload, dict)
        or set(workload) != {"id", "artifacts", "parameters", "sha256"}
        or not isinstance(workload.get("id"), str)
        or not workload["id"]
        or not isinstance(workload.get("artifacts"), list)
        or not workload["artifacts"]
        or not isinstance(workload.get("parameters"), dict)
        or not isinstance(workload.get("sha256"), str)
        or HEX_DIGEST_RE.fullmatch(workload["sha256"]) is None
    ):
        return False
    if (
        not isinstance(experiment, dict)
        or set(experiment) != {"invariant", "invariant_sha256", "variant", "variant_sha256"}
        or not isinstance(experiment.get("invariant"), dict)
        or not isinstance(experiment.get("variant"), dict)
        or any(
            not isinstance(experiment.get(field), str)
            or HEX_DIGEST_RE.fullmatch(experiment[field]) is None
            for field in ("invariant_sha256", "variant_sha256")
        )
    ):
        return False
    if payload.get("primary_metric") not in metric_names:
        return False
    if (
        not isinstance(binding, dict)
        or set(binding) != {"role", "baseline_receipt", "baseline_sha256"}
        or binding.get("role") != "candidate"
        or not isinstance(binding.get("baseline_receipt"), str)
        or not binding["baseline_receipt"]
        or not isinstance(binding.get("baseline_sha256"), str)
        or HEX_DIGEST_RE.fullmatch(binding["baseline_sha256"]) is None
    ):
        return False
    if (
        not isinstance(enabled_methods, list)
        or not enabled_methods
        or any(not isinstance(method, str) or not method for method in enabled_methods)
    ):
        return False

    def canonical_digest(item: Any) -> str:
        return hashlib.sha256(
            json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    workload_descriptor = {
        "id": workload["id"],
        "artifacts": workload["artifacts"],
        "parameters": workload["parameters"],
    }
    return (
        target["sha256"] == canonical_digest(target["descriptor"])
        and workload["sha256"] == canonical_digest(workload_descriptor)
        and experiment["invariant_sha256"] == canonical_digest(experiment["invariant"])
        and experiment["variant_sha256"] == canonical_digest(experiment["variant"])
        and experiment["invariant"].get("primary_metric") == payload["primary_metric"]
        and experiment["variant"].get("enabled_methods") == enabled_methods
        and digest == canonical_digest(payload)
    )


def validate_source_registry(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise SkillError("sources registry must contain a sources list")
    source_ids: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            raise SkillError("every source record must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise SkillError("source record is missing id")
        source_ids.append(source_id.strip())
    duplicates = sorted({source_id for source_id in source_ids if source_ids.count(source_id) > 1})
    if duplicates:
        raise SkillError(f"duplicate source IDs: {', '.join(duplicates)}")
    return {str(source["id"]).strip(): source for source in sources}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assessments(
    report: dict[str, Any],
    *,
    receipt_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if report.get("schema_version") != 1:
        raise SkillError("receipt assessments schema_version must be 1")
    rows = report.get("assessments")
    if not isinstance(rows, list):
        raise SkillError("receipt assessments must contain an assessments list")
    by_receipt: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SkillError("every receipt assessment must be an object")
        receipt = row.get("receipt")
        label = row.get("label")
        digest = row.get("receipt_sha256")
        classification = row.get("classification")
        if not isinstance(receipt, str) or not receipt.strip():
            raise SkillError("receipt assessment is missing receipt")
        receipt = receipt.strip()
        if receipt in by_receipt:
            raise SkillError(f"duplicate receipt assessment: {receipt}")
        if not isinstance(label, str) or not label.strip():
            raise SkillError(f"receipt assessment {receipt} is missing label")
        if not isinstance(digest, str) or HEX_DIGEST_RE.fullmatch(digest) is None:
            raise SkillError(f"receipt assessment {receipt} has invalid receipt_sha256")
        if receipt_root is not None:
            root = receipt_root.resolve()
            receipt_path = (root / receipt).resolve()
            try:
                receipt_path.relative_to(root)
            except ValueError as exc:
                raise SkillError(f"receipt assessment {receipt} escapes the benchmark directory") from exc
            if not receipt_path.is_file() or _file_sha256(receipt_path) != digest:
                raise SkillError(
                    f"receipt assessment {receipt} receipt_sha256 does not match the receipt artifact"
                )
        if classification not in ASSESSMENT_CLASSIFICATIONS:
            raise SkillError(f"receipt assessment {receipt} has invalid classification {classification!r}")
        enabled_methods = _string_list(
            row.get("enabled_methods"),
            field=f"receipt assessment {receipt} enabled_methods",
        )
        reasons = _string_list(row.get("reasons"), field=f"receipt assessment {receipt} reasons")
        gates = row.get("gates")
        if not isinstance(gates, dict):
            raise SkillError(f"receipt assessment {receipt} gates must be an object")
        missing_gates = sorted(REQUIRED_BENCHMARK_GATES - set(gates))
        if missing_gates:
            raise SkillError(
                f"receipt assessment {receipt} is missing required gates: {', '.join(missing_gates)}"
            )
        invalid_gates = sorted(
            gate for gate in REQUIRED_BENCHMARK_GATES if not isinstance(gates.get(gate), bool)
        )
        if invalid_gates:
            raise SkillError(
                f"receipt assessment {receipt} has non-boolean gates: {', '.join(invalid_gates)}"
            )
        promotion_ready = row.get("promotion_ready")
        if not isinstance(promotion_ready, bool):
            raise SkillError(f"receipt assessment {receipt} promotion_ready must be boolean")
        fingerprint = row.get("experiment_fingerprint")
        if fingerprint is not None and not experiment_fingerprint_valid(fingerprint):
            raise SkillError(f"receipt assessment {receipt} has invalid experiment_fingerprint")
        if fingerprint is not None:
            fingerprint_payload = fingerprint["payload"]
            fingerprint_binding = fingerprint_payload["candidate_baseline_binding"]
            baseline = row.get("baseline")
            baseline_file = baseline.get("file") if isinstance(baseline, dict) else None
            if fingerprint_payload["enabled_methods"] != enabled_methods:
                raise SkillError(
                    f"receipt assessment {receipt} experiment_fingerprint enabled_methods mismatch"
                )
            if fingerprint_payload["candidate_receipt_sha256"] != digest:
                raise SkillError(
                    f"receipt assessment {receipt} experiment_fingerprint receipt_sha256 mismatch"
                )
            if fingerprint_payload["primary_metric"] != row.get("primary_metric"):
                raise SkillError(
                    f"receipt assessment {receipt} experiment_fingerprint primary_metric mismatch"
                )
            if fingerprint_binding["baseline_receipt"] != baseline_file:
                raise SkillError(
                    f"receipt assessment {receipt} experiment_fingerprint baseline mismatch"
                )
            if (
                not isinstance(baseline, dict)
                or fingerprint_binding["baseline_sha256"] != baseline.get("expected_sha256")
                or fingerprint_binding["baseline_sha256"] != baseline.get("actual_sha256")
            ):
                raise SkillError(
                    f"receipt assessment {receipt} experiment_fingerprint baseline digest mismatch"
                )
        by_receipt[receipt] = {
            **row,
            "receipt": receipt,
            "enabled_methods": enabled_methods,
            "reasons": reasons,
        }
    return by_receipt


def load_optional_assessments(path: Path) -> tuple[dict[str, Any], bool]:
    """Read the generated sidecar once; a concurrently absent file means no promotion."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"schema_version": 1, "assessments": []}, False
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillError(f"could not parse receipt assessments {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SkillError("receipt assessments root must be an object")
    return value, True


def evidence_source_ids(method: dict[str, Any], known_sources: dict[str, dict[str, Any]]) -> list[str]:
    method_id = str(method.get("id", "<missing-id>"))
    collected: list[str] = []
    evidence_refs = method.get("evidence_refs", {})
    if not isinstance(evidence_refs, dict):
        raise SkillError(f"method {method_id} evidence_refs must be an object")
    for evidence_class in sorted(evidence_refs):
        collected.extend(
            _string_list(
                evidence_refs[evidence_class],
                field=f"method {method_id} evidence_refs.{evidence_class}",
            )
        )
    constraints = method.get("target_constraints")
    if constraints is not None and not isinstance(constraints, dict):
        raise SkillError(f"method {method_id} target_constraints must be an object")
    if isinstance(constraints, dict) and "source_ids" in constraints:
        collected.extend(
            _string_list(
                constraints["source_ids"],
                field=f"method {method_id} target_constraints.source_ids",
            )
        )
    missing = sorted(set(collected) - set(known_sources))
    if missing:
        raise SkillError(f"method {method_id} references missing evidence source {missing[0]}")
    return sorted(set(collected))


def source_reported_constraint_reasons(
    method: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> list[str]:
    constraints = method.get("target_constraints")
    target_source_ids = constraints.get("source_ids", []) if isinstance(constraints, dict) else []
    if not isinstance(target_source_ids, list) or not target_source_ids:
        return ["source-claim-missing-target-sources"]
    reasons: list[str] = []
    has_performance_evidence = False
    allowed_scopes = {"paper_only", "third_party_pinned", "official_mlx", "official_mlx_project"}
    for source_id in target_source_ids:
        source = sources[str(source_id)]
        if source.get("review_depth") != "synthesized":
            reasons.append(f"source-claim-not-synthesized:{source_id}")
        if not isinstance(source.get("snapshot"), str) or not source["snapshot"].strip():
            reasons.append(f"source-claim-not-snapshot-pinned:{source_id}")
        if not isinstance(source.get("evidence_class"), str) or not source["evidence_class"].strip():
            reasons.append(f"source-claim-missing-evidence-class:{source_id}")
        support_scope = source.get("support_scope")
        if support_scope not in allowed_scopes:
            reasons.append(f"source-claim-invalid-scope:{source_id}:{support_scope}")
        claim_types = source.get("claim_types")
        if not isinstance(claim_types, list) or not claim_types:
            reasons.append(f"source-claim-missing-claim-types:{source_id}")
        elif "performance" in claim_types:
            has_performance_evidence = True
    if not has_performance_evidence:
        reasons.append("source-claim-missing-performance-evidence")
    return reasons


def canonical_constraint_reasons(
    method: dict[str, Any],
    source_ids: list[str],
    lineages: list[str],
) -> list[str]:
    reasons: list[str] = []
    constraints = method.get("target_constraints")
    if not source_ids:
        reasons.append("missing-evidence-source-ids")
    if not lineages:
        reasons.append("missing-evidence-lineage-ids")
    if not isinstance(constraints, dict) or not constraints:
        reasons.append("missing-target-constraints")
        return reasons
    required_fields = constraints.get("required_profile_fields")
    if not isinstance(required_fields, list) or not required_fields:
        reasons.append("missing-required-profile-fields")
    target_sources = constraints.get("source_ids")
    if not isinstance(target_sources, list) or not target_sources:
        reasons.append("missing-target-source-ids")
    return reasons


def _format_ratio(value: float) -> str:
    rendered = format(value, ".6g")
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "x"


def receipt_links(
    method_id: str,
    metric: Any,
    receipts: list[str],
    assessments: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], str | None, dict[str, Any] | None]:
    links_by_receipt: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    expected_baselines: dict[str, str] = {}
    promotion_candidates: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    promotion_fingerprints: dict[str, dict[str, Any]] = {}
    promotion_identities: dict[str, dict[str, Any]] = {}
    promotion_candidate_count = 0
    candidate_set_blocked = False
    candidate_regression_present = False
    metric_policy = LOCAL_METRIC_RATIOS.get(str(metric))
    if metric_policy is None:
        reasons.append(f"unsupported-local-metric:{metric}")
        candidate_set_blocked = True

    for receipt in sorted(receipts):
        assessment = assessments.get(receipt)
        if assessment is None:
            links_by_receipt[receipt] = {
                "receipt": receipt,
                "role": "missing",
                "classification": "missing",
                "promotion_ready": False,
                "receipt_sha256": None,
                "method_enabled": False,
                "required_gates_passed": False,
                "experiment_fingerprint": None,
            }
            reasons.append(f"receipt-assessment-missing:{receipt}")
            continue
        method_enabled = method_id in assessment["enabled_methods"]
        candidate_like = (
            assessment["classification"] == "promotion_ready"
            or assessment["promotion_ready"] is True
            or isinstance(assessment.get("baseline"), dict)
            or method_enabled
        )
        if not candidate_like:
            continue
        failed_gates = sorted(
            gate for gate in REQUIRED_BENCHMARK_GATES if assessment["gates"].get(gate) is not True
        )
        links_by_receipt[receipt] = {
            "receipt": receipt,
            "role": "candidate",
            "classification": assessment["classification"],
            "promotion_ready": assessment["promotion_ready"],
            "receipt_sha256": assessment["receipt_sha256"],
            "method_enabled": method_enabled,
            "required_gates_passed": not failed_gates,
            "experiment_fingerprint": assessment.get("experiment_fingerprint"),
        }
        if assessment["classification"] != "promotion_ready":
            reasons.append(f"receipt-not-promotion-ready:{receipt}")
            candidate_set_blocked = True
        if assessment["promotion_ready"] is not True:
            reasons.append(f"receipt-promotion-flag-false:{receipt}")
            candidate_set_blocked = True
        if assessment["reasons"]:
            reasons.append(f"receipt-has-assessment-reasons:{receipt}")
            candidate_set_blocked = True
        if failed_gates:
            reasons.append(f"receipt-required-gates-failed:{receipt}:{','.join(failed_gates)}")
            candidate_set_blocked = True
        if not method_enabled:
            reasons.append(f"receipt-method-link-missing:{receipt}")
            candidate_set_blocked = True
        promotable_candidate = (
            assessment["classification"] == "promotion_ready"
            and assessment["promotion_ready"] is True
            and method_enabled
        )
        if promotable_candidate:
            promotion_candidate_count += 1
            fingerprint = assessment.get("experiment_fingerprint")
            if not experiment_fingerprint_valid(fingerprint):
                reasons.append(f"candidate-experiment-fingerprint-missing:{receipt}")
                candidate_set_blocked = True
            else:
                identity = experiment_identity_from_fingerprint(fingerprint)
                if identity is None:
                    reasons.append(f"candidate-experiment-identity-invalid:{receipt}")
                    candidate_set_blocked = True
                else:
                    promotion_fingerprints[receipt] = fingerprint
                    promotion_identities[identity["sha256"]] = identity

        baseline = assessment.get("baseline")
        baseline_file = baseline.get("file") if isinstance(baseline, dict) else None
        baseline_label = baseline.get("label") if isinstance(baseline, dict) else None
        if (
            not isinstance(baseline_file, str)
            or not baseline_file
            or not isinstance(baseline_label, str)
            or not baseline_label
        ):
            reasons.append(f"candidate-baseline-missing:{receipt}")
            candidate_set_blocked = True
        elif baseline.get("compatible") is not True:
            reasons.append(f"candidate-baseline-incompatible:{receipt}")
            candidate_set_blocked = True
        elif baseline_file == receipt or baseline_file not in receipts:
            reasons.append(f"candidate-baseline-not-referenced:{receipt}:{baseline_file}")
            candidate_set_blocked = True
        elif baseline_file in expected_baselines and expected_baselines[baseline_file] != baseline_label:
            reasons.append(f"candidate-baseline-label-conflict:{receipt}:{baseline_file}")
            candidate_set_blocked = True
        else:
            expected_baselines[baseline_file] = baseline_label

        if metric_policy is not None:
            ratio_key, primary_metric = metric_policy
            if assessment.get("primary_metric") != primary_metric:
                reasons.append(f"candidate-primary-metric-mismatch:{receipt}")
                candidate_set_blocked = True
            ratios = assessment.get("recomputed_median_ratios")
            ratio = ratios.get(ratio_key) if isinstance(ratios, dict) else None
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(float(ratio))
                or float(ratio) <= 0
            ):
                reasons.append(f"candidate-ratio-missing:{receipt}:{ratio_key}")
                candidate_set_blocked = True
            else:
                if promotable_candidate and receipt in promotion_fingerprints:
                    identity = experiment_identity_from_fingerprint(promotion_fingerprints[receipt])
                    if identity is not None:
                        promotion_candidates.append(
                            (float(ratio), receipt, promotion_fingerprints[receipt], identity)
                        )
                if float(ratio) <= 1.0:
                    reasons.append("candidate-no-improvement")
                    candidate_set_blocked = True
                    candidate_regression_present = True

    if promotion_candidate_count == 0:
        reasons.append("missing-promotion-ready-candidate")

    for receipt in sorted(receipts):
        if receipt in links_by_receipt:
            continue
        assessment = assessments[receipt]
        if receipt in expected_baselines:
            label_matches = assessment.get("label") == expected_baselines[receipt]
            baseline_shape = (
                assessment["classification"] == "performance_observation"
                and assessment["promotion_ready"] is False
                and assessment["enabled_methods"] == []
                and label_matches
            )
            links_by_receipt[receipt] = {
                "receipt": receipt,
                "role": "baseline",
                "classification": assessment["classification"],
                "promotion_ready": assessment["promotion_ready"],
                "receipt_sha256": assessment["receipt_sha256"],
                "method_enabled": False,
                "required_gates_passed": False,
                "experiment_fingerprint": assessment.get("experiment_fingerprint"),
            }
            if not baseline_shape:
                reasons.append(f"invalid-baseline-assessment:{receipt}")
                candidate_set_blocked = True
        else:
            links_by_receipt[receipt] = {
                "receipt": receipt,
                "role": "unrelated",
                "classification": assessment["classification"],
                "promotion_ready": assessment["promotion_ready"],
                "receipt_sha256": assessment["receipt_sha256"],
                "method_enabled": method_id in assessment["enabled_methods"],
                "required_gates_passed": False,
                "experiment_fingerprint": assessment.get("experiment_fingerprint"),
            }
            kind = "rejected" if assessment["classification"] == "rejected" else "observation"
            reasons.append(f"unrelated-receipt-{kind}:{receipt}")

    experiment_fingerprint: dict[str, Any] | None = None
    if len(promotion_identities) > 1:
        reasons.append("heterogeneous-promotion-candidate-fingerprints")
        candidate_set_blocked = True

    effective_range: str | None = None
    if promotion_candidates and not candidate_set_blocked:
        conservative_ratio, _, experiment_fingerprint, _ = min(
            promotion_candidates,
            key=lambda item: (item[0], item[1]),
        )
        if conservative_ratio <= 1.0:
            reasons.append("candidate-no-improvement")
        else:
            effective_range = f"1.0x-{_format_ratio(conservative_ratio)}"
    if candidate_set_blocked:
        reasons.append(
            "candidate-regression-present"
            if candidate_regression_present
            else "candidate-set-incomplete"
        )
        effective_range = None
        experiment_fingerprint = None
    return (
        [links_by_receipt[receipt] for receipt in sorted(links_by_receipt)],
        reasons,
        effective_range,
        experiment_fingerprint,
    )


def build_claim_catalog(
    guidance: dict[str, Any],
    sources: dict[str, Any],
    assessment_report: dict[str, Any],
    *,
    assessments_available: bool,
    receipt_root: Path | None = None,
) -> dict[str, Any]:
    methods = guidance.get("methods")
    if not isinstance(methods, list):
        raise SkillError("optimization guidance must contain a methods list")
    known_sources = validate_source_registry(sources)
    assessments = validate_assessments(assessment_report, receipt_root=receipt_root)

    method_ids: list[str] = []
    lineage_owners: dict[str, str] = {}
    claims: list[dict[str, Any]] = []
    for method in methods:
        if not isinstance(method, dict) or "improvement_band" not in method:
            continue
        method_id = method.get("id")
        if not isinstance(method_id, str) or not method_id.strip():
            raise SkillError("improvement-band method is missing id")
        method_id = method_id.strip()
        method_ids.append(method_id)
        band = method.get("improvement_band")
        if not isinstance(band, dict):
            raise SkillError(f"method {method_id} improvement_band must be an object")
        observed_range = band.get("range")
        if observed_range is None:
            observed_range = band.get("observed_source_range")
        if observed_range is not None and (not isinstance(observed_range, str) or not observed_range.strip()):
            raise SkillError(f"method {method_id} improvement_band range must be a non-empty string or null")
        observed_range = observed_range.strip() if isinstance(observed_range, str) else None
        lineages = _string_list(
            band.get("evidence_lineage_ids", []),
            field=f"method {method_id} improvement_band.evidence_lineage_ids",
        )
        claim_status = str(band.get("claim_status", "unclassified"))
        method_status = str(method.get("status", "unclassified"))
        if claim_status not in {"superseded"} and method_status not in INACTIVE_METHOD_STATUSES:
            for lineage in lineages:
                owner = lineage_owners.get(lineage)
                if owner is not None:
                    raise SkillError(
                        f"evidence lineage {lineage} is reused by active methods {owner} and {method_id}"
                    )
                lineage_owners[lineage] = method_id

        source_ids = evidence_source_ids(method, known_sources)
        receipts = _string_list(
            band.get("receipts", []),
            field=f"method {method_id} improvement_band.receipts",
        )
        provenance = str(band.get("provenance", "unclassified"))
        links, receipt_reasons, derived_local_range, derived_fingerprint = receipt_links(
            method_id,
            band.get("metric"),
            receipts,
            assessments,
        ) if receipts else ([], [], None, None)
        profile_eligible_range: str | None = None
        effective_range: str | None = None
        promotion_state = "withheld"
        withheld_reasons: list[str] = []

        if provenance == "profile_required":
            withheld_reasons.append("profile-required-provenance")
        if provenance == "source_reported":
            withheld_reasons.append("source-reported-range-requires-local-reproduction")
        if provenance == "performance_observation":
            withheld_reasons.append("performance-observation-not-promotable")
        if claim_status == "held":
            withheld_reasons.append("claim-status-held")
        if claim_status == "superseded" or method_status in INACTIVE_METHOD_STATUSES:
            withheld_reasons.append("claim-status-superseded")

        is_inactive = (
            provenance in {"profile_required", "performance_observation"}
            or claim_status in INACTIVE_CLAIM_STATUSES
            or method_status in INACTIVE_METHOD_STATUSES
        )
        canonical_reasons = canonical_constraint_reasons(method, source_ids, lineages)
        if not is_inactive and provenance == "source_reported" and claim_status == "eligible-with-profile":
            if observed_range is None:
                withheld_reasons.append("missing-observed-range")
            withheld_reasons.extend(canonical_reasons)
            withheld_reasons.extend(source_reported_constraint_reasons(method, known_sources))
        elif not is_inactive and provenance == "local_reproduced":
            if not receipts:
                withheld_reasons.append("missing-receipts")
            if not assessments_available:
                withheld_reasons.append("receipt-assessments-unavailable")
            withheld_reasons.extend(canonical_reasons)
            withheld_reasons.extend(receipt_reasons)
            if not withheld_reasons:
                effective_range = derived_local_range
                promotion_state = "local-promotion"
        elif not is_inactive:
            withheld_reasons.append(f"unsupported-claim-state:{provenance}:{claim_status}")

        claims.append({
            "method_id": method_id,
            "method_status": method_status,
            "provenance": provenance,
            "claim_status": claim_status,
            "metric": band.get("metric"),
            "observed_range": observed_range,
            "profile_eligible_range": profile_eligible_range,
            "effective_range": effective_range,
            "experiment_fingerprint": derived_fingerprint if promotion_state == "local-promotion" else None,
            "promotion_state": promotion_state,
            "evidence_source_ids": source_ids,
            "evidence_lineage_ids": sorted(lineages),
            "target_constraints": _canonical(method.get("target_constraints")),
            "receipt_assessments": links,
            "withheld_reasons": sorted(set(withheld_reasons)),
        })

    duplicate_methods = sorted({method_id for method_id in method_ids if method_ids.count(method_id) > 1})
    if duplicate_methods:
        raise SkillError(f"duplicate improvement-band method IDs: {', '.join(duplicate_methods)}")
    claims.sort(key=lambda claim: claim["method_id"])
    return {
        "schema_version": 1,
        "guidance_schema_version": guidance.get("schema_version"),
        "receipt_assessment_state": "available" if assessments_available else "unavailable",
        "claim_count": len(claims),
        "claims": claims,
    }


def render_catalog(catalog: dict[str, Any]) -> str:
    return json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main() -> int:
    args = parse_args()
    try:
        guidance = load_structured(args.guidance)
        sources = load_structured(args.sources)
        assessments, assessments_available = load_optional_assessments(Path(args.assessments))
        if not isinstance(guidance, dict) or not isinstance(sources, dict):
            raise SkillError("guidance and sources roots must be objects")
        catalog = build_claim_catalog(
            guidance,
            sources,
            assessments,
            assessments_available=assessments_available,
            receipt_root=Path(args.assessments).resolve().parent if assessments_available else None,
        )
        rendered = render_catalog(catalog)
        output = Path(args.output)
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
                print(f"error: claim catalog drift: regenerate {output}", file=sys.stderr)
                return 1
            print(f"claim catalog is current: {output}")
            return 0
        atomic_write_text(output, rendered)
        print(f"wrote {catalog['claim_count']} effective-claim records to {output}")
        return 0
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
