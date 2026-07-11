#!/usr/bin/env python3
"""Recommend evidence-backed MLX optimization experiments for an inspection report."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from _common import SkillError, compose_stack_band, dump_json, load_structured
from generate_claim_catalog import build_claim_catalog, experiment_fingerprint_valid
from validate_benchmarks import build_assessment_report

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_RECEIPT_ASSESSMENTS = SKILL_ROOT / "assets" / "benchmarks" / "receipt_assessments.json"
DEFAULT_EFFECTIVE_CLAIMS = SKILL_ROOT / "assets" / "effective_claims.json"
DEFAULT_SOURCES = SKILL_ROOT / "assets" / "sources.yaml"
DEFAULT_KNOWLEDGE_GRAPH = SKILL_ROOT / "assets" / "knowledge_graph.json"
MAX_KNOWLEDGE_GRAPH_BYTES = 2 * 1024 * 1024
MAX_KNOWLEDGE_GRAPH_NODES = 10_000
MAX_KNOWLEDGE_GRAPH_EDGES = 20_000
GRAPH_NODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,511}$")
GRAPH_SIGNAL_RELATIONS = (
    "candidate_relevant_to",
    "evidence_for",
    "evidence_for_outcome",
    "candidate_version_of",
)
GRAPH_ALLOWED_RELATIONS = {*GRAPH_SIGNAL_RELATIONS, "contributor_in_refresh"}
STATUS_RANK = {
    "native-mlx": 0,
    "official-mlx-project": 1,
    "proven-mlx-port": 2,
    "research-candidate": 3,
    "rejected-or-superseded": 4,
}
NUMERIC_MULTIPLIER_RE = re.compile(r"\b\d+(?:\.\d+)?x\b", re.IGNORECASE)
TRUSTED_INSPECTION_SCHEMA_VERSION = 1
TRUSTED_INSPECTION_MODE = "static-no-model-import"
TRUSTED_INSPECTION_FIELDS = {
    "schema_version",
    "generated_at",
    "inspection_mode",
    "source",
    "artifact_identity",
    "local_path",
    "configs",
    "file_summary",
    "tensor_summary",
    "tensors",
    "source_format_summary",
    "architecture_candidates",
    "architecture_profile",
    "architecture_traits",
    "routing_decision",
    "recommended_family",
    "recommended_runbook",
    "recommended_families",
    "recommended_runbooks",
    "recommendation_blockers",
    "license",
    "risks",
    "limitations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend MLX optimization candidates")
    parser.add_argument("inspection")
    parser.add_argument("--guidance", default=str(SKILL_ROOT / "assets" / "optimization_guidance.yaml"))
    parser.add_argument("--stacks", default=str(SKILL_ROOT / "assets" / "optimization_stacks.yaml"))
    parser.add_argument("--taxonomy", default=str(SKILL_ROOT / "assets" / "recommendation-taxonomy.yaml"))
    parser.add_argument("--architectures", default=str(SKILL_ROOT / "assets" / "architectures.yaml"))
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument(
        "--knowledge-graph",
        default=str(DEFAULT_KNOWLEDGE_GRAPH),
        help="Review-only knowledge graph; malformed or oversized graphs are reported as unavailable",
    )
    parser.add_argument(
        "--effective-claims",
        default=str(DEFAULT_EFFECTIVE_CLAIMS),
        help="Generated effective-claim catalog; must exactly match guidance and assessments",
    )
    parser.add_argument(
        "--receipt-assessments",
        help=(
            "Canonical receipt_assessments.json colocated with its benchmark receipts; "
            "the report is recomputed before use"
        ),
    )
    parser.add_argument("--target-profile", help="TargetProfile JSON/YAML with hardware, software, capabilities, and workloads")
    parser.add_argument("--family", help="Override detected architecture family")
    parser.add_argument("--objective", action="append", default=[], help="Filter by objective tag; repeatable")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", help="Write JSON recommendation report")
    parser.add_argument("--markdown", help="Write a compact Markdown shortlist")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Surface recommendations even when intake risks block the port (default: hold them).",
    )
    return parser.parse_args()


def _normalized(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _path_value(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _require_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise SkillError(
            f"Untrusted inspection report: {label} is missing mandatory fields: "
            + ", ".join(missing)
        )


def _string_list_field(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SkillError(f"Untrusted inspection report: {label} must be a list of strings")
    return value


def _nonnegative_int_field(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SkillError(f"Untrusted inspection report: {label} must be a non-negative integer")
    return value


def validate_trusted_inspection(inspection: Any) -> dict[str, Any]:
    """Validate the complete inspect_model.py safety contract before downstream use."""
    if not isinstance(inspection, dict):
        raise SkillError("Untrusted inspection report: root must be an object")
    _require_fields(inspection, TRUSTED_INSPECTION_FIELDS, "root")
    if inspection.get("schema_version") != TRUSTED_INSPECTION_SCHEMA_VERSION:
        raise SkillError(
            "Untrusted inspection report: schema_version must be "
            f"{TRUSTED_INSPECTION_SCHEMA_VERSION}"
        )
    if inspection.get("inspection_mode") != TRUSTED_INSPECTION_MODE:
        raise SkillError(
            f"Untrusted inspection report: inspection_mode must be {TRUSTED_INSPECTION_MODE!r}"
        )
    if not isinstance(inspection.get("generated_at"), str) or not inspection["generated_at"]:
        raise SkillError("Untrusted inspection report: generated_at must be a non-empty string")
    if not isinstance(inspection.get("local_path"), str):
        raise SkillError("Untrusted inspection report: local_path must be a string")

    source = inspection.get("source")
    if not isinstance(source, dict):
        raise SkillError("Untrusted inspection report: source must be an object")
    _require_fields(source, {"kind", "input", "revision"}, "source")
    if source.get("kind") not in {"local", "huggingface"}:
        raise SkillError("Untrusted inspection report: source.kind is invalid")
    if not isinstance(source.get("input"), str) or not source["input"]:
        raise SkillError("Untrusted inspection report: source.input must be a non-empty string")
    if source.get("revision") is not None and not isinstance(source.get("revision"), str):
        raise SkillError("Untrusted inspection report: source.revision must be a string or null")
    if source["kind"] == "huggingface" and (
        not isinstance(source.get("revision"), str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]) is None
    ):
        raise SkillError("Untrusted inspection report: Hugging Face source revision must be a pinned commit")

    artifact_identity = inspection.get("artifact_identity")
    if not isinstance(artifact_identity, dict):
        raise SkillError("Untrusted inspection report: artifact_identity must be an object")
    _require_fields(
        artifact_identity,
        {
            "schema_version", "algorithm", "status", "immutable", "fingerprint",
            "file_count", "total_bytes", "manifest", "errors",
        },
        "artifact_identity",
    )
    if artifact_identity.get("schema_version") != 1:
        raise SkillError("Untrusted inspection report: artifact_identity.schema_version must be 1")
    if artifact_identity.get("algorithm") != "sha256-tree-v1":
        raise SkillError("Untrusted inspection report: artifact_identity.algorithm is invalid")
    if artifact_identity.get("status") not in {"verified", "incomplete"}:
        raise SkillError("Untrusted inspection report: artifact_identity.status is invalid")
    if not isinstance(artifact_identity.get("immutable"), bool):
        raise SkillError("Untrusted inspection report: artifact_identity.immutable must be boolean")
    manifest = artifact_identity.get("manifest")
    if not isinstance(manifest, list):
        raise SkillError("Untrusted inspection report: artifact_identity.manifest must be a list")
    identity_errors = _string_list_field(
        artifact_identity.get("errors"),
        "artifact_identity.errors",
    )
    file_count = _nonnegative_int_field(
        artifact_identity.get("file_count"),
        "artifact_identity.file_count",
    )
    total_bytes = _nonnegative_int_field(
        artifact_identity.get("total_bytes"),
        "artifact_identity.total_bytes",
    )
    normalized_manifest: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(manifest):
        label = f"artifact_identity.manifest[{index}]"
        if not isinstance(record, dict):
            raise SkillError(f"Untrusted inspection report: {label} must be an object")
        _require_fields(record, {"path", "size_bytes", "sha256"}, label)
        path = record.get("path")
        digest = record.get("sha256")
        normalized_path = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or normalized_path is None
            or normalized_path.is_absolute()
            or "\\" in path
            or ".." in normalized_path.parts
            or normalized_path.as_posix() != path
        ):
            raise SkillError(f"Untrusted inspection report: {label}.path is not portable")
        if path in seen_paths:
            raise SkillError(f"Untrusted inspection report: duplicate artifact path {path!r}")
        seen_paths.add(path)
        size_bytes = _nonnegative_int_field(record.get("size_bytes"), f"{label}.size_bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SkillError(f"Untrusted inspection report: {label}.sha256 is invalid")
        normalized_manifest.append({"path": path, "size_bytes": size_bytes, "sha256": digest})
    if normalized_manifest != sorted(normalized_manifest, key=lambda record: record["path"]):
        raise SkillError("Untrusted inspection report: artifact_identity.manifest must be path-sorted")
    if file_count != len(normalized_manifest):
        raise SkillError("Untrusted inspection report: artifact_identity.file_count does not match manifest")
    if total_bytes != sum(record["size_bytes"] for record in normalized_manifest):
        raise SkillError("Untrusted inspection report: artifact_identity.total_bytes does not match manifest")
    canonical_identity = json.dumps(
        {"schema_version": 1, "files": normalized_manifest},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected_fingerprint = "sha256:" + hashlib.sha256(canonical_identity).hexdigest()
    identity_verified = (
        artifact_identity["status"] == "verified"
        and artifact_identity["immutable"] is True
        and bool(normalized_manifest)
        and not identity_errors
        and artifact_identity.get("fingerprint") == expected_fingerprint
    )
    if artifact_identity["status"] == "verified" and not identity_verified:
        raise SkillError("Untrusted inspection report: verified artifact identity is internally inconsistent")
    if artifact_identity["status"] == "incomplete" and (
        artifact_identity["immutable"] is not False
        or artifact_identity.get("fingerprint") is not None
    ):
        raise SkillError("Untrusted inspection report: incomplete artifact identity is internally inconsistent")

    if not isinstance(inspection.get("configs"), dict):
        raise SkillError("Untrusted inspection report: configs must be an object")
    if inspection.get("architecture_profile") is not None and not isinstance(
        inspection.get("architecture_profile"),
        dict,
    ):
        raise SkillError("Untrusted inspection report: architecture_profile must be an object or null")
    for field in ("architecture_candidates", "architecture_traits", "tensors", "risks", "limitations"):
        if not isinstance(inspection.get(field), list):
            raise SkillError(f"Untrusted inspection report: {field} must be a list")

    file_summary = inspection.get("file_summary")
    if not isinstance(file_summary, dict):
        raise SkillError("Untrusted inspection report: file_summary must be an object")
    _require_fields(file_summary, {"count", "truncated", "extensions", "files"}, "file_summary")
    _nonnegative_int_field(file_summary.get("count"), "file_summary.count")
    if not isinstance(file_summary.get("truncated"), bool):
        raise SkillError("Untrusted inspection report: file_summary.truncated must be boolean")
    if not isinstance(file_summary.get("files"), list):
        raise SkillError("Untrusted inspection report: file_summary.files must be a list")
    if file_summary["count"] != len(file_summary["files"]):
        raise SkillError("Untrusted inspection report: file_summary.count does not match files")
    for index, record in enumerate(file_summary["files"]):
        label = f"file_summary.files[{index}]"
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise SkillError(f"Untrusted inspection report: {label} must contain a path")
        if identity_verified and record["path"] not in seen_paths:
            raise SkillError(
                f"Untrusted inspection report: {label} is not bound to artifact_identity"
            )

    tensor_summary = inspection.get("tensor_summary")
    if not isinstance(tensor_summary, dict):
        raise SkillError("Untrusted inspection report: tensor_summary must be an object")
    _require_fields(
        tensor_summary,
        {"count", "parameters", "estimated_bytes", "dtypes", "files", "metadata", "errors", "integrity_ok"},
        "tensor_summary",
    )
    tensor_errors = _string_list_field(tensor_summary.get("errors"), "tensor_summary.errors")
    if not isinstance(tensor_summary.get("integrity_ok"), bool):
        raise SkillError("Untrusted inspection report: tensor_summary.integrity_ok must be boolean")

    source_formats = inspection.get("source_format_summary")
    if not isinstance(source_formats, dict):
        raise SkillError("Untrusted inspection report: source_format_summary must be an object")
    _require_fields(
        source_formats,
        {"count", "formats", "errors", "integrity_ok", "aggregate", "manifests"},
        "source_format_summary",
    )
    source_errors = _string_list_field(source_formats.get("errors"), "source_format_summary.errors")
    if not isinstance(source_formats.get("integrity_ok"), bool):
        raise SkillError("Untrusted inspection report: source_format_summary.integrity_ok must be boolean")

    routing = inspection.get("routing_decision")
    if not isinstance(routing, dict):
        raise SkillError("Untrusted inspection report: routing_decision must be an object")
    _require_fields(
        routing,
        {
            "status", "minimum_score", "minimum_margin", "winner_family", "winner_score",
            "runner_up_family", "runner_up_score", "winner_margin", "reasons",
        },
        "routing_decision",
    )
    if routing.get("status") not in {"recommended", "ambiguous"}:
        raise SkillError("Untrusted inspection report: routing_decision.status is invalid")
    _string_list_field(routing.get("reasons"), "routing_decision.reasons")
    candidate_families = {
        str(candidate.get("family"))
        for candidate in inspection["architecture_candidates"]
        if isinstance(candidate, dict) and isinstance(candidate.get("family"), str)
    }
    if routing["status"] == "recommended" and (
        not isinstance(routing.get("winner_family"), str)
        or routing["winner_family"] not in candidate_families
    ):
        raise SkillError("Untrusted inspection report: routing winner is not an architecture candidate")

    blockers = _string_list_field(
        inspection.get("recommendation_blockers"),
        "recommendation_blockers",
    )
    families = _string_list_field(inspection.get("recommended_families"), "recommended_families")
    _string_list_field(inspection.get("recommended_runbooks"), "recommended_runbooks")
    recommended_family = inspection.get("recommended_family")
    recommended_runbook = inspection.get("recommended_runbook")
    if recommended_family is not None and not isinstance(recommended_family, str):
        raise SkillError("Untrusted inspection report: recommended_family must be a string or null")
    if recommended_runbook is not None and not isinstance(recommended_runbook, str):
        raise SkillError("Untrusted inspection report: recommended_runbook must be a string or null")
    if recommended_family is not None and (not families or families[0] != recommended_family):
        raise SkillError(
            "Untrusted inspection report: recommended_family must be the first recommended_families entry"
        )
    unknown_recommended_families = sorted(set(families) - candidate_families)
    if unknown_recommended_families:
        raise SkillError(
            "Untrusted inspection report: recommended families are not architecture candidates: "
            + ", ".join(unknown_recommended_families)
        )

    license_info = inspection.get("license")
    if not isinstance(license_info, dict):
        raise SkillError("Untrusted inspection report: license must be an object")
    _require_fields(
        license_info,
        {
            "declared", "license_files", "file_evidence", "accepted_evidence", "status",
            "requires_review", "compatibility_assessed", "reasons",
        },
        "license",
    )
    for field in ("declared", "file_evidence", "accepted_evidence"):
        if not isinstance(license_info.get(field), list):
            raise SkillError(f"Untrusted inspection report: license.{field} must be a list")
    _string_list_field(license_info.get("license_files"), "license.license_files")
    _string_list_field(license_info.get("reasons"), "license.reasons")
    if license_info.get("status") not in {"acceptable-evidence", "review-required"}:
        raise SkillError("Untrusted inspection report: license.status is invalid")
    if not isinstance(license_info.get("requires_review"), bool):
        raise SkillError("Untrusted inspection report: license.requires_review must be boolean")
    if not isinstance(license_info.get("compatibility_assessed"), bool):
        raise SkillError("Untrusted inspection report: license.compatibility_assessed must be boolean")
    if license_info["compatibility_assessed"] is not False:
        raise SkillError("Untrusted inspection report: static intake cannot claim license compatibility")
    for index, evidence in enumerate(license_info["accepted_evidence"]):
        label = f"license.accepted_evidence[{index}]"
        if not isinstance(evidence, dict):
            raise SkillError(f"Untrusted inspection report: {label} must be an object")
        _require_fields(evidence, {"kind", "source", "value"}, label)
        if evidence.get("kind") not in {"declaration", "license-file"}:
            raise SkillError(f"Untrusted inspection report: {label}.kind is invalid")
        if evidence.get("source") not in seen_paths:
            raise SkillError(f"Untrusted inspection report: {label} is not bound to the artifact manifest")
        if not isinstance(evidence.get("value"), str) or not evidence["value"]:
            raise SkillError(f"Untrusted inspection report: {label}.value must be a non-empty string")
    license_acceptable = (
        license_info["status"] == "acceptable-evidence"
        and license_info["requires_review"] is False
        and bool(license_info["accepted_evidence"])
    )
    if license_info["status"] == "acceptable-evidence" and not license_acceptable:
        raise SkillError("Untrusted inspection report: acceptable license evidence is internally inconsistent")
    if license_info["status"] == "review-required" and license_info["requires_review"] is not True:
        raise SkillError("Untrusted inspection report: review-required license must set requires_review")

    unsafe_without_blocker = (
        bool(tensor_errors)
        or tensor_summary["integrity_ok"] is not True
        or bool(source_errors)
        or source_formats["integrity_ok"] is not True
        or file_summary["truncated"] is True
        or not identity_verified
        or routing["status"] != "recommended"
        or not license_acceptable
        or any(
            isinstance(risk, dict) and risk.get("severity") == "high"
            for risk in inspection["risks"]
        )
    )
    if unsafe_without_blocker and not blockers:
        raise SkillError(
            "Untrusted inspection report: safety, provenance, or ambiguity state requires at least one blocker"
        )
    if blockers and (recommended_family is not None or families):
        raise SkillError(
            "Untrusted inspection report: blocked reports must not contain recommended families"
        )
    if not blockers and routing["status"] == "recommended" and recommended_family is None:
        raise SkillError(
            "Untrusted inspection report: an unblocked recommended route must contain recommended_family"
        )
    if recommended_family is not None and recommended_family != routing.get("winner_family"):
        raise SkillError(
            "Untrusted inspection report: recommended_family must match the routing winner"
        )
    return inspection


def trusted_inspection_sha256(inspection: dict[str, Any]) -> str:
    payload = json.dumps(
        inspection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_target_profile(path: str | None, inspection: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    profile = load_structured(path) if path else inspection.get("target_profile")
    if profile is None:
        return None, "missing"
    if not isinstance(profile, dict):
        raise SkillError("TargetProfile must be a mapping")
    if profile.get("schema_version") != 1:
        raise SkillError("TargetProfile schema_version must be 1")
    for section in ("hardware", "software"):
        if section in profile and not isinstance(profile[section], dict):
            raise SkillError(f"TargetProfile {section} must be a mapping")
    for section in ("models", "target", "workload"):
        if section in profile and not isinstance(profile[section], dict):
            raise SkillError(f"TargetProfile {section} must be a mapping")
    for section in ("capabilities", "workloads"):
        if section in profile and not isinstance(profile[section], list):
            raise SkillError(f"TargetProfile {section} must be a list of controlled identifiers")
    fingerprint = profile.get("experiment_fingerprint_sha256")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise SkillError(
            "TargetProfile experiment_fingerprint_sha256 must be a lowercase 64-hex digest"
        )
    full_fingerprint = profile.get("experiment_fingerprint")
    if full_fingerprint is not None and not experiment_fingerprint_valid(full_fingerprint):
        raise SkillError(
            "TargetProfile experiment_fingerprint must be the full canonical receipt-derived fingerprint"
        )
    return profile, "provided"


def validate_target_profile_ids(profile: dict[str, Any] | None, taxonomy: dict[str, Any]) -> None:
    if profile is None:
        return
    allowed_capabilities = {str(value) for value in taxonomy.get("capability_ids", [])}
    allowed_workloads = {str(value) for value in taxonomy.get("workload_ids", [])}
    unknown_capabilities = sorted(
        {str(value) for value in profile.get("capabilities", [])} - allowed_capabilities
    )
    unknown_workloads = sorted(
        {str(value) for value in profile.get("workloads", [])} - allowed_workloads
    )
    if unknown_capabilities:
        raise SkillError("TargetProfile contains unknown capability ids: " + ", ".join(unknown_capabilities))
    if unknown_workloads:
        raise SkillError("TargetProfile contains unknown workload ids: " + ", ".join(unknown_workloads))


def validate_objective_ids(objectives: set[str], taxonomy: dict[str, Any]) -> None:
    allowed = {
        str(item.get("id")).lower()
        for item in taxonomy.get("objective_tags", [])
        if isinstance(item, dict) and item.get("id")
    }
    unknown = sorted(objectives - allowed)
    if unknown:
        raise SkillError("unknown objective ids: " + ", ".join(unknown))


def validate_family_ids(families: list[str], architectures: dict[str, Any]) -> None:
    allowed = {
        str(item.get("id"))
        for item in architectures.get("families", [])
        if isinstance(item, dict) and item.get("id")
    }
    if not allowed:
        raise SkillError("Architecture registry contains no controlled family ids")
    unknown = sorted(set(families) - allowed)
    if unknown:
        raise SkillError("unknown architecture family ids: " + ", ".join(unknown))


def _graph_unavailable(families: list[str], reason: str) -> dict[str, Any]:
    return {
        "label": "Unreviewed research signals (experimental/review queue)",
        "status": "unavailable",
        "review_only": True,
        "distinct_from_advisor_buckets": True,
        "execution_allowed": False,
        "numeric_authority": "assets/effective_claims.json only",
        "families": list(families),
        "reason": reason[:500],
        "relation_counts": {relation: 0 for relation in GRAPH_SIGNAL_RELATIONS},
        "selected_count": 0,
        "items": [],
    }


def _load_bounded_graph_json(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_KNOWLEDGE_GRAPH_BYTES + 1)
    except OSError as exc:
        raise SkillError(f"could not read {path}: {exc}") from exc
    if len(raw) > MAX_KNOWLEDGE_GRAPH_BYTES:
        raise SkillError(
            f"graph exceeds the {MAX_KNOWLEDGE_GRAPH_BYTES}-byte advisory read limit"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError(f"graph is not valid UTF-8: {exc}") from exc

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SkillError(f"graph contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise SkillError(f"graph is not valid JSON: {exc}") from exc


def _graph_plain_text(value: Any, label: str, *, required: bool = False) -> str:
    if value in (None, "") and not required:
        return ""
    if not isinstance(value, str) or (required and not value):
        raise SkillError(f"graph {label} must be a non-empty string")
    if len(value) > 2048 or any(ord(character) < 32 for character in value):
        raise SkillError(f"graph {label} contains invalid text")
    return value


def _graph_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SkillError(f"graph {label} must be a list of non-empty strings")
    return [str(item) for item in value]


def validate_knowledge_graph(graph: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(graph, dict) or graph.get("schema_version") != 1:
        raise SkillError("graph root must be a schema_version 1 object")
    policy = graph.get("policy")
    if not isinstance(policy, dict) or (
        policy.get("review_only") is not True
        or policy.get("auto_promote_sources") is not False
        or policy.get("auto_modify_recommendations") is not False
    ):
        raise SkillError("graph policy must remain review-only and non-promoting")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or len(nodes) > MAX_KNOWLEDGE_GRAPH_NODES:
        raise SkillError("graph nodes must be a bounded list")
    if not isinstance(edges, list) or len(edges) > MAX_KNOWLEDGE_GRAPH_EDGES:
        raise SkillError("graph edges must be a bounded list")
    if graph.get("node_count") != len(nodes) or graph.get("edge_count") != len(edges):
        raise SkillError("graph node_count or edge_count does not match its records")

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise SkillError(f"graph node[{index}] must be an object")
        node_id = _graph_plain_text(node.get("id"), f"node[{index}].id", required=True)
        if GRAPH_NODE_ID_RE.fullmatch(node_id) is None:
            raise SkillError(f"graph node[{index}].id is not a controlled identifier")
        if NUMERIC_MULTIPLIER_RE.search(node_id):
            raise SkillError(f"graph node[{index}].id contains a numeric-claim-like token")
        if node_id in nodes_by_id:
            raise SkillError(f"graph contains duplicate node id {node_id!r}")
        _graph_plain_text(node.get("kind"), f"node[{index}].kind", required=True)
        _graph_plain_text(node.get("label"), f"node[{index}].label")
        locator = _graph_plain_text(node.get("locator"), f"node[{index}].locator")
        if locator and (not locator.startswith("https://") or len(locator) > 2048):
            raise SkillError(f"graph node[{index}].locator must be an HTTPS URL")
        if locator and NUMERIC_MULTIPLIER_RE.search(locator):
            raise SkillError(f"graph node[{index}].locator contains a numeric-claim-like token")
        for field in ("applies_to", "families"):
            _graph_string_list(node.get(field), f"node[{index}].{field}")
        for field in ("review_status", "read_state", "review_depth", "decision_state", "status"):
            _graph_plain_text(node.get(field), f"node[{index}].{field}")
        nodes_by_id[node_id] = node

    validated_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise SkillError(f"graph edge[{index}] must be an object")
        source = _graph_plain_text(edge.get("source"), f"edge[{index}].source", required=True)
        target = _graph_plain_text(edge.get("target"), f"edge[{index}].target", required=True)
        relation = _graph_plain_text(edge.get("relation"), f"edge[{index}].relation", required=True)
        if relation not in GRAPH_ALLOWED_RELATIONS:
            raise SkillError(f"graph edge[{index}] has unknown relation {relation!r}")
        if source not in nodes_by_id or target not in nodes_by_id:
            raise SkillError(f"graph edge[{index}] has a dangling endpoint")
        identity = (source, target, relation)
        if identity in seen_edges:
            raise SkillError(f"graph contains duplicate edge {identity!r}")
        seen_edges.add(identity)
        validated_edges.append(edge)
    return nodes_by_id, validated_edges


def _graph_review_status(node: dict[str, Any]) -> str:
    for field in ("review_status", "read_state", "review_depth", "decision_state", "status"):
        value = node.get(field)
        if isinstance(value, str) and value:
            return value
    return "unspecified-review-state"


def _graph_node_matches_families(
    node: dict[str, Any],
    families: list[str],
    taxonomy: dict[str, Any],
) -> bool:
    tags = {
        str(value).lower()
        for field in ("applies_to", "families")
        for value in node.get(field, [])
        if isinstance(value, str)
    }
    if "all" in tags:
        return True
    routed = {str(value).lower() for value in families}
    family_groups = taxonomy.get("family_groups", {})
    return any(_family_tag_matches(tag, routed, family_groups) for tag in tags)


def _graph_signal_item(
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = nodes_by_id[str(edge["source"])]
    target = nodes_by_id[str(edge["target"])]
    provenance: dict[str, Any] = {
        "source_node_id": source["id"],
        "target_node_id": target["id"],
        "source_review_status": _graph_review_status(source),
        "target_review_status": _graph_review_status(target),
    }
    if source.get("locator"):
        provenance["source_url"] = source["locator"]
    if target.get("locator"):
        provenance["target_url"] = target["locator"]
    return {
        "advice_class": "unreviewed-research-signal",
        "relation": edge["relation"],
        "review_status": "experimental-review-queue",
        "execution_allowed": False,
        "numeric_claims_included": False,
        "graph_provenance": provenance,
    }


def load_graph_advisory(
    path: str | Path,
    families: list[str],
    taxonomy: dict[str, Any],
    *,
    per_relation_limit: int,
) -> dict[str, Any]:
    try:
        graph = _load_bounded_graph_json(Path(path))
        nodes_by_id, edges = validate_knowledge_graph(graph)
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        return _graph_unavailable(families, str(exc))

    relevant_targets = {
        node_id
        for node_id, node in nodes_by_id.items()
        if _graph_node_matches_families(node, families, taxonomy)
    }
    relevant_candidates = {
        str(edge["source"])
        for edge in edges
        if edge["relation"] == "candidate_relevant_to"
        and str(edge["target"]) in relevant_targets
    }
    relevant: dict[str, list[dict[str, Any]]] = {
        relation: [] for relation in GRAPH_SIGNAL_RELATIONS
    }
    for edge in edges:
        relation = str(edge["relation"])
        if relation in {"candidate_relevant_to", "evidence_for", "evidence_for_outcome"}:
            if str(edge["target"]) in relevant_targets:
                relevant[relation].append(edge)
        elif relation == "candidate_version_of" and str(edge["source"]) in relevant_candidates:
            relevant[relation].append(edge)

    selected: list[dict[str, Any]] = []
    relation_counts: dict[str, int] = {}
    for relation in GRAPH_SIGNAL_RELATIONS:
        relation_edges = sorted(
            relevant[relation],
            key=lambda edge: (str(edge["source"]), str(edge["target"])),
        )
        relation_counts[relation] = len(relation_edges)
        selected.extend(
            _graph_signal_item(edge, nodes_by_id)
            for edge in relation_edges[:per_relation_limit]
        )
    return {
        "label": "Unreviewed research signals (experimental/review queue)",
        "status": "available",
        "review_only": True,
        "distinct_from_advisor_buckets": True,
        "execution_allowed": False,
        "numeric_authority": "assets/effective_claims.json only",
        "families": list(families),
        "reason": None,
        "relation_counts": relation_counts,
        "selected_count": len(selected),
        "items": selected,
    }


def resolve_route_families(
    inspection: dict[str, Any],
    family_override: str | None = None,
) -> list[str]:
    """Resolve one primary family without discarding an inspected hybrid route."""
    candidate_families = [
        str(candidate.get("family"))
        for candidate in inspection.get("architecture_candidates", [])
        if isinstance(candidate, dict) and isinstance(candidate.get("family"), str)
    ]
    if family_override is not None and family_override not in candidate_families:
        raise SkillError(
            f"Family override {family_override!r} is not present in the trusted inspection candidates"
        )

    recommended = [str(value) for value in inspection.get("recommended_families", [])]
    profile = inspection.get("architecture_profile")
    profile_families = [
        str(value)
        for value in profile.get("families", [])
    ] if isinstance(profile, dict) else []
    if not profile_families and isinstance(profile, dict):
        profile_families = [
            str(component.get("family"))
            for component in profile.get("components", [])
            if isinstance(component, dict) and isinstance(component.get("family"), str)
        ]

    inspected_composition = recommended or profile_families
    if family_override is None:
        if inspected_composition:
            return inspected_composition
        recommended_family = inspection.get("recommended_family")
        return [str(recommended_family)] if recommended_family else []

    if len(inspected_composition) > 1:
        if family_override not in inspected_composition:
            raise SkillError(
                f"Family override {family_override!r} is outside the inspected hybrid composition; "
                "an explicit full-route override is required"
            )
        return [family_override, *[value for value in inspected_composition if value != family_override]]
    if inspected_composition == [family_override]:
        return inspected_composition
    return [family_override]


def load_receipt_assessments(
    path: str | None,
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    assessment_path = (Path(path) if path else DEFAULT_RECEIPT_ASSESSMENTS).expanduser().resolve()
    if not assessment_path.exists():
        if path:
            raise SkillError(f"Receipt assessment report not found: {assessment_path}")
        return {"schema_version": 1, "assessments": []}, "missing", {}
    if assessment_path.name != "receipt_assessments.json":
        raise SkillError(
            "Receipt assessment overrides must be named receipt_assessments.json and colocated "
            "with the receipt files they assess"
        )
    report = load_structured(assessment_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise SkillError("Receipt assessment report must be a schema_version 1 mapping")
    if not isinstance(report.get("assessments"), list):
        raise SkillError("Receipt assessment report assessments must be a list")
    recomputed = build_assessment_report(assessment_path.parent)
    if report != recomputed:
        raise SkillError(
            "Receipt assessment report is stale or is not backed by the colocated benchmark receipts"
        )
    verified_receipts = {
        str(row["receipt"]): copy.deepcopy(row)
        for row in recomputed["assessments"]
        if isinstance(row, dict) and isinstance(row.get("receipt"), str)
    }
    return report, "provided" if path else "checked-in", verified_receipts


def load_effective_claim_catalog(
    path: str,
    guidance: dict[str, Any],
    sources: dict[str, Any],
    receipt_assessments: dict[str, Any],
    *,
    assessments_available: bool,
) -> tuple[dict[str, dict[str, Any]], str]:
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise SkillError(
            f"Effective claim catalog not found: {catalog_path}. "
            "Run scripts/generate_claim_catalog.py before recommending numeric claims."
        )
    catalog = load_structured(catalog_path)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise SkillError("Effective claim catalog must be a schema_version 1 mapping")
    claims = catalog.get("claims")
    if not isinstance(claims, list):
        raise SkillError("Effective claim catalog claims must be a list")
    expected_method_ids = sorted(
        str(method.get("id"))
        for method in guidance.get("methods", [])
        if isinstance(method, dict) and isinstance(method.get("improvement_band"), dict)
    )
    claim_ids = [
        str(claim.get("method_id"))
        for claim in claims
        if isinstance(claim, dict) and claim.get("method_id")
    ]
    if (
        len(claim_ids) != len(claims)
        or len(claim_ids) != len(set(claim_ids))
        or sorted(claim_ids) != expected_method_ids
        or catalog.get("claim_count") != len(claims)
    ):
        raise SkillError(
            "Effective claim catalog must contain exactly one record for every improvement-band method"
        )
    expected = build_claim_catalog(
        guidance,
        sources,
        receipt_assessments,
        assessments_available=assessments_available,
    )
    if catalog != expected:
        raise SkillError(
            "Effective claim catalog is stale for the selected guidance, sources, or receipt assessments"
        )
    return {str(claim["method_id"]): claim for claim in claims}, "current"


def prepare_effective_claim(
    method: dict[str, Any],
    claim: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Replace raw guidance numbers with the generated catalog's effective state."""
    candidate = copy.deepcopy(method)
    raw_band = candidate.get("improvement_band")
    if not isinstance(raw_band, dict):
        if claim is not None:
            raise SkillError(f"Effective claim catalog unexpectedly references {method.get('id')}")
        return candidate, []
    if not isinstance(claim, dict) or claim.get("method_id") != method.get("id"):
        raise SkillError(f"Effective claim catalog lacks method {method.get('id')}")

    promotion_state = claim.get("promotion_state")
    selected_range: str | None
    selected_provenance: str
    catalog_holds: list[str] = []
    if promotion_state == "withheld":
        selected_range = None
        selected_provenance = "profile_required"
        reasons = [str(reason) for reason in claim.get("withheld_reasons", [])]
        detail = ", ".join(reasons) or "catalog-withheld"
        catalog_holds.append(f"Effective claim catalog withholds this numeric range: {detail}")
        candidate["expected_effect"] = (
            "No numeric effect is claimed because the effective-claim catalog withholds this observation."
        )
        candidate["improvement_band"] = {
            "provenance": selected_provenance,
            "range": None,
            "metric": claim.get("metric"),
            "basis": "Numeric guidance is withheld by the generated effective-claim catalog.",
            "applies_when": "Resolve the catalog's withheld reasons and regenerate it before claiming a range.",
            "evidence_lineage_ids": claim.get("evidence_lineage_ids", []),
            "claim_status": claim.get("claim_status"),
            "numeric_authority": "effective_claims",
            "catalog_promotion_state": "withheld",
        }
    elif promotion_state == "profile-eligible":
        selected_range = claim.get("profile_eligible_range")
        selected_provenance = "source_reported"
        if not isinstance(selected_range, str):
            raise SkillError(f"Profile-eligible claim for {method.get('id')} lacks profile_eligible_range")
        candidate["improvement_band"] = {
            "provenance": selected_provenance,
            "range": selected_range,
            "metric": claim.get("metric"),
            "basis": "Numeric range selected by the generated effective-claim catalog.",
            "applies_when": "The catalog target constraints and the supplied TargetProfile both match.",
            "evidence_lineage_ids": claim.get("evidence_lineage_ids", []),
            "claim_status": claim.get("claim_status"),
            "numeric_authority": "effective_claims",
            "catalog_promotion_state": "profile-eligible",
        }
        candidate["expected_effect"] = (
            f"Effective-claim catalog authorizes the profile-eligible {selected_range} range "
            f"for {claim.get('metric')}; TargetProfile gates still apply."
        )
    elif promotion_state == "local-promotion":
        selected_range = claim.get("effective_range")
        selected_provenance = "local_reproduced"
        if not isinstance(selected_range, str):
            raise SkillError(f"Local-promotion claim for {method.get('id')} lacks effective_range")
        experiment_fingerprint = claim.get("experiment_fingerprint")
        if not experiment_fingerprint_valid(experiment_fingerprint):
            raise SkillError(
                f"Local-promotion claim for {method.get('id')} lacks a valid experiment_fingerprint"
            )
        candidate["improvement_band"] = {
            "provenance": selected_provenance,
            "range": selected_range,
            "metric": claim.get("metric"),
            "basis": "Numeric range selected by the generated effective-claim catalog.",
            "applies_when": "The catalog's controlled local-promotion gates remain satisfied.",
            "evidence_lineage_ids": claim.get("evidence_lineage_ids", []),
            "claim_status": claim.get("claim_status"),
            "numeric_authority": "effective_claims",
            "catalog_promotion_state": "local-promotion",
            "experiment_fingerprint": copy.deepcopy(experiment_fingerprint),
        }
        candidate["expected_effect"] = (
            f"Effective-claim catalog authorizes the locally promoted {selected_range} range "
            f"for {claim.get('metric')}."
        )
    else:
        raise SkillError(
            f"Effective claim for {method.get('id')} has unsupported promotion_state {promotion_state}"
        )
    candidate["effective_claim_state"] = {
        "promotion_state": promotion_state,
        "withheld_reasons": [str(reason) for reason in claim.get("withheld_reasons", [])],
        "experiment_fingerprint": (
            copy.deepcopy(claim.get("experiment_fingerprint"))
            if promotion_state == "local-promotion"
            else None
        ),
    }
    return candidate, catalog_holds


def build_match_context(
    inspection: dict[str, Any],
    families: list[str],
    target_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    config = inspection.get("configs", {}).get("config.json", {})
    profile_capabilities = target_profile.get("capabilities", []) if target_profile else []
    return {
        "families": {str(value).lower() for value in families},
        "model_type": _normalized(config.get("model_type", "")),
        "capabilities": {
            str(value).lower()
            for value in [*inspection.get("architecture_traits", []), *profile_capabilities]
        },
        "workloads": {
            str(value).lower()
            for value in (target_profile.get("workloads", []) if target_profile else [])
        },
    }


def _family_tag_matches(tag: str, families: set[str], family_groups: dict[str, Any]) -> bool:
    if tag in families:
        return True
    group = family_groups.get(tag)
    return isinstance(group, list) and bool(families.intersection(str(value).lower() for value in group))


def method_match(
    method: dict[str, Any],
    context: dict[str, Any],
    taxonomy: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Return exact applicability plus missing controlled prerequisites."""
    families = context["families"]
    capabilities = context["capabilities"]
    workloads = context["workloads"]
    family_groups = taxonomy.get("family_groups", {})
    explicit = method.get("match")
    prerequisite_holds: list[str] = []

    if isinstance(explicit, dict):
        required_families = {str(value).lower() for value in explicit.get("families_any", [])}
        if required_families and not any(
            _family_tag_matches(tag, families, family_groups) for tag in required_families
        ):
            return False, []
        model_types = {_normalized(value) for value in explicit.get("model_types", [])}
        if model_types and context["model_type"] not in model_types:
            return False, []
        missing_capabilities = sorted(
            {str(value).lower() for value in explicit.get("capabilities_all", [])} - capabilities
        )
        if missing_capabilities:
            prerequisite_holds.append(
                "missing exact capabilities: " + ", ".join(missing_capabilities)
            )
        required_workloads = {str(value).lower() for value in explicit.get("workloads_any", [])}
        if required_workloads and not required_workloads.intersection(workloads):
            prerequisite_holds.append(
                "missing one of the exact workloads: " + ", ".join(sorted(required_workloads))
            )
        return True, prerequisite_holds

    tags = {str(value).lower() for value in method.get("applies_to", [])}
    matched = any(_family_tag_matches(tag, families, family_groups) for tag in tags)
    matched = matched or bool(tags.intersection(capabilities)) or bool(tags.intersection(workloads))
    return matched, []


def objective_match(method: dict[str, Any], objectives: set[str]) -> bool:
    if not objectives:
        return True
    values = {str(x).lower() for x in method.get("objectives", [])}
    return bool(values.intersection(objectives))


def method_sort_key(method: dict[str, Any]) -> tuple[int, str, str]:
    return (STATUS_RANK.get(str(method.get("status")), 99), str(method.get("category", "")), str(method.get("id", "")))


def select_stack(
    stacks: dict[str, Any],
    families: list[str],
    workloads: set[str],
    objectives: set[str],
) -> dict[str, Any] | None:
    targets = {str(value).lower() for value in families}
    matches = []
    for stack in stacks.get("stacks", []) if isinstance(stacks, dict) else []:
        if not isinstance(stack, dict):
            continue
        if not targets.intersection(str(value).lower() for value in stack.get("families", [])):
            continue
        required_workloads = {str(value).lower() for value in stack.get("workloads_any", [])}
        if required_workloads and not required_workloads.intersection(workloads):
            continue
        required_objectives = {str(value).lower() for value in stack.get("objectives_any", [])}
        if objectives and required_objectives and not required_objectives.intersection(objectives):
            continue
        matches.append(stack)
    matches.sort(key=lambda stack: (len(stack.get("families", [])), str(stack.get("id", ""))))
    return matches[0] if matches else None


def build_planning_stack(
    stack: dict[str, Any],
    prepared_methods: list[dict[str, Any]],
    receipt_assessments: dict[str, Any] | None = None,
    verified_receipts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    methods_by_id = {str(method.get("id")): method for method in prepared_methods}
    filtered = copy.deepcopy(stack)
    filtered["steps"] = [
        step
        for step in stack.get("steps", [])
        if step.get("method") in methods_by_id
        and not methods_by_id[step["method"]].get("eligibility_holds")
        and methods_by_id[step["method"]].get("status") != "rejected-or-superseded"
    ]
    retained_ids = {str(step.get("method")) for step in filtered["steps"]}
    filtered["composition_notes"] = [
        note
        for note in stack.get("composition_notes", [])
        if set(str(value) for value in note.get("pair", [])).issubset(retained_ids)
    ]
    compound = compose_stack_band(
        filtered,
        methods_by_id,
        receipt_assessments,
        verified_receipts=verified_receipts,
    )
    steps = []
    for index, step in enumerate(compound["per_step"], start=1):
        candidate = methods_by_id[step["method"]]
        steps.append({
            "step": index,
            "method": step["method"],
            "band": step["band"] or "profile-required",
            "lossiness": step["lossiness"],
            "first_gate": step["gate"],
            "status": candidate.get("status"),
            "advisor_bucket": candidate.get("advisor_bucket"),
            "requires_user_opt_in": bool(candidate.get("requires_user_opt_in")),
            "opt_in_prompt": candidate.get("opt_in_prompt"),
            "execution_allowed": bool(candidate.get("execution_allowed")),
        })
    return {
        "id": stack.get("id"),
        "label": stack.get("label"),
        "status": stack.get("status", "planning-only"),
        "steps": steps,
        "compound": compound,
        "composition_notes": filtered.get("composition_notes", []),
        "rejected_observations": stack.get("observed_configurations", []),
    }


def target_claim_holds(
    method: dict[str, Any],
    target_profile: dict[str, Any] | None,
) -> list[str]:
    band = method.get("improvement_band")
    if not isinstance(band, dict):
        return []
    reasons = [str(value) for value in band.get("hold_reasons", [])]
    if band.get("provenance") == "profile_required":
        return reasons or ["No promotion-ready numeric claim exists for this method"]
    if not isinstance(band.get("range"), str):
        return []
    if band.get("claim_status") == "held":
        if target_profile is None:
            reasons.insert(0, "TargetProfile is required before reconsidering this held observation")
        return reasons or ["stored observation has not passed its promotion gate"]

    if band.get("provenance") == "local_reproduced":
        fingerprint = band.get("experiment_fingerprint")
        if not experiment_fingerprint_valid(fingerprint):
            reasons.append("Local promotion lacks a valid receipt-derived experiment fingerprint")
        elif target_profile is None:
            reasons.append(
                "TargetProfile must include the full canonical receipt-derived experiment_fingerprint"
            )
        else:
            expected = fingerprint["sha256"]
            supplied_fingerprint = target_profile.get("experiment_fingerprint")
            if supplied_fingerprint != fingerprint:
                reasons.append(
                    "TargetProfile experiment_fingerprint does not exactly match the promoted "
                    f"receipt, measurements, quality result, and experiment {expected}"
                )
            supplied_digest = target_profile.get("experiment_fingerprint_sha256")
            if supplied_digest is not None and supplied_digest != expected:
                reasons.append(
                    "TargetProfile experiment_fingerprint_sha256 conflicts with the full canonical fingerprint"
                )

            payload = fingerprint["payload"]
            expected_target = payload["target"]
            target_descriptor = expected_target["descriptor"]
            expected_hardware = target_descriptor.get("hardware")
            expected_software = target_descriptor.get("software")
            if not isinstance(expected_hardware, dict) or not expected_hardware:
                reasons.append("Promoted receipt lacks a non-empty canonical hardware descriptor")
            elif target_profile.get("hardware") != expected_hardware:
                reasons.append("TargetProfile hardware does not exactly match the promoted receipt target")
            if not isinstance(expected_software, dict) or not expected_software:
                reasons.append("Promoted receipt lacks a non-empty canonical software descriptor")
            elif target_profile.get("software") != expected_software:
                reasons.append("TargetProfile software does not exactly match the promoted receipt target")
            if target_profile.get("models") != payload["models"]:
                reasons.append("TargetProfile models do not exactly match the promoted receipt models")
            if target_profile.get("target") != expected_target:
                reasons.append("TargetProfile target does not exactly match the promoted receipt target descriptor")
            if target_profile.get("workload") != payload["workload"]:
                reasons.append(
                    "TargetProfile workload does not exactly match the promoted receipt workload descriptor"
                )
            profile_workloads = target_profile.get("workloads")
            if not isinstance(profile_workloads, list) or not profile_workloads:
                reasons.append(
                    "TargetProfile workloads must be non-empty; a generic profile cannot unlock local promotion"
                )

    constraints = method.get("target_constraints")
    if target_profile is None:
        reasons.append(
            "TargetProfile is required for this version- and workload-sensitive numeric observation"
        )
        return list(dict.fromkeys(reasons))
    if not isinstance(constraints, dict):
        reasons.append("TargetProfile constraints are missing from the method registry")
        return list(dict.fromkeys(reasons))

    for field in constraints.get("required_profile_fields", []):
        if _path_value(target_profile, str(field)) in (None, [], {}):
            reasons.append(f"TargetProfile is missing required field {field}")
    software = target_profile.get("software", {})
    for key, expected in constraints.get("exact_software", {}).items():
        actual = software.get(key) if isinstance(software, dict) else None
        if actual != expected:
            reasons.append(
                f"TargetProfile software.{key}={actual!r} does not match observed version {expected!r}"
            )
    required_workloads = {str(value).lower() for value in constraints.get("workloads_any", [])}
    actual_workloads = {str(value).lower() for value in target_profile.get("workloads", [])}
    if required_workloads and not required_workloads.intersection(actual_workloads):
        reasons.append(
            "TargetProfile lacks a compatible workload: " + ", ".join(sorted(required_workloads))
        )
    return list(dict.fromkeys(reasons))


def advisor_bucket(
    method: dict[str, Any],
    claim_holds: list[str],
    taxonomy: dict[str, Any],
) -> str:
    status = str(method.get("status"))
    if status in {"research-candidate", "rejected-or-superseded"}:
        return str(taxonomy.get("status_to_advisor_bucket", {}).get(status, "benchmark-required"))
    band = method.get("improvement_band")
    provenance = band.get("provenance") if isinstance(band, dict) else None
    claim_bucket = taxonomy.get("claim_provenance_to_advisor_bucket", {}).get(provenance)
    if claim_bucket and not claim_holds:
        return str(claim_bucket)
    return str(taxonomy.get("status_to_advisor_bucket", {}).get(status, "benchmark-required"))


def prepare_candidate(
    method: dict[str, Any],
    effective_claim: dict[str, Any] | None,
    eligibility_holds: list[str],
    target_profile: dict[str, Any] | None,
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    candidate, catalog_holds = prepare_effective_claim(method, effective_claim)
    target_holds = target_claim_holds(candidate, target_profile)
    claim_holds = list(dict.fromkeys([*catalog_holds, *target_holds]))
    if target_holds and isinstance(candidate.get("improvement_band"), dict):
        original_band = candidate["improvement_band"]
        candidate["improvement_band"] = {
            "provenance": "profile_required",
            "range": None,
            "metric": original_band.get("metric"),
            "basis": "Numeric observation withheld until the TargetProfile and promotion gates match.",
            "applies_when": "Resolve every claim_holds entry, then reproduce against a controlled local baseline.",
            "evidence_lineage_ids": original_band.get("evidence_lineage_ids", []),
        }
        if NUMERIC_MULTIPLIER_RE.search(str(candidate.get("expected_effect", ""))):
            candidate["expected_effect"] = (
                "Profile-required. No numeric effect is claimed until the TargetProfile and promotion gates match."
            )
    bucket = advisor_bucket(candidate, claim_holds, taxonomy)
    bucket_def = next(
        (item for item in taxonomy.get("advisor_buckets", []) if item.get("id") == bucket),
        {},
    )
    requires_opt_in = bool(bucket_def.get("requires_user_opt_in"))
    candidate.update({
        "advisor_bucket": bucket,
        "requires_user_opt_in": requires_opt_in,
        "opt_in_prompt": bucket_def.get("prompt") if requires_opt_in else None,
        "execution_allowed": not requires_opt_in and not eligibility_holds and bucket != "rejected-do-not-use",
        "eligibility_holds": eligibility_holds,
        "claim_holds": claim_holds,
    })
    return candidate


def _candidate_rows(items: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in items:
        gate = (item.get("validation_gates") or [""])[0]
        effect = str(item.get("expected_effect", ""))
        if item.get("claim_holds"):
            effect += " Claim held: " + "; ".join(str(value) for value in item["claim_holds"])
        rows.append(f"| `{item['id']}` | `{item['status']}` | {effect} | {gate} |")
    return rows or ["| None | | | |"]


def _stack_markdown(title: str, stack: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- Stack: `{stack['id']}`",
        "",
        "| Step | Method | Band | Lossiness | Advisor contract | Execution | First gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for step in stack["steps"]:
        advisor = str(step.get("advisor_bucket") or "unclassified")
        if step.get("requires_user_opt_in"):
            advisor += "; opt-in required"
        lines.append(
            f"| {step['step']} | `{step['method']}` | `{step['band']}` | "
            f"{step['lossiness']} | {advisor} | "
            f"{'allowed' if step.get('execution_allowed') else 'held'} | {step['first_gate']} |"
        )
    compound = stack["compound"]
    hypothesis = compound["hypothesis_ceiling"]
    measured = compound.get("measured") if isinstance(compound.get("measured"), dict) else None
    lines.append("")
    if measured:
        measured_on = measured.get("measured_on") if isinstance(measured.get("measured_on"), dict) else {}
        measured_where = ", ".join(
            str(value)
            for value in (
                measured_on.get("chip"),
                f"MLX-LM {measured_on.get('mlx_lm')}" if measured_on.get("mlx_lm") else None,
            )
            if value
        )
        measured_line = f"Measured together: `{measured['ratio']}`"
        if measured.get("metric"):
            measured_line += f" `{measured['metric']}`"
        if measured_where:
            measured_line += f" on {measured_where}"
        measured_line += f" ({measured['provenance']})"
        lines.append(measured_line)
        if measured.get("basis"):
            lines.append(f"Basis: {measured['basis']}")
        if measured.get("caveat"):
            lines.append(f"Caveat: {measured['caveat']}")
        if measured.get("receipt"):
            lines.append(f"Receipt: `{measured['receipt']}`")
        lines.append("")
    measured_evidence = compound.get("measured_evidence", [])
    if measured_evidence:
        lines += ["Measured evidence not promoted:", ""]
        for evidence in measured_evidence:
            reasons = "; ".join(str(reason) for reason in evidence.get("reasons", [])) or "not promotion-ready"
            lines.append(
                f"- `{evidence.get('receipt')}`: **{evidence.get('classification')}** - {reasons}"
            )
        lines.append("")
    if hypothesis.get("ceiling") is None:
        lines += [f"Numeric compound: **withheld** - {hypothesis['flag']}", ""]
        for reason in compound.get("withheld_reasons", []):
            lines.append(f"- Hold: {reason}")
        if compound.get("withheld_reasons"):
            lines.append("")
    else:
        ceiling_line = (
            f"Hypothesis ceiling: `{hypothesis['ceiling']}` with floor `{hypothesis['floor']}` "
            f"`{hypothesis['metric']}` ({hypothesis['provenance']}) - {hypothesis['flag']}"
        )
        lines += [ceiling_line, ""]
    unmeasured = compound.get("unmeasured_upside", [])
    if unmeasured:
        lines.append("Unmeasured upside: " + ", ".join(f"`{method}`" for method in unmeasured))
    else:
        lines.append("Unmeasured upside: none")
    other_metric = compound.get("other_metric_upside", [])
    if other_metric:
        rendered = ", ".join(
            f"`{item['method']}` `{item['metric']}` `{item['range']}`"
            for item in other_metric
            if isinstance(item, dict)
        )
        lines.append("Workload-conditional upside (different metric): " + rendered)
    else:
        lines.append("Workload-conditional upside (different metric): none")
    conflicts = compound.get("excluded_conflicts", [])
    if conflicts:
        lines.append("Excluded conflicts: " + ", ".join(f"`{pair[0]}` + `{pair[1]}`" for pair in conflicts))
    else:
        lines.append("Excluded conflicts: none")
    mutually_exclusive = compound.get("mutually_exclusive_pairs", [])
    if mutually_exclusive:
        lines.append(
            "Mutually exclusive: "
            + ", ".join(f"`{pair[0]}` + `{pair[1]}`" for pair in mutually_exclusive)
        )
    observations = stack.get("rejected_observations", [])
    if observations:
        lines += ["", "### Rejected observed configurations", ""]
        for observation in observations:
            lines.append(
                f"- `{observation.get('id')}`: **rejected**, observed `{observation.get('measured_ratio')}` "
                f"`{observation.get('metric')}`. {observation.get('basis', '')}"
            )
    return lines


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# MLX optimization recommendations",
        "",
        "- Families: " + ", ".join(f"`{family}`" for family in report["families"]),
        f"- TargetProfile: `{report['target_profile_status']}`",
        f"- Blocked by intake risks: `{str(report['blocked']).lower()}`",
        f"- Objectives: {', '.join(report['objectives']) or 'all'}",
        "",
    ]
    if report.get("blocked"):
        lines.append(
            "> **Intake is blocked.** Resolve these gates before porting; candidates are held until then "
            "(re-run with `--allow-blocked` to inspect them anyway):"
        )
        lines.append("")
        for blocker in report.get("blockers", []):
            lines.append(f"> - {blocker}")
        if report.get("held_planning_stack"):
            lines += ["", *_stack_markdown("Held optimization experiment plan", report["held_planning_stack"])]
    else:
        if report.get("planning_stack"):
            lines += _stack_markdown("Optimization experiment plan", report["planning_stack"])
            lines.append("")
    bucket_titles = {
        "validated-locally": "Validated locally",
        "validated-source-theory": "Validated by source or theory",
        "benchmark-required": "Benchmark required",
        "experimental-approach": "Experimental approaches",
        "rejected-do-not-use": "Rejected / do not use",
    }
    for bucket_id, title in bucket_titles.items():
        items = report["advisor_buckets"].get(bucket_id, [])
        lines += ["", f"## {title}", ""]
        if bucket_id == "experimental-approach":
            if not items:
                lines.append("None.")
            for item in items:
                caution = (item.get("tradeoffs") or [""])[0]
                lines += [
                    f"- `{item['id']}`: {item['expected_effect']}",
                    f"  - Why cautious: {caution}",
                    f"  - Required opt-in: **{item['opt_in_prompt']}**",
                    "  - Execution allowed before opt-in: `false`",
                ]
            continue
        lines += ["| Method | Status | Expected effect | First gate |", "|---|---|---|---|"]
        lines += _candidate_rows(items)
    research = report["research_advisory"]
    lines += [
        "",
        "## Unreviewed research signals (experimental/review queue)",
        "",
        "> Review-only graph provenance. These signals are separate from the five advisor buckets, "
        "cannot authorize execution, and never supply numeric claims.",
        "",
    ]
    if research["status"] == "unavailable":
        lines.append(f"Graph unavailable: {research['reason']}")
    elif not research["items"]:
        lines.append("No family-relevant graph edges were found.")
    else:
        for item in research["items"]:
            provenance = item["graph_provenance"]
            lines += [
                f"- `{item['relation']}`: `{provenance['source_node_id']}` -> "
                f"`{provenance['target_node_id']}`",
                "  - Review status: "
                f"source `{provenance['source_review_status']}`; "
                f"target `{provenance['target_review_status']}`",
                "  - Execution allowed: `false`",
            ]
            if provenance.get("source_url"):
                lines.append(f"  - Source URL: {provenance['source_url']}")
    if report.get("held_candidates"):
        lines += ["", "## Held by missing capability or workload evidence", "", "| Method | Status | Expected effect | First gate |", "|---|---|---|---|"]
        lines += _candidate_rows(report["held_candidates"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        inspection = validate_trusted_inspection(load_structured(args.inspection))
        guidance = load_structured(args.guidance)
        stacks = load_structured(args.stacks)
        taxonomy = load_structured(args.taxonomy)
        architectures = load_structured(args.architectures)
        sources = load_structured(args.sources)
        (
            receipt_assessments,
            receipt_assessments_status,
            verified_receipts,
        ) = load_receipt_assessments(args.receipt_assessments)
        effective_claims, effective_claim_catalog_status = load_effective_claim_catalog(
            args.effective_claims,
            guidance,
            sources,
            receipt_assessments,
            assessments_available=receipt_assessments_status != "missing",
        )
        target_profile, target_profile_status = load_target_profile(args.target_profile, inspection)
        validate_target_profile_ids(target_profile, taxonomy)
        if args.family:
            validate_family_ids([str(args.family)], architectures)
        families = resolve_route_families(inspection, str(args.family) if args.family else None)
        if not families:
            raise SkillError("No detected family; pass --family after manual architecture review")
        validate_family_ids(families, architectures)
        family = families[0]
        objectives = {str(x).lower() for x in args.objective}
        validate_objective_ids(objectives, taxonomy)
        context = build_match_context(inspection, families, target_profile)
        research_advisory = load_graph_advisory(
            args.knowledge_graph,
            families,
            taxonomy,
            per_relation_limit=min(max(1, args.limit), 25),
        )
        prepared_methods: list[dict[str, Any]] = []
        for method in guidance.get("methods", []):
            if not objective_match(method, objectives):
                continue
            matches, eligibility_holds = method_match(method, context, taxonomy)
            if not matches:
                continue
            prepared_methods.append(
                prepare_candidate(
                    method,
                    effective_claims.get(str(method.get("id"))),
                    eligibility_holds,
                    target_profile,
                    taxonomy,
                )
            )
        prepared_methods.sort(key=method_sort_key)
        limit = max(1, args.limit)
        has_blockers = bool(inspection.get("recommendation_blockers"))
        suppress_blocked_advice = has_blockers and not args.allow_blocked
        if has_blockers:
            for method in prepared_methods:
                method["execution_allowed"] = False
        bucket_ids = [str(item.get("id")) for item in taxonomy.get("advisor_buckets", [])]
        advisor_buckets: dict[str, list[dict[str, Any]]] = {bucket_id: [] for bucket_id in bucket_ids}
        for method in prepared_methods:
            if not suppress_blocked_advice and not method.get("eligibility_holds"):
                advisor_buckets.setdefault(str(method["advisor_bucket"]), []).append(method)
        advisor_buckets = {
            bucket_id: methods[:limit]
            for bucket_id, methods in advisor_buckets.items()
        }
        eligible_ready = [
            method
            for method in prepared_methods
            if method.get("advisor_bucket")
            in {"validated-locally", "validated-source-theory", "benchmark-required"}
            and not method.get("eligibility_holds")
        ][:limit]
        eligible_research = [
            method
            for method in prepared_methods
            if method.get("advisor_bucket") == "experimental-approach"
            and not method.get("eligibility_holds")
        ][:limit]
        exclusions = [
            method
            for method in prepared_methods
            if method.get("advisor_bucket") == "rejected-do-not-use"
        ][:limit]
        eligibility_held = [method for method in prepared_methods if method.get("eligibility_holds")][:limit]
        stack = select_stack(stacks, families, context["workloads"], objectives)
        planning_stack = (
            build_planning_stack(
                stack,
                prepared_methods,
                receipt_assessments,
                verified_receipts,
            )
            if stack
            else None
        )
        held_candidates = (
            [
                method
                for method in prepared_methods
                if method.get("advisor_bucket") != "rejected-do-not-use"
            ][:limit]
            if suppress_blocked_advice
            else eligibility_held
        )
        report = {
            "schema_version": 2,
            "ok": True,
            "inspection_sha256": trusted_inspection_sha256(inspection),
            "family": family,
            "families": families,
            "objectives": sorted(objectives),
            "target_profile_status": target_profile_status,
            "target_profile": target_profile,
            "receipt_assessments_status": receipt_assessments_status,
            "verified_receipt_count": len(verified_receipts),
            "effective_claim_catalog_status": effective_claim_catalog_status,
            "candidate_limit": limit,
            "match_context": {
                "families": sorted(context["families"]),
                "model_type": context["model_type"],
                "capabilities": sorted(context["capabilities"]),
                "workloads": sorted(context["workloads"]),
            },
            "blocked": has_blockers,
            "blocked_advice_visible": bool(has_blockers and args.allow_blocked),
            "blockers": inspection.get("recommendation_blockers", []),
            "advisor_buckets": advisor_buckets,
            "research_advisory": research_advisory,
            "ready_candidates": [] if suppress_blocked_advice else eligible_ready,
            "research_candidates": [] if suppress_blocked_advice else eligible_research,
            "held_candidates": held_candidates,
            "notable_exclusions": exclusions,
            "guidance_reviewed": guidance.get("reviewed"),
            "taxonomy_reviewed": taxonomy.get("reviewed"),
        }
        if planning_stack and has_blockers:
            report["held_planning_stack"] = planning_stack
        elif planning_stack:
            report["planning_stack"] = planning_stack
        if args.output:
            dump_json(report, args.output)
        if args.markdown:
            write_markdown(report, args.markdown)
        if not args.output and not args.markdown:
            print(json.dumps(report, indent=2))
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
