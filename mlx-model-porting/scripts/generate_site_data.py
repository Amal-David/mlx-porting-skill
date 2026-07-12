#!/usr/bin/env python3
"""Generate deterministic, offline-loadable data for the static public site."""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from _common import SkillError, atomic_write_text


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SKILL_ROOT.parent
ASSETS = SKILL_ROOT / "assets"
DEFAULT_OUTPUT = REPO_ROOT / "site" / "data.js"
MAX_STRUCTURED_BYTES = 4 * 1024 * 1024
MAX_VERSION_BYTES = 256
MAX_REFERENCE_ENTRIES = 512
MAX_RUNBOOK_HEADING_BYTES = 4096
GLOBAL_NAME = "window.MLX_PORTING_SITE_DATA"
EXPECTED_CHECKPOINT_ORDER = (
    "inspect",
    "oracle",
    "implement",
    "map",
    "parity",
    "profile",
    "optimize",
    "publish",
)
GUIDANCE_SITE_FIELDS = (
    "id",
    "technique_id",
    "status",
    "objectives",
    "applies_to",
    "recommendation",
    "tradeoffs",
    "validation_gates",
    "rollback_conditions",
)


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    """Load one JSON-compatible registry without allowing an unbounded read."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise SkillError(f"{label} not found: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"{label} must be a regular, non-symlink file: {path}")
    if metadata.st_size > MAX_STRUCTURED_BYTES:
        raise SkillError(
            f"{label} exceeds size limit: {metadata.st_size} > {MAX_STRUCTURED_BYTES} bytes"
        )
    with path.open("rb") as handle:
        raw = handle.read(MAX_STRUCTURED_BYTES + 1)
    if len(raw) > MAX_STRUCTURED_BYTES:
        raise SkillError(f"{label} exceeds size limit while reading: {path}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"{label} is not valid JSON-compatible structured data: {path}") from exc
    if not isinstance(value, dict):
        raise SkillError(f"{label} must contain a mapping: {path}")
    return value


def load_version(repo_root: Path) -> str:
    path = repo_root / "VERSION"
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise SkillError(f"VERSION not found: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"VERSION must be a regular, non-symlink file: {path}")
    if metadata.st_size > MAX_VERSION_BYTES:
        raise SkillError(f"VERSION exceeds size limit: {metadata.st_size} > {MAX_VERSION_BYTES} bytes")
    with path.open("rb") as handle:
        raw = handle.read(MAX_VERSION_BYTES + 1)
    if len(raw) > MAX_VERSION_BYTES:
        raise SkillError(f"VERSION exceeds size limit while reading: {path}")
    try:
        version = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SkillError("VERSION must be valid UTF-8") from exc
    if not version:
        raise SkillError("VERSION must not be empty")
    return version


def require_rows(mapping: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SkillError(f"{label}.{key} must be a list of mappings")
    return value


def count_by(rows: list[dict[str, Any]], field: str, missing: str) -> dict[str, int]:
    values = [str(row[field]) if row.get(field) is not None else missing for row in rows]
    return dict(sorted(Counter(values).items()))


def row_ids(rows: list[dict[str, Any]], label: str) -> set[str]:
    identifiers: list[str] = []
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise SkillError(f"{label} id must be a non-empty string")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise SkillError(f"{label} contains duplicate ids")
    return set(identifiers)


def string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SkillError(f"{label} must be a list of non-empty strings")
    return value


def validate_learning(
    learning: dict[str, Any],
    architecture_ids: set[str],
    hybrid_profile_ids: set[str],
    guidance_by_id: dict[str, dict[str, Any]],
    technique_ids: set[str],
    source_ids: set[str],
    outcome_ids: set[str],
) -> None:
    """Reject curriculum drift before it reaches the static site payload."""
    if learning.get("schema_version") != 1:
        raise SkillError("learning curriculum schema_version must be 1")
    if not isinstance(learning.get("reviewed"), str) or not learning["reviewed"]:
        raise SkillError("learning curriculum reviewed date must be non-empty")
    checkpoints = tuple(string_list(learning.get("checkpoint_order"), "checkpoint_order"))
    if checkpoints != EXPECTED_CHECKPOINT_ORDER:
        raise SkillError("learning curriculum checkpoint order is not canonical")

    official_sources = string_list(
        learning.get("official_learning_source_ids"),
        "official_learning_source_ids",
    )
    missing_sources = sorted(set(official_sources) - source_ids)
    if missing_sources:
        raise SkillError(f"learning curriculum references missing source: {missing_sources[0]}")

    foundations = require_rows(learning, "foundations", "learning curriculum")
    translations = require_rows(learning, "translation_lens", "learning curriculum")
    journeys = require_rows(learning, "journeys", "learning curriculum")
    optimization_families = require_rows(
        learning,
        "optimization_families",
        "learning curriculum",
    )
    glossary = require_rows(learning, "glossary", "learning curriculum")
    row_ids(foundations, "learning foundations")
    row_ids(translations, "learning translation lens")
    row_ids(journeys, "learning journeys")
    row_ids(optimization_families, "learning optimization families")
    terms = [row.get("term") for row in glossary]
    if not all(isinstance(term, str) and term for term in terms):
        raise SkillError("learning glossary terms must be non-empty strings")
    if len(terms) != len({term.casefold() for term in terms}):
        raise SkillError("learning glossary contains duplicate terms")

    teaching_fields = {
        "plain_language",
        "why_it_matters",
        "pytorch_cuda",
        "mlx_translation",
        "example",
        "common_trap",
        "proof_check",
        "next_step",
    }
    for label, rows in (("foundation", foundations), ("translation", translations)):
        for row in rows:
            missing = sorted(
                field
                for field in teaching_fields
                if not isinstance(row.get(field), str) or not row[field].strip()
            )
            if missing:
                raise SkillError(f"learning {label} {row.get('id')} lacks {missing[0]}")
            referenced_sources = string_list(row.get("source_ids", []), f"{label}.source_ids")
            missing_sources = sorted(set(referenced_sources) - source_ids)
            if missing_sources:
                raise SkillError(
                    f"learning {label} {row.get('id')} references missing source: "
                    f"{missing_sources[0]}"
                )

    node_fields = {
        "concept",
        "why_mlx_differs",
        "inspect",
        "prerequisite",
        "proof",
        "evidence_state",
    }
    valid_statuses = {"proven", "simulation"}
    all_journey_methods: set[str] = set()
    for journey in journeys:
        journey_id = journey["id"]
        status = journey.get("status")
        if status not in valid_statuses:
            raise SkillError(f"learning journey {journey_id} has invalid status {status}")
        missing_architectures = sorted(
            set(string_list(journey.get("architecture_ids"), f"{journey_id}.architecture_ids"))
            - architecture_ids
        )
        if missing_architectures:
            raise SkillError(
                f"learning journey {journey_id} references missing architecture: "
                f"{missing_architectures[0]}"
            )
        profile_id = journey.get("hybrid_profile_id")
        if profile_id is not None and profile_id not in hybrid_profile_ids:
            raise SkillError(f"learning journey {journey_id} references missing profile: {profile_id}")
        outcome_id = journey.get("model_outcome_id")
        if outcome_id is not None and outcome_id not in outcome_ids:
            raise SkillError(f"learning journey {journey_id} references missing outcome: {outcome_id}")
        journey_sources = string_list(journey.get("source_ids", []), f"{journey_id}.source_ids")
        missing_sources = sorted(set(journey_sources) - source_ids)
        if missing_sources:
            raise SkillError(
                f"learning journey {journey_id} references missing source: {missing_sources[0]}"
            )
        checkpoint_notes = journey.get("checkpoint_notes")
        if not isinstance(checkpoint_notes, dict) or set(checkpoint_notes) != set(checkpoints):
            raise SkillError(f"learning journey {journey_id} lacks canonical checkpoint notes")
        if not all(isinstance(note, str) and note.strip() for note in checkpoint_notes.values()):
            raise SkillError(f"learning journey {journey_id} has an empty checkpoint note")
        proof_boundary = journey.get("proof_boundary")
        if not isinstance(proof_boundary, str) or not proof_boundary.strip():
            raise SkillError(f"learning journey {journey_id} lacks a proof boundary")
        components = require_rows(journey, "component_path", f"learning journey {journey_id}")
        row_ids(components, f"learning journey {journey_id} components")
        for node in components:
            if node.get("checkpoint") not in checkpoints:
                raise SkillError(
                    f"learning journey {journey_id} node {node.get('id')} has invalid checkpoint"
                )
            missing = sorted(
                field
                for field in node_fields
                if not isinstance(node.get(field), str) or not node[field].strip()
            )
            if missing:
                raise SkillError(
                    f"learning journey {journey_id} node {node.get('id')} lacks {missing[0]}"
                )
            if node["evidence_state"] != status:
                raise SkillError(
                    f"learning journey {journey_id} node {node.get('id')} overstates evidence"
                )
        methods = string_list(
            journey.get("optimization_method_ids"),
            f"{journey_id}.optimization_method_ids",
        )
        missing_methods = sorted(set(methods) - set(guidance_by_id))
        if missing_methods:
            raise SkillError(
                f"learning journey {journey_id} references missing method: {missing_methods[0]}"
            )
        all_journey_methods.update(methods)

    categorized: list[str] = []
    method_family: dict[str, str] = {}
    for family in optimization_families:
        family_id = family["id"]
        for field in ("title", "bottleneck", "proof_gate", "rollback"):
            if not isinstance(family.get(field), str) or not family[field].strip():
                raise SkillError(f"learning optimization family {family_id} lacks {field}")
        methods = string_list(family.get("method_ids"), f"{family_id}.method_ids")
        for method_id in methods:
            if method_id in method_family:
                raise SkillError(f"learning optimization method is categorized twice: {method_id}")
            method_family[method_id] = family_id
        categorized.extend(methods)
    missing_methods = sorted(set(guidance_by_id) - set(categorized))
    unknown_methods = sorted(set(categorized) - set(guidance_by_id))
    if missing_methods:
        raise SkillError(f"learning optimization taxonomy omits method: {missing_methods[0]}")
    if unknown_methods:
        raise SkillError(f"learning optimization taxonomy references missing method: {unknown_methods[0]}")
    for method_id in categorized:
        technique_id = guidance_by_id[method_id].get("technique_id")
        if technique_id not in technique_ids:
            raise SkillError(
                f"learning optimization method {method_id} references missing technique: {technique_id}"
            )
    if method_family.get("fast-sdpa") == method_family.get("draft-model-speculation"):
        raise SkillError("fast-sdpa and draft-model-speculation must be separate families")
    method_quality_gates = learning.get("method_quality_gates")
    if not isinstance(method_quality_gates, dict):
        raise SkillError("method_quality_gates must be a mapping")
    unknown_quality_methods = sorted(set(method_quality_gates) - set(guidance_by_id))
    if unknown_quality_methods:
        raise SkillError(
            f"quality-gated optimization references missing method: {unknown_quality_methods[0]}"
        )
    for method_id, gates in method_quality_gates.items():
        string_list(gates, f"method_quality_gates.{method_id}")
    if not all(method in guidance_by_id for method in all_journey_methods):
        raise SkillError("learning journey contains an unknown optimization method")


def build_learning_payload(
    learning: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
    technique_by_id: dict[str, dict[str, Any]],
    guidance_by_id: dict[str, dict[str, Any]],
    claim_by_method: dict[str, dict[str, Any]],
    architecture_by_id: dict[str, dict[str, str]],
    advisor_by_status: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(learning)
    payload["journeys"] = []
    for journey in learning["journeys"]:
        rendered_journey = dict(journey)
        rendered_journey["runbooks"] = [
            {
                "id": architecture_id,
                "label": architecture_by_id[architecture_id]["label"],
                "path": architecture_by_id[architecture_id]["runbook"],
            }
            for architecture_id in journey["architecture_ids"]
        ]
        payload["journeys"].append(rendered_journey)
    payload["official_learning_sources"] = [
        {
            "id": identifier,
            "title": source_by_id[identifier]["title"],
            "url": source_by_id[identifier]["url"],
        }
        for identifier in learning["official_learning_source_ids"]
    ]
    method_ids = {
        identifier
        for family in learning["optimization_families"]
        for identifier in family["method_ids"]
    }
    payload["guidance_methods"] = []
    method_quality_gates = learning["method_quality_gates"]
    optimization_family_by_id = {
        family["id"]: family for family in learning["optimization_families"]
    }
    family_id_by_method = {
        method_id: family["id"]
        for family in learning["optimization_families"]
        for method_id in family["method_ids"]
    }
    for identifier in sorted(method_ids):
        method = {field: guidance_by_id[identifier][field] for field in GUIDANCE_SITE_FIELDS}
        technique_id = method["technique_id"]
        method["title"] = technique_by_id[technique_id]["title"]
        evidence_rows: list[tuple[str, str]] = []
        for role, identifiers in guidance_by_id[identifier].get("evidence_refs", {}).items():
            if isinstance(identifiers, list):
                evidence_rows.extend((value, role) for value in identifiers if isinstance(value, str))
        evidence_ids = [evidence_id for evidence_id, _role in evidence_rows]
        if not evidence_ids:
            raise SkillError(f"optimization guidance {identifier} lacks canonical evidence")
        missing_evidence = sorted(set(evidence_ids) - set(source_by_id))
        if missing_evidence:
            raise SkillError(
                f"optimization guidance {identifier} references missing source: {missing_evidence[0]}"
            )
        indexed = sorted({
            evidence_id
            for evidence_id in evidence_ids
            if source_by_id[evidence_id].get("review_depth") == "indexed"
        })
        if indexed:
            raise SkillError(
                f"optimization guidance {identifier} exposes indexed evidence: {indexed[0]}"
            )
        evidence_role_by_id = dict(reversed(evidence_rows))
        evidence_rank = {"synthesized": 0, "screened": 1}
        preferred_scopes = {
            "native-mlx": {"official_mlx"},
            "official-mlx-project": {"official_mlx_project"},
            "proven-mlx-port": {"third_party_pinned", "local_reproduced"},
        }.get(method["status"], set())
        ordered_evidence_ids = sorted(
            set(evidence_ids),
            key=lambda evidence_id: (
                0 if source_by_id[evidence_id].get("support_scope") in preferred_scopes else 1,
                evidence_rank.get(source_by_id[evidence_id].get("review_depth"), 2),
                evidence_id,
            ),
        )
        method["evidence_links"] = []
        for evidence_id in ordered_evidence_ids:
            source = source_by_id[evidence_id]
            method["evidence_links"].append({
                "id": evidence_id,
                "title": source["title"],
                "url": source["url"],
                "role": evidence_role_by_id[evidence_id],
                "review_depth": source.get("review_depth"),
                "support_scope": source.get("support_scope") or "unspecified",
                "claim_types": source.get("claim_types") or [],
            })
        method["canonical_source"] = method["evidence_links"][0]
        method["quality_gated"] = identifier in method_quality_gates
        method["family_id"] = family_id_by_method[identifier]
        family = optimization_family_by_id[method["family_id"]]
        method["prerequisite"] = (
            "A parity-passing readable baseline and a profile showing this bottleneck: "
            f"{family['bottleneck']}"
        )
        method["proof_gate"] = family["proof_gate"]
        method["quality_gate"] = method_quality_gates.get(identifier, [])
        method["advisor"] = advisor_by_status[method["status"]]
        claim = claim_by_method.get(identifier)
        method["claim_eligibility"] = (
            claim.get("promotion_state", "not-catalogued")
            if claim is not None
            else "not-catalogued"
        )
        method["numeric_authority"] = "effective_claims"
        method["numeric_claim"] = (
            {
                "range": claim["effective_range"],
                "metric": claim.get("metric"),
                "target_constraints": claim.get("target_constraints"),
                "experiment_fingerprint": claim.get("experiment_fingerprint"),
            }
            if claim is not None
            and claim.get("promotion_state") == "local-promotion"
            and isinstance(claim.get("effective_range"), str)
            and claim["effective_range"].strip()
            else None
        )
        objectives = ", ".join(value.replace("-", " ") for value in method["objectives"])
        method["expected_effect"] = (
            f"Targets {objectives}. No numeric effect is claimed; profile the declared target "
            "workload and consult the effective-claim catalog before publishing a number."
            if method["numeric_claim"] is None
            else "A scoped local-promotion claim exists only within its attached metric, target "
            "constraints, and experiment fingerprint."
        )
        payload["guidance_methods"].append(method)
    return payload


def load_runbook_label(skill_root: Path, runbook: str) -> str:
    relative = PurePosixPath(runbook)
    if relative.is_absolute() or ".." in relative.parts:
        raise SkillError(f"architecture runbook path is unsafe: {runbook}")
    path = skill_root.joinpath(*relative.parts)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise SkillError(f"architecture runbook not found: {runbook}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"architecture runbook must be a regular, non-symlink file: {runbook}")
    with path.open("rb") as handle:
        raw = handle.read(MAX_RUNBOOK_HEADING_BYTES + 1)
    if b"\n" not in raw and len(raw) > MAX_RUNBOOK_HEADING_BYTES:
        raise SkillError(f"architecture runbook heading exceeds size limit: {runbook}")
    try:
        heading = raw.splitlines()[0].decode("utf-8")
    except (IndexError, UnicodeDecodeError) as exc:
        raise SkillError(f"architecture runbook heading is invalid: {runbook}") from exc
    prefix = "# Runbook:"
    if not heading.startswith(prefix) or not heading[len(prefix):].strip():
        raise SkillError(f"architecture runbook lacks a canonical heading: {runbook}")
    label = heading[len(prefix):].strip()
    return label[:1].upper() + label[1:]


def count_local_docs(reference_root: Path) -> dict[str, int]:
    try:
        scanned = os.scandir(reference_root)
    except OSError as exc:
        raise SkillError(f"could not enumerate local references: {reference_root}: {exc}") from exc
    references = 0
    runbooks = 0
    seen = 0
    with scanned:
        for entry in scanned:
            seen += 1
            if seen > MAX_REFERENCE_ENTRIES:
                raise SkillError(
                    f"local reference entry limit exceeded: more than {MAX_REFERENCE_ENTRIES}"
                )
            mode = entry.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode) or Path(entry.name).suffix != ".md":
                continue
            references += 1
            if entry.name.startswith("runbook-"):
                runbooks += 1
    return {"references": references, "runbooks": runbooks}


def build_site_data(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    skill_root = repo_root / "mlx-model-porting"
    assets = skill_root / "assets"
    architectures = load_mapping(assets / "architectures.yaml", "architecture registry")
    sources = load_mapping(assets / "sources.yaml", "source registry")
    techniques = load_mapping(assets / "techniques.yaml", "technique registry")
    guidance = load_mapping(assets / "optimization_guidance.yaml", "optimization guidance")
    recommendation_taxonomy = load_mapping(
        assets / "recommendation-taxonomy.yaml",
        "recommendation taxonomy",
    )
    learning = load_mapping(assets / "learning_paths.json", "learning curriculum")
    outcomes = load_mapping(assets / "model_outcomes.json", "model outcomes")
    assessments = load_mapping(
        assets / "benchmarks" / "receipt_assessments.json",
        "benchmark receipt assessments",
    )
    claims = load_mapping(assets / "effective_claims.json", "effective claim catalog")

    family_rows = require_rows(architectures, "families", "architecture registry")
    hybrid_profile_rows = require_rows(
        architectures,
        "hybrid_profiles",
        "architecture registry",
    )
    source_rows = require_rows(sources, "sources", "source registry")
    technique_rows = require_rows(techniques, "techniques", "technique registry")
    guidance_rows = require_rows(guidance, "methods", "optimization guidance")
    outcome_rows = require_rows(outcomes, "records", "model outcomes")
    assessment_rows = require_rows(assessments, "assessments", "benchmark receipt assessments")
    claim_rows = require_rows(claims, "claims", "effective claim catalog")
    advisor_rows = require_rows(
        recommendation_taxonomy,
        "advisor_buckets",
        "recommendation taxonomy",
    )
    if sources.get("count") != len(source_rows):
        raise SkillError("source registry count does not match its source records")
    if claims.get("claim_count") != len(claim_rows):
        raise SkillError("effective claim catalog count does not match its claim records")

    architecture_ids = row_ids(family_rows, "architecture registry")
    hybrid_profile_ids = row_ids(hybrid_profile_rows, "architecture hybrid profiles")
    source_by_id = {row["id"]: row for row in source_rows}
    source_ids = row_ids(source_rows, "source registry")
    technique_ids = row_ids(technique_rows, "technique registry")
    technique_by_id = {row["id"]: row for row in technique_rows}
    guidance_by_id = {row["id"]: row for row in guidance_rows}
    row_ids(guidance_rows, "optimization guidance")
    outcome_ids = row_ids(outcome_rows, "model outcomes")
    claim_by_method: dict[str, dict[str, Any]] = {}
    for claim in claim_rows:
        method_id = claim.get("method_id")
        if not isinstance(method_id, str) or method_id not in guidance_by_id:
            raise SkillError(f"effective claim references missing method: {method_id}")
        if method_id in claim_by_method:
            raise SkillError(f"effective claim catalog duplicates method: {method_id}")
        claim_by_method[method_id] = claim
    advisor_by_id = {row["id"]: row for row in advisor_rows}
    row_ids(advisor_rows, "recommendation advisor buckets")
    status_to_bucket = recommendation_taxonomy.get("status_to_advisor_bucket")
    if not isinstance(status_to_bucket, dict):
        raise SkillError("recommendation taxonomy lacks status_to_advisor_bucket")
    advisor_by_status: dict[str, dict[str, Any]] = {}
    for status in {row["status"] for row in guidance_rows}:
        bucket_id = status_to_bucket.get(status)
        bucket = advisor_by_id.get(bucket_id)
        if bucket is None:
            raise SkillError(f"recommendation status lacks advisor bucket: {status}")
        advisor_by_status[status] = {
            "id": bucket["id"],
            "label": bucket["label"],
            "description": bucket["description"],
            "requires_user_opt_in": bucket["requires_user_opt_in"],
        }
    validate_learning(
        learning,
        architecture_ids,
        hybrid_profile_ids,
        guidance_by_id,
        technique_ids,
        source_ids,
        outcome_ids,
    )

    families: list[dict[str, str]] = []
    for family in family_rows:
        identifier = family.get("id")
        runbook = family.get("runbook")
        if not isinstance(identifier, str) or not identifier:
            raise SkillError("architecture family id must be a non-empty string")
        if not isinstance(runbook, str) or not runbook:
            raise SkillError(f"architecture family {identifier} lacks a runbook")
        families.append({
            "id": identifier,
            "label": load_runbook_label(skill_root, runbook),
            "runbook": runbook,
        })
    families.sort(key=lambda row: row["id"])
    family_site_by_id = {row["id"]: row for row in families}

    return {
        "schema_version": 1,
        "version": load_version(repo_root),
        "architectures": {
            "families": families,
            "total": len(families),
        },
        "sources": {
            "by_classification": count_by(source_rows, "evidence_class", "unclassified"),
            "by_kind": count_by(source_rows, "kind", "unclassified"),
            "by_review_depth": count_by(source_rows, "review_depth", "unclassified"),
            "by_support_scope": count_by(source_rows, "support_scope", "unspecified"),
            "total": len(source_rows),
        },
        "techniques": {
            "by_status": count_by(technique_rows, "status", "unclassified"),
            "total": len(technique_rows),
        },
        "guidance": {
            "by_status": count_by(guidance_rows, "status", "unclassified"),
            "total": len(guidance_rows),
        },
        "learning": build_learning_payload(
            learning,
            source_by_id,
            technique_by_id,
            guidance_by_id,
            claim_by_method,
            family_site_by_id,
            advisor_by_status,
        ),
        "benchmarks": {
            "by_classification": count_by(
                assessment_rows,
                "classification",
                "unclassified",
            ),
            "promotion_ready": sum(row.get("promotion_ready") is True for row in assessment_rows),
            "total": len(assessment_rows),
        },
        "effective_claims": {
            "by_state": count_by(claim_rows, "promotion_state", "unclassified"),
            "total": len(claim_rows),
        },
        "local_docs": count_local_docs(skill_root / "references"),
    }


def render_site_data(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{GLOBAL_NAME} = {payload};\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic static site data")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true", help="Fail if the output is missing or stale")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rendered = render_site_data(build_site_data())
        output = Path(args.output)
        if args.check:
            if not output.is_file() or output.is_symlink():
                print(f"site data is missing or not a regular file: {output}", file=sys.stderr)
                return 1
            if output.stat().st_size != len(rendered.encode("utf-8")):
                print(f"site data is stale: {output}", file=sys.stderr)
                return 1
            try:
                current = output.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print(f"site data is not valid UTF-8: {output}", file=sys.stderr)
                return 1
            if current != rendered:
                print(f"site data is stale: {output}", file=sys.stderr)
                return 1
            print(f"site data is current: {output}")
            return 0
        atomic_write_text(output, rendered)
        print(f"wrote site data: {output}")
        return 0
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
