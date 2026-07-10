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
    assessments = load_mapping(
        assets / "benchmarks" / "receipt_assessments.json",
        "benchmark receipt assessments",
    )
    claims = load_mapping(assets / "effective_claims.json", "effective claim catalog")

    family_rows = require_rows(architectures, "families", "architecture registry")
    source_rows = require_rows(sources, "sources", "source registry")
    technique_rows = require_rows(techniques, "techniques", "technique registry")
    guidance_rows = require_rows(guidance, "methods", "optimization guidance")
    assessment_rows = require_rows(assessments, "assessments", "benchmark receipt assessments")
    claim_rows = require_rows(claims, "claims", "effective claim catalog")
    if sources.get("count") != len(source_rows):
        raise SkillError("source registry count does not match its source records")
    if claims.get("claim_count") != len(claim_rows):
        raise SkillError("effective claim catalog count does not match its claim records")

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
