#!/usr/bin/env python3
"""Shared helpers for the MLX model porting skill scripts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


class SkillError(RuntimeError):
    """Raised for expected, actionable skill-tool errors."""


def load_structured(path: str | Path) -> Any:
    """Load JSON or YAML. Registry .yaml files are JSON-compatible by design."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError(f"Could not read {p}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SkillError(
                f"{p} is not JSON-compatible YAML and PyYAML is not installed. "
                "Install PyYAML or convert the file to JSON-compatible YAML."
            ) from exc
        try:
            return yaml.safe_load(text)
        except Exception as yaml_exc:  # pragma: no cover - dependency-specific
            raise SkillError(f"Could not parse {p}: {yaml_exc}; JSON error: {json_exc}") from yaml_exc


def dump_json(data: Any, path: str | Path | None = None) -> str:
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return text


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse the deliberately simple YAML frontmatter used by SKILL.md."""
    if not text.startswith("---\n"):
        raise SkillError("SKILL.md must start with YAML frontmatter delimited by ---")
    try:
        raw, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise SkillError("SKILL.md frontmatter is missing the closing --- delimiter") from exc

    # Prefer PyYAML when present. Fall back to a conservative parser for scalar
    # fields and one-level metadata maps used by this repository.
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise SkillError("SKILL.md frontmatter must be a mapping")
        return data, body
    except ImportError:
        result: dict[str, Any] = {}
        current_map: dict[str, str] | None = None
        current_key: str | None = None
        for line in raw.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("  ") and current_map is not None and ":" in line:
                key, value = line.strip().split(":", 1)
                current_map[key.strip()] = _strip_yaml_scalar(value.strip())
                continue
            if ":" not in line:
                raise SkillError(f"Unsupported frontmatter line without PyYAML: {line!r}")
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if not value:
                current_map = {}
                result[key] = current_map
                current_key = key
            else:
                result[key] = _strip_yaml_scalar(value)
                current_map = None
                current_key = None
        return result, body


def _strip_yaml_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def safe_relpath(root: Path, candidate: Path) -> str:
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise SkillError(f"Path escapes allowed root {root}: {candidate}") from exc


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "item"


def applies_to_family(applies_to: list, family: str) -> bool:
    """Whether an ``applies_to`` list matches an architecture family id.

    Single source of truth for the fuzzy family matcher shared by
    ``recommend_optimizations.py``, ``make_port_plan.py``, and the reachability
    guard test, so production and test semantics cannot silently diverge.
    """
    f = family.lower()
    tokens = set(f.replace("-", " ").split())
    for value in [str(x).lower() for x in (applies_to or [])]:
        if value == "all" or value == f or value in f or f in value:
            return True
        if "-" not in value and value in tokens:
            return True
    return False


_BAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)x\s*-\s*(\d+(?:\.\d+)?)x\s*$")


def parse_band(range_str: str) -> tuple[float, float]:
    """Parse a multiplier band like ``1.0x-4.3x`` into numeric bounds."""
    if not isinstance(range_str, str):
        raise ValueError(f"improvement band must be a string, got {type(range_str).__name__}")
    match = _BAND_RE.fullmatch(range_str)
    if not match:
        raise ValueError(f"malformed improvement band {range_str!r}; expected '<floor>x-<ceiling>x'")
    floor, ceiling = (float(match.group(1)), float(match.group(2)))
    if not (math.isfinite(floor) and math.isfinite(ceiling)):
        raise ValueError(f"improvement band {range_str!r} contains a non-finite value")
    if floor <= 0 or ceiling <= 0:
        raise ValueError(f"improvement band {range_str!r} must contain positive multipliers")
    if floor > ceiling:
        raise ValueError(f"improvement band {range_str!r} has floor above ceiling")
    return floor, ceiling


def _format_multiplier(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"{text}x"


def compose_stack_band(stack: dict[str, Any], guidance_methods: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    """Derive an advisory compound band for an optimization stack.

    Compound ceilings are a multiplicative hypothesis unless the full stack has
    been measured together. Unmeasured or conflicting steps contribute 1.0x.
    """
    if isinstance(guidance_methods, dict):
        methods_by_id = guidance_methods
    else:
        methods_by_id = {str(method.get("id")): method for method in guidance_methods if isinstance(method, dict)}

    steps = stack.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError(f"stack {stack.get('id', '<unknown>')} steps must be a list")

    excluded_conflicts: list[list[str]] = []
    excluded_methods: set[str] = set()
    for note in stack.get("composition_notes", []) or []:
        if not isinstance(note, dict) or note.get("validity") != "known-conflicting":
            continue
        pair = note.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"stack {stack.get('id', '<unknown>')} has malformed known-conflicting pair {pair!r}")
        conflict_pair = [str(pair[0]), str(pair[1])]
        excluded_conflicts.append(conflict_pair)
        excluded_methods.update(conflict_pair)

    per_step: list[dict[str, Any]] = []
    unmeasured_upside: list[str] = []
    step_floors = [1.0]
    ceiling = 1.0

    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"stack {stack.get('id', '<unknown>')} contains a non-mapping step")
        method_id = str(step.get("method", ""))
        method = methods_by_id.get(method_id, {})
        improvement_band = method.get("improvement_band") if isinstance(method, dict) else None
        band_range = None
        band_provenance = None
        if isinstance(improvement_band, dict):
            band_range = improvement_band.get("range")
            band_provenance = improvement_band.get("provenance")

        per_step.append({
            "method": method_id,
            "band": band_range if isinstance(band_range, str) else None,
            "lossiness": step.get("lossiness"),
            "gate": step.get("gate"),
        })

        if method_id in excluded_methods:
            continue
        if not isinstance(improvement_band, dict) or band_provenance == "profile_required":
            unmeasured_upside.append(method_id)
            continue
        if band_provenance not in {"source_reported", "local_reproduced"}:
            unmeasured_upside.append(method_id)
            continue
        step_floor, step_ceiling = parse_band(str(band_range))
        step_floors.append(step_floor)
        ceiling *= step_ceiling

    compound = stack.get("compound", {}) if isinstance(stack.get("compound", {}), dict) else {}
    measured_together = compound.get("measured_together") is True
    floor = max(step_floors)
    receipts = compound.get("receipts", [])
    has_measured_floor = measured_together and any(
        isinstance(receipt, dict) and ("measured_floor" in receipt or "floor" in receipt)
        for receipt in (receipts if isinstance(receipts, list) else [])
    )
    if not has_measured_floor:
        floor = min(floor, 1.0)
    provenance = "local_reproduced" if measured_together else "multiplicative_hypothesis"
    result = {
        "floor": _format_multiplier(floor),
        "ceiling": _format_multiplier(ceiling),
        "provenance": provenance,
        "unmeasured_upside": unmeasured_upside,
        "excluded_conflicts": excluded_conflicts,
        "per_step": per_step,
    }
    if provenance == "multiplicative_hypothesis":
        result["flag"] = "unmeasured composition - multiplicative hypothesis, not a claim"
    return result
