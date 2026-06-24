#!/usr/bin/env python3
"""Shared helpers for the MLX model porting skill scripts."""
from __future__ import annotations

import hashlib
import json
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
