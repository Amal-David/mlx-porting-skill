#!/usr/bin/env python3
"""Generate and verify MANIFEST.json: a sha256 integrity record of every distributed file.

The manifest covers the whole repository tree (skill, tests, CI, docs) minus local
scaffolding and the manifest itself. `check` recomputes hashes from disk and fails loud
on any added, removed, or changed file so the integrity record cannot silently rot.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = "mlx-porting-skill"

# Mirrors .gitignore: local editor/agent scaffolding and build noise are not distributed.
EXCLUDE_DIRS = {
    ".git", ".superplan", ".agents", ".amazonq", ".claude", ".codex",
    ".cursor", ".gemini", ".opencode", "__pycache__", ".pytest_cache", ".ruff_cache",
}
EXCLUDE_NAMES = {".DS_Store", "MANIFEST.json"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log"}


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if set(rel.parts) & EXCLUDE_DIRS:
            continue
        if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
            continue
        yield rel


def build_files(root: Path) -> list[dict[str, Any]]:
    records = [
        {"path": str(rel), "size_bytes": (root / rel).stat().st_size, "sha256": sha256_file(root / rel)}
        for rel in iter_files(root)
    ]
    records.sort(key=lambda r: r["path"])
    return records


def read_version(root: Path) -> str:
    version_file = root / "VERSION"
    return version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "0.0.0"


def generate(root: Path) -> dict[str, Any]:
    files = build_files(root)
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "version": read_version(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }


def diff(committed: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, list[str]]:
    committed_by = {f["path"]: f for f in committed}
    current_by = {f["path"]: f for f in current}
    added = sorted(set(current_by) - set(committed_by))
    removed = sorted(set(committed_by) - set(current_by))
    changed = sorted(
        p for p in set(committed_by) & set(current_by)
        if committed_by[p].get("sha256") != current_by[p].get("sha256")
        or committed_by[p].get("size_bytes") != current_by[p].get("size_bytes")
    )
    return {"added": added, "removed": removed, "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify MANIFEST.json")
    parser.add_argument("action", choices=["generate", "check"])
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root to scan")
    parser.add_argument("--output", help="Manifest path (default <root>/MANIFEST.json)")
    try:
        args = parser.parse_args()
        root = Path(args.root).resolve()
        manifest_path = Path(args.output) if args.output else root / "MANIFEST.json"
        if args.action == "generate":
            report = generate(root)
            dump_json(report, manifest_path)
            print(f"wrote {report['file_count']} entries to {manifest_path}")
            return 0
        if not manifest_path.exists():
            raise SkillError(f"manifest not found: {manifest_path}; run `manifest.py generate` first")
        committed = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
        delta = diff(committed, build_files(root))
        drifted = any(delta.values())
        print(json.dumps({"ok": not drifted, "manifest": str(manifest_path), **delta}, indent=2))
        return 1 if drifted else 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
