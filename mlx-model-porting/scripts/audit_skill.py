#!/usr/bin/env python3
"""Audit Agent Skill structure, registries, references, and bundled Python scripts."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, load_structured, parse_frontmatter

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE_PATH_RE = re.compile(r"`((?:references|scripts|assets)/[^`\s]+)`")
DESCRIPTION_USE_RE = re.compile(r"Use when|Use this skill when")
DESCRIPTION_NEGATIVE_RE = re.compile(r"Do not use|Not for")
DANGEROUS_PATTERNS = {
    "eval(": "dynamic eval",
    "exec(": "dynamic exec",
    "pickle.load": "unsafe pickle deserialization",
    "os.system": "shell execution",
    "shell=True": "subprocess shell execution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an MLX Agent Skill package")
    parser.add_argument("skill", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    return parser.parse_args()


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if condition:
        errors.append(message)


def main() -> int:
    args = parse_args()
    skill = Path(args.skill).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        if not skill.is_dir():
            raise SkillError(f"Skill directory not found: {skill}")
        skill_md = skill / "SKILL.md"
        if not skill_md.exists():
            raise SkillError(f"Missing {skill_md}")
        text = skill_md.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(text)
        name = frontmatter.get("name")
        description = frontmatter.get("description")
        add_error(errors, not isinstance(name, str), "frontmatter.name is required")
        if isinstance(name, str):
            add_error(errors, len(name) > 64 or not NAME_RE.fullmatch(name), f"invalid skill name: {name!r}")
            add_error(errors, name != skill.name, f"skill name {name!r} must match directory {skill.name!r}")
        add_error(errors, not isinstance(description, str), "frontmatter.description is required")
        if isinstance(description, str):
            add_error(errors, not (200 <= len(description) <= 1024), "description must be 200-1024 characters")
            add_error(errors, DESCRIPTION_USE_RE.search(description) is None, 'description must contain a "Use when" clause')
            add_error(errors, DESCRIPTION_NEGATIVE_RE.search(description) is None, 'description must contain a "Do not use" or "Not for" clause')
            add_error(errors, not ("MLX" in description and "Apple Silicon" in description), "description must mention MLX and Apple Silicon")
        compatibility = frontmatter.get("compatibility")
        add_error(errors, compatibility is not None and (not isinstance(compatibility, str) or len(compatibility) > 500), "compatibility must be <=500 characters")
        lines = text.count("\n") + 1
        if lines > 500:
            errors.append(f"SKILL.md has {lines} lines; keep it at or below 500")
        if len(body) > 12000:
            errors.append(f"SKILL.md body has {len(body)} characters; keep it at or below 12000")
        if re.search(r"^## Trigger map\s*$", body, re.MULTILINE) is None:
            errors.append("SKILL.md must contain a ## Trigger map heading")

        referenced: set[str] = set()
        for match in LINK_RE.findall(body):
            if match.startswith(("http://", "https://", "#", "mailto:")):
                continue
            referenced.add(match.split("#", 1)[0])
        referenced.update(CODE_PATH_RE.findall(body))
        for rel in sorted(referenced):
            candidate = (skill / rel).resolve()
            try:
                candidate.relative_to(skill)
            except ValueError:
                errors.append(f"reference escapes skill root: {rel}")
                continue
            if not candidate.exists():
                errors.append(f"missing referenced path: {rel}")

        # Symlinks are allowed only when they resolve inside the skill.
        for path in skill.rglob("*"):
            if path.is_symlink():
                try:
                    path.resolve().relative_to(skill)
                except ValueError:
                    errors.append(f"symlink escapes skill root: {path.relative_to(skill)} -> {path.resolve()}")

        assets = skill / "assets"
        sources = load_structured(assets / "sources.yaml")
        techniques = load_structured(assets / "techniques.yaml")
        architectures = load_structured(assets / "architectures.yaml")
        weight_map_template = load_structured(assets / "WEIGHT_MAP.json")
        add_error(
            errors,
            not isinstance(weight_map_template, dict)
            or weight_map_template.get("schema_version") != 1
            or not isinstance(weight_map_template.get("entries"), list)
            or not isinstance(weight_map_template.get("ignored_source"), list)
            or not isinstance(weight_map_template.get("generated_target"), list),
            "WEIGHT_MAP.json example template has an invalid schema",
        )
        source_items = sources.get("sources", [])
        if sources.get("count") != len(source_items):
            errors.append("sources.yaml count does not match source list length")
        source_ids = [s.get("id") for s in source_items if isinstance(s, dict)]
        if len(source_ids) != len(set(source_ids)):
            errors.append("sources.yaml contains duplicate IDs")
        valid_depth = {"synthesized", "screened", "indexed"}
        for source in source_items:
            if source.get("review_depth") not in valid_depth:
                errors.append(f"invalid review_depth for source {source.get('id')}")
            if not str(source.get("url", "")).startswith("https://"):
                warnings.append(f"source URL is not https: {source.get('id')}")

        source_id_set = set(source_ids)
        for technique in techniques.get("techniques", []):
            for evidence in technique.get("evidence", []):
                if evidence not in source_id_set:
                    errors.append(f"technique {technique.get('id')} references missing evidence {evidence}")
        architecture_ids: list[str] = []
        for family in architectures.get("families", []):
            architecture_ids.append(family.get("id"))
            runbook = family.get("runbook")
            if not isinstance(runbook, str) or not (skill / runbook).exists():
                errors.append(f"architecture {family.get('id')} has missing runbook {runbook}")
        if len(architecture_ids) != len(set(architecture_ids)):
            errors.append("architectures.yaml contains duplicate IDs")

        scripts = sorted((skill / "scripts").glob("*.py"))
        for script in scripts:
            source_text = script.read_text(encoding="utf-8")
            try:
                ast.parse(source_text, filename=str(script))
            except SyntaxError as exc:
                errors.append(f"Python compile failed for {script.name}: {exc}")
            if script.name != "audit_skill.py":
                for pattern, label in DANGEROUS_PATTERNS.items():
                    if pattern in source_text:
                        warnings.append(f"{script.name}: contains {label} pattern {pattern!r}; review manually")

        # Detect unreferenced reference files. Runbooks may be registry-routed;
        # every other reference must be linked directly from SKILL.md.
        direct_ref_names = {Path(x).name for x in referenced if x.startswith("references/")}
        registry_ref_names = {Path(f.get("runbook", "")).name for f in architectures.get("families", [])}
        for ref in (skill / "references").glob("*.md"):
            if ref.name in direct_ref_names:
                continue
            if ref.name.startswith("runbook-"):
                if ref.name not in registry_ref_names:
                    warnings.append(f"runbook reference is not directly routed from SKILL.md or architecture registry: {ref.name}")
                continue
            errors.append(f"reference must be directly linked from SKILL.md: {ref.name}")

        if args.strict and warnings:
            errors.extend(f"strict warning: {x}" for x in warnings)
        print(f"Skill: {skill}")
        print(f"SKILL.md lines: {lines}")
        print(f"Architecture families: {len(architectures.get('families', []))}")
        print(f"Techniques: {len(techniques.get('techniques', []))}")
        print(f"Sources: {len(source_items)}")
        print(f"Python scripts: {len(scripts)}")
        if warnings:
            print("Warnings:")
            for item in warnings:
                print(f"  - {item}")
        if errors:
            print("Errors:", file=sys.stderr)
            for item in errors:
                print(f"  - {item}", file=sys.stderr)
            return 1
        print("Audit passed.")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
