#!/usr/bin/env python3
"""Recommend evidence-backed MLX optimization experiments for an inspection report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
STATUS_RANK = {
    "native-mlx": 0,
    "official-mlx-project": 1,
    "proven-mlx-port": 2,
    "research-candidate": 3,
    "rejected-or-superseded": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend MLX optimization candidates")
    parser.add_argument("inspection")
    parser.add_argument("--guidance", default=str(SKILL_ROOT / "assets" / "optimization_guidance.yaml"))
    parser.add_argument("--family", help="Override detected architecture family")
    parser.add_argument("--objective", action="append", default=[], help="Filter by objective tag; repeatable")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", help="Write JSON recommendation report")
    parser.add_argument("--markdown", help="Write a compact Markdown shortlist")
    return parser.parse_args()


def relevant(method: dict[str, Any], family: str) -> bool:
    values = [str(x).lower() for x in method.get("applies_to", [])]
    f = family.lower()
    tokens = set(f.replace("-", " ").split())
    for value in values:
        if value == "all" or value == f or value in f or f in value:
            return True
        if "-" not in value and value in tokens:
            return True
    return False


def objective_match(method: dict[str, Any], objectives: set[str]) -> bool:
    if not objectives:
        return True
    values = {str(x).lower() for x in method.get("objectives", [])}
    return bool(values.intersection(objectives))


def method_sort_key(method: dict[str, Any]) -> tuple[int, str, str]:
    return (STATUS_RANK.get(str(method.get("status")), 99), str(method.get("category", "")), str(method.get("id", "")))


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# MLX optimization recommendations",
        "",
        f"- Family: `{report['family']}`",
        f"- Blocked by intake risks: `{str(report['blocked']).lower()}`",
        f"- Objectives: {', '.join(report['objectives']) or 'all'}",
        "",
        "## Ready candidates",
        "",
        "| Method | Status | Expected effect | First gate |",
        "|---|---|---|---|",
    ]
    for item in report["ready_candidates"]:
        gate = (item.get("validation_gates") or [""])[0]
        lines.append(f"| `{item['id']}` | `{item['status']}` | {item['expected_effect']} | {gate} |")
    if not report["ready_candidates"]:
        lines.append("| None | | | |")
    lines += ["", "## Research experiments", "", "| Method | Expected effect | Why cautious |", "|---|---|---|"]
    for item in report["research_candidates"]:
        caution = (item.get("tradeoffs") or [""])[0]
        lines.append(f"| `{item['id']}` | {item['expected_effect']} | {caution} |")
    if not report["research_candidates"]:
        lines.append("| None | | |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        inspection = load_structured(args.inspection)
        guidance = load_structured(args.guidance)
        family = args.family or inspection.get("recommended_family")
        if not family:
            raise SkillError("No detected family; pass --family after manual architecture review")
        objectives = {str(x).lower() for x in args.objective}
        methods = [
            method
            for method in guidance.get("methods", [])
            if relevant(method, str(family)) and objective_match(method, objectives)
        ]
        methods.sort(key=method_sort_key)
        limit = max(1, args.limit)
        ready = [m for m in methods if m.get("status") not in {"research-candidate", "rejected-or-superseded"}][:limit]
        research = [m for m in methods if m.get("status") == "research-candidate"][:limit]
        report = {
            "ok": True,
            "family": family,
            "objectives": sorted(objectives),
            "blocked": bool(inspection.get("recommendation_blockers")),
            "blockers": inspection.get("recommendation_blockers", []),
            "ready_candidates": ready,
            "research_candidates": research,
            "guidance_reviewed": guidance.get("reviewed"),
        }
        if args.output:
            dump_json(report, args.output)
        if args.markdown:
            write_markdown(report, args.markdown)
        print(json.dumps(report, indent=2))
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
