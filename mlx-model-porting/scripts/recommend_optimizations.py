#!/usr/bin/env python3
"""Recommend evidence-backed MLX optimization experiments for an inspection report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, applies_to_family, compose_stack_band, dump_json, load_structured

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
    parser.add_argument("--stacks", default=str(SKILL_ROOT / "assets" / "optimization_stacks.yaml"))
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


def relevant(method: dict[str, Any], family: str) -> bool:
    return applies_to_family(method.get("applies_to", []), family)


def objective_match(method: dict[str, Any], objectives: set[str]) -> bool:
    if not objectives:
        return True
    values = {str(x).lower() for x in method.get("objectives", [])}
    return bool(values.intersection(objectives))


def method_sort_key(method: dict[str, Any]) -> tuple[int, str, str]:
    return (STATUS_RANK.get(str(method.get("status")), 99), str(method.get("category", "")), str(method.get("id", "")))


def select_stack(stacks: dict[str, Any], family: str) -> dict[str, Any] | None:
    for stack in stacks.get("stacks", []) if isinstance(stacks, dict) else []:
        if isinstance(stack, dict) and applies_to_family(stack.get("families", []), family):
            return stack
    return None


def build_recommended_stack(stack: dict[str, Any], guidance_methods: list[dict[str, Any]]) -> dict[str, Any]:
    compound = compose_stack_band(stack, guidance_methods)
    steps = []
    for index, step in enumerate(compound["per_step"], start=1):
        steps.append({
            "step": index,
            "method": step["method"],
            "band": step["band"] or "profile-required",
            "lossiness": step["lossiness"],
            "first_gate": step["gate"],
        })
    return {
        "id": stack.get("id"),
        "label": stack.get("label"),
        "steps": steps,
        "compound": compound,
        "composition_notes": stack.get("composition_notes", []),
    }


def _candidate_rows(items: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in items:
        gate = (item.get("validation_gates") or [""])[0]
        rows.append(f"| `{item['id']}` | `{item['status']}` | {item['expected_effect']} | {gate} |")
    return rows or ["| None | | | |"]


def _stack_markdown(title: str, stack: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- Stack: `{stack['id']}`",
        "",
        "| Step | Method | Band | Lossiness | First gate |",
        "|---|---|---|---|---|",
    ]
    for step in stack["steps"]:
        lines.append(
            f"| {step['step']} | `{step['method']}` | `{step['band']}` | "
            f"{step['lossiness']} | {step['first_gate']} |"
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
    return lines


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# MLX optimization recommendations",
        "",
        f"- Family: `{report['family']}`",
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
        if report.get("held_recommended_stack"):
            lines += ["", *_stack_markdown("Held recommended stack (intake blocked)", report["held_recommended_stack"])]
        lines += ["", "## Held candidates (intake blocked)", "", "| Method | Status | Expected effect | First gate |", "|---|---|---|---|"]
        lines += _candidate_rows(report.get("held_candidates", []))
    else:
        if report.get("recommended_stack"):
            lines += _stack_markdown("Recommended stack", report["recommended_stack"])
            lines.append("")
        lines += ["## Ready candidates", "", "| Method | Status | Expected effect | First gate |", "|---|---|---|---|"]
        lines += _candidate_rows(report["ready_candidates"])
        lines += ["", "## Research experiments", "", "| Method | Expected effect | Why cautious |", "|---|---|---|"]
        for item in report["research_candidates"]:
            caution = (item.get("tradeoffs") or [""])[0]
            lines.append(f"| `{item['id']}` | {item['expected_effect']} | {caution} |")
        if not report["research_candidates"]:
            lines.append("| None | | |")
    exclusions = report.get("notable_exclusions", [])
    if exclusions:
        lines += ["", "## Rejected for MLX (do not port)", "", "| Method | Why rejected |", "|---|---|"]
        for item in exclusions:
            why = (item.get("tradeoffs") or [item.get("recommendation", "")])[0]
            lines.append(f"| `{item['id']}` | {why} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        inspection = load_structured(args.inspection)
        guidance = load_structured(args.guidance)
        stacks = load_structured(args.stacks)
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
        exclusions = [m for m in methods if m.get("status") == "rejected-or-superseded"][:limit]
        blocked = bool(inspection.get("recommendation_blockers")) and not args.allow_blocked
        stack = select_stack(stacks, str(family))
        recommended_stack = build_recommended_stack(stack, guidance.get("methods", [])) if stack else None
        report = {
            "schema_version": 1,
            "ok": True,
            "family": family,
            "objectives": sorted(objectives),
            "blocked": blocked,
            "blockers": inspection.get("recommendation_blockers", []),
            "ready_candidates": [] if blocked else ready,
            "research_candidates": [] if blocked else research,
            "held_candidates": (ready + research) if blocked else [],
            "notable_exclusions": exclusions,
            "guidance_reviewed": guidance.get("reviewed"),
        }
        if recommended_stack and blocked:
            report["held_recommended_stack"] = recommended_stack
        elif recommended_stack:
            report["recommended_stack"] = recommended_stack
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
