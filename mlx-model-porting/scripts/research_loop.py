#!/usr/bin/env python3
"""Generate and synthesize review-only multi-agent MLX research loops."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured, safe_relpath, slugify

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SKILL_ROOT / "assets" / "research_loop_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a review-only MLX deep research loop")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--objective", default="Deepen MLX model-porting evidence beyond GitHub")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--agent-count", type=int, default=None)
    parser.add_argument("--offline-fixture", help="Fixture with returned agent findings; no network is used")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_structured(path)
    if not isinstance(config, dict):
        raise SkillError("research loop config must be an object")
    if config.get("review_only") is not True:
        raise SkillError("research loop config must set review_only=true")
    lanes = config.get("source_lanes")
    personas = config.get("personas")
    decisions = config.get("decision_states")
    if not isinstance(lanes, list) or not lanes:
        raise SkillError("research loop config must define source_lanes")
    if not isinstance(personas, list) or not personas:
        raise SkillError("research loop config must define personas")
    if not isinstance(decisions, list) or not decisions:
        raise SkillError("research loop config must define decision_states")
    lane_ids = {lane.get("id") for lane in lanes if isinstance(lane, dict)}
    for persona in personas:
        for lane in persona.get("source_lanes", []):
            if lane not in lane_ids:
                raise SkillError(f"persona {persona.get('id')} references unknown source lane {lane}")
    return config


def select_personas(config: dict[str, Any], count: int | None) -> list[dict[str, Any]]:
    personas = list(config.get("personas", []))
    requested = count or int(config.get("default_agent_count") or len(personas))
    if requested <= 0:
        raise SkillError("--agent-count must be positive")
    return personas[: min(requested, len(personas))]


def build_assignments(config: dict[str, Any], objective: str, run_id: str, count: int | None) -> list[dict[str, Any]]:
    policy = config.get("review_policy", {})
    assignments = []
    for persona in select_personas(config, count):
        prompt = (
            f"You are the {persona['title']} for MLX model-porting research loop {run_id}. "
            f"Objective: {objective}. Source lanes: {', '.join(persona.get('source_lanes', []))}. "
            "Return source-backed findings only. Do not execute remote model code. "
            "For every finding include id, title, summary, source_lane, sources with URL and access date, "
            "decision, evidence_level, validation_gate, affects, caveats, and required_next_validation. "
            "Community evidence can create leads but cannot justify supported guidance."
        )
        assignments.append({
            "persona_id": persona["id"],
            "title": persona["title"],
            "source_lanes": persona.get("source_lanes", []),
            "mission": persona.get("mission", ""),
            "prompt": prompt,
            "execution": {
                "kind": "offline-scaffold",
                "state": "scaffolded_not_run",
                "executor": None,
                "log_path": None,
                "result_path": None,
            },
            "review_only": True,
            "policy": {
                "auto_modify_recommendations": policy.get("auto_modify_recommendations", False),
                "execute_remote_model_code": policy.get("execute_remote_model_code", False),
            },
        })
    return assignments


def load_fixture(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"agents": [], "limitations": ["No offline fixture supplied; assignments only."]}
    data = load_structured(path)
    if not isinstance(data, dict):
        raise SkillError("offline fixture must be an object")
    return data


def validate_findings(config: dict[str, Any], fixture: dict[str, Any]) -> None:
    lane_ids = {lane["id"] for lane in config["source_lanes"]}
    decision_ids = {state["id"] for state in config["decision_states"]}
    required = set(config.get("finding_required_fields", []))
    agents = fixture.get("agents", [])
    if not isinstance(agents, list):
        raise SkillError("fixture agents must be a list")
    for agent in agents:
        persona_id = agent.get("persona_id")
        if not persona_id:
            raise SkillError("fixture agent is missing persona_id")
        findings = agent.get("findings", [])
        if not isinstance(findings, list):
            raise SkillError(f"agent {persona_id} findings must be a list")
        for finding in findings:
            if not isinstance(finding, dict):
                raise SkillError(f"agent {persona_id} has non-object finding")
            missing = sorted(field for field in required if not finding.get(field))
            if missing:
                raise SkillError(f"finding {finding.get('id') or '<unknown>'} missing required fields: {', '.join(missing)}")
            if finding["source_lane"] not in lane_ids:
                raise SkillError(f"finding {finding['id']} has unknown source_lane {finding['source_lane']}")
            if finding["decision"] not in decision_ids:
                raise SkillError(f"finding {finding['id']} has unknown decision {finding['decision']}")
            sources = finding.get("sources", [])
            if not isinstance(sources, list) or not sources:
                raise SkillError(f"finding {finding['id']} must include at least one source")
            for source in sources:
                if not isinstance(source, dict) or not source.get("url") or not source.get("accessed"):
                    raise SkillError(f"finding {finding['id']} has a source without url/accessed")
            if finding["decision"] == "adopted":
                if not finding.get("validation_gate") or not finding.get("affects"):
                    raise SkillError(f"adopted finding {finding['id']} needs validation_gate and affects")
                if finding.get("source_lane") == "community_discussions":
                    raise SkillError(f"community-only finding {finding['id']} cannot be adopted")


def flatten_findings(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for agent in fixture.get("agents", []):
        for finding in agent.get("findings", []):
            row = dict(finding)
            row["persona_id"] = agent.get("persona_id")
            rows.append(row)
    return rows


def summarize(config: dict[str, Any], assignments: list[dict[str, Any]], fixture: dict[str, Any], run_id: str, objective: str) -> dict[str, Any]:
    findings = flatten_findings(fixture)
    decision_counts = {state["id"]: 0 for state in config["decision_states"]}
    lanes: dict[str, int] = {lane["id"]: 0 for lane in config["source_lanes"]}
    for finding in findings:
        decision_counts[finding["decision"]] += 1
        lanes[finding["source_lane"]] += 1
    non_github_lanes = sorted(lane for lane, count in lanes.items() if count and lane != "repositories")
    returned_personas = {agent.get("persona_id") for agent in fixture.get("agents", [])}
    execution_counts = {"scaffolded_not_run": 0, "fixture_ingested": 0}
    for assignment in assignments:
        if assignment["persona_id"] in returned_personas:
            assignment["execution"] = {
                "kind": "offline-fixture",
                "state": "fixture_ingested",
                "executor": None,
                "log_path": None,
                "result_path": None,
            }
            execution_counts["fixture_ingested"] += 1
        else:
            execution_counts["scaffolded_not_run"] += 1
    return {
        "schema_version": 1,
        "run_id": run_id,
        "objective": objective,
        "generated_at": utc_now(),
        "review_only": True,
        "agent_count": len(assignments),
        "execution_counts": execution_counts,
        "finding_count": len(findings),
        "source_lane_counts": lanes,
        "non_github_lanes_covered": non_github_lanes,
        "decision_counts": decision_counts,
        "adopted": [f for f in findings if f["decision"] == "adopted"],
        "held": [f for f in findings if f["decision"] == "held"],
        "rejected": [f for f in findings if f["decision"] == "rejected"],
        "needs_validation": [f for f in findings if f["decision"] == "needs-validation"],
        "limitations": fixture.get("limitations", []),
        "instructions": [
            "This synthesis is review material, not an automatic recommendation merge.",
            "Promote findings only through explicit source, validation, tests, and manifest updates.",
            "Do not execute remote model code while investigating candidates.",
        ],
    }


def markdown_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def write_blogs(output_dir: Path, assignments: list[dict[str, Any]], fixture: dict[str, Any]) -> None:
    by_persona = {agent.get("persona_id"): agent for agent in fixture.get("agents", [])}
    blog_dir = output_dir / "blogs"
    blog_dir.mkdir(parents=True, exist_ok=True)
    for assignment in assignments:
        agent = by_persona.get(assignment["persona_id"], {})
        findings = agent.get("findings", [])
        sampled = []
        for finding in findings:
            for source in finding.get("sources", []):
                sampled.append(f"{source.get('title', source.get('url'))} ({source.get('url')}, accessed {source.get('accessed')})")
        finding_lines = [
            f"{finding['id']}: {finding['title']} [{finding['decision']}] - {finding['summary']}"
            for finding in findings
        ]
        validation = [
            f"{finding['id']}: {finding.get('required_next_validation') or finding.get('validation_gate')}"
            for finding in findings
            if finding.get("required_next_validation") or finding.get("validation_gate")
        ]
        text = "\n".join([
            f"# {assignment['title']} Research Blog",
            "",
            "## Assignment",
            assignment["mission"],
            "",
            "## Sources sampled",
            markdown_list(sampled),
            "",
            "## Candidate findings",
            markdown_list(finding_lines),
            "",
            "## Decision notes",
            markdown_list(agent.get("decision_notes", [])),
            "",
            "## Open validation",
            markdown_list(validation + agent.get("limitations", [])),
            "",
        ])
        (blog_dir / f"{slugify(assignment['persona_id'])}.md").write_text(text, encoding="utf-8")


def write_markdown_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Research Loop {summary['run_id']}",
        "",
        f"Objective: {summary['objective']}",
        "",
        f"Review only: {summary['review_only']}",
        f"Findings: {summary['finding_count']}",
        f"Non-GitHub lanes covered: {', '.join(summary['non_github_lanes_covered']) or 'none'}",
        "",
        "## Decision Counts",
    ]
    for decision, count in summary["decision_counts"].items():
        lines.append(f"- {decision}: {count}")
    for section, key in [
        ("Adopted", "adopted"),
        ("Held", "held"),
        ("Rejected", "rejected"),
        ("Needs Validation", "needs_validation"),
    ]:
        lines.extend(["", f"## {section}"])
        rows = []
        for finding in summary[key]:
            rows.append(f"{finding['id']}: {finding['title']} - {finding['summary']}")
        lines.append(markdown_list(rows))
    lines.extend(["", "## Limitations", markdown_list(summary.get("limitations", [])), ""])
    (output_dir / "synthesis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        run_id = args.run_id or default_run_id()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        assignments = build_assignments(config, args.objective, run_id, args.agent_count)
        fixture = load_fixture(args.offline_fixture)
        validate_findings(config, fixture)
        summary = summarize(config, assignments, fixture, run_id, args.objective)
        if args.offline_fixture:
            fixture_path = Path(args.offline_fixture)
            summary["offline_fixture"] = safe_relpath(Path.cwd(), fixture_path) if fixture_path.exists() else args.offline_fixture
        dump_json({
            "schema_version": 1,
            "run_id": run_id,
            "objective": args.objective,
            "generated_at": summary["generated_at"],
            "assignments": assignments,
        }, output_dir / "assignments.json")
        dump_json(summary, output_dir / "synthesis.json")
        write_blogs(output_dir, assignments, fixture)
        write_markdown_summary(output_dir, summary)
        print(f"wrote research loop {run_id} to {output_dir}: {summary['finding_count']} findings")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
