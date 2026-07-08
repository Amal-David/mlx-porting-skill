#!/usr/bin/env python3
"""Generate and synthesize review-only multi-agent MLX research loops."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured, safe_relpath, slugify

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SKILL_ROOT / "assets" / "research_loop_config.json"
BLOG_REQUIRED_SECTIONS = [
    "Assignment",
    "Planned sampling",
    "Sources sampled",
    "Sampling coverage",
    "Candidate findings",
    "Decision notes",
    "Open validation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a review-only MLX deep research loop")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--objective", default="Deepen MLX model-porting evidence beyond GitHub")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--agent-count", type=int, default=None)
    parser.add_argument(
        "--assignment-mode",
        choices=("auto", "config-order", "dynamic"),
        default="auto",
        help="Use config order, or dynamically score personas from objective/gap hints",
    )
    parser.add_argument(
        "--gap-hint",
        action="append",
        default=[],
        help="Optional research gap hint used by dynamic assignment planning; may be repeated",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of review-only research iterations to run",
    )
    parser.add_argument(
        "--iteration-index",
        type=int,
        default=None,
        help="Advanced orchestration metadata for a single scaffolded wave",
    )
    parser.add_argument(
        "--iteration-count-total",
        type=int,
        default=None,
        help="Advanced orchestration metadata for the total planned wave count",
    )
    parser.add_argument(
        "--until-review-gate",
        action="store_true",
        help="Treat --iterations as a cap and stop once the review gate passes",
    )
    parser.add_argument("--offline-fixture", help="Fixture with returned agent findings; no network is used")
    parser.add_argument(
        "--executor-command",
        help="Explicit local command to run once per assignment; writes JSON to MLX_RESEARCH_RESULT_PATH",
    )
    parser.add_argument(
        "--executor-workers",
        type=int,
        default=1,
        help="Maximum local executor workers to run concurrently",
    )
    parser.add_argument(
        "--min-sampled-targets",
        type=int,
        default=0,
        help="Minimum matched planned sample targets required for the review gate",
    )
    parser.add_argument(
        "--min-non-github-lanes",
        type=int,
        default=0,
        help="Minimum non-GitHub source lanes with returned findings required for the review gate",
    )
    parser.add_argument(
        "--require-source-lane",
        action="append",
        default=[],
        help="Require at least one returned finding from this configured source lane; may be repeated",
    )
    parser.add_argument(
        "--fail-on-review-gate",
        action="store_true",
        help="Exit non-zero after writing receipts when the review gate does not pass",
    )
    parser.add_argument(
        "--require-explicit-sampling-receipts",
        action="store_true",
        help="Require returned sources to declare the planned sample target they satisfied",
    )
    parser.add_argument(
        "--require-worker-blog-contract",
        action="store_true",
        help="Exit non-zero after writing receipts when a worker-authored blog misses required sections",
    )
    parser.add_argument(
        "--ingest-subagent-results",
        action="store_true",
        help="Read existing per-agent result JSON files from the generated handoff paths",
    )
    parser.add_argument("--execution-timeout", type=float, default=120.0)
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
    for lane in lanes:
        if not isinstance(lane, dict) or not lane.get("id"):
            raise SkillError("each source lane must be an object with an id")
        targets = lane.get("sample_targets", [])
        if targets and not isinstance(targets, list):
            raise SkillError(f"source lane {lane.get('id')} sample_targets must be a list")
        for target in targets:
            if not isinstance(target, dict) or not target.get("title") or not target_locator(target):
                raise SkillError(f"source lane {lane.get('id')} sample target needs title and url/query/path/target")
    for persona in personas:
        for lane in persona.get("source_lanes", []):
            if lane not in lane_ids:
                raise SkillError(f"persona {persona.get('id')} references unknown source lane {lane}")
    return config


STOP_TERMS = {
    "and",
    "are",
    "before",
    "for",
    "from",
    "into",
    "mlx",
    "not",
    "only",
    "or",
    "should",
    "source",
    "that",
    "the",
    "this",
    "to",
    "with",
    "adding",
    "claims",
    "com",
    "json",
    "md",
    "org",
    "py",
    "references",
    "www",
}


def tokenize(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 1 and term not in STOP_TERMS}


def objective_tokens(value: str) -> set[str]:
    terms = tokenize(value)
    if re.search(r"\b(?:beyond|non|outside)[-\s]+github\b", value.lower()):
        terms.discard("github")
    return terms


def collect_terms(value: Any) -> set[str]:
    if isinstance(value, str):
        return tokenize(value)
    if isinstance(value, dict):
        terms: set[str] = set()
        for item in value.values():
            terms.update(collect_terms(item))
        return terms
    if isinstance(value, list):
        terms = set()
        for item in value:
            terms.update(collect_terms(item))
        return terms
    return set()


def lane_terms(lane: dict[str, Any]) -> set[str]:
    fields = {
        "id": lane.get("id"),
        "title": lane.get("title"),
        "evidence_role": lane.get("evidence_role"),
        "examples": lane.get("examples", []),
        "planner_keywords": lane.get("planner_keywords", []),
        "sample_targets": lane.get("sample_targets", []),
    }
    return collect_terms(fields)


def persona_terms(config: dict[str, Any], persona: dict[str, Any]) -> set[str]:
    lanes = lane_catalog(config)
    fields = {
        "id": persona.get("id"),
        "title": persona.get("title"),
        "mission": persona.get("mission"),
        "source_lanes": persona.get("source_lanes", []),
        "planner_keywords": persona.get("planner_keywords", []),
    }
    terms = collect_terms(fields)
    for lane_id in persona.get("source_lanes", []):
        lane = lanes.get(lane_id)
        if lane:
            terms.update(lane_terms(lane))
    return terms


def persona_own_terms(persona: dict[str, Any]) -> set[str]:
    return collect_terms({
        "id": persona.get("id"),
        "title": persona.get("title"),
        "mission": persona.get("mission"),
        "planner_keywords": persona.get("planner_keywords", []),
    })


def requested_agent_count(config: dict[str, Any], count: int | None) -> int:
    personas = list(config.get("personas", []))
    requested = count or int(config.get("default_agent_count") or len(personas))
    if requested <= 0:
        raise SkillError("--agent-count must be positive")
    return min(requested, len(personas))


def planning_reason(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record["score"] == 0:
        reasons.append("No objective or gap terms matched; kept deterministic config-order fallback.")
    if record["matched_gap_terms"]:
        reasons.append(f"Matched gap terms: {', '.join(record['matched_gap_terms'])}.")
    if record["matched_objective_terms"]:
        reasons.append(f"Matched objective terms: {', '.join(record['matched_objective_terms'])}.")
    if record["source_lanes"]:
        reasons.append(f"Covers source lanes: {', '.join(record['source_lanes'])}.")
    return reasons


def plan_personas(
    config: dict[str, Any],
    objective: str,
    count: int | None,
    gap_hints: list[str],
    assignment_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    personas = list(config.get("personas", []))
    requested = requested_agent_count(config, count)
    objective_terms = objective_tokens(objective)
    gap_terms = set()
    for hint in gap_hints:
        gap_terms.update(tokenize(hint))
    mode = assignment_mode
    if mode == "auto":
        mode = "dynamic" if gap_terms else "config-order"

    scored = []
    for index, persona in enumerate(personas):
        terms = persona_terms(config, persona)
        own_terms = persona_own_terms(persona)
        matched_objective = sorted(terms & objective_terms)
        matched_gap = sorted(terms & gap_terms)
        own_objective = sorted(own_terms & objective_terms)
        own_gap = sorted(own_terms & gap_terms)
        score = len(matched_objective) + (3 * len(matched_gap)) + len(own_objective) + (2 * len(own_gap))
        if mode == "config-order":
            score = 0
            matched_objective = []
            matched_gap = []
        record = {
            "persona_id": persona["id"],
            "title": persona["title"],
            "rank": index + 1,
            "config_order": index + 1,
            "score": score,
            "source_lanes": persona.get("source_lanes", []),
            "matched_objective_terms": matched_objective,
            "matched_gap_terms": matched_gap,
        }
        record["reasons"] = planning_reason(record)
        scored.append((persona, record))

    if mode == "dynamic":
        scored.sort(key=lambda item: (-item[1]["score"], item[1]["config_order"]))

    selected_pairs = scored[:requested]
    held_pairs = scored[requested:]
    selected_ids = {record["persona_id"] for _, record in selected_pairs}
    lane_counts = Counter()
    selected_records = []
    for rank, (persona, record) in enumerate(selected_pairs, 1):
        record = dict(record)
        record["rank"] = rank
        selected_records.append(record)
        for lane in persona.get("source_lanes", []):
            lane_counts[lane] += 1
    held_records = []
    for _, record in held_pairs:
        held = dict(record)
        held["held_reason"] = "agent-count cap" if held["persona_id"] not in selected_ids else "selected"
        held_records.append(held)
    receipt = {
        "schema_version": 1,
        "mode": mode,
        "requested_agent_count": requested,
        "objective_terms": sorted(objective_terms),
        "gap_hints": gap_hints,
        "gap_terms": sorted(gap_terms),
        "selected_personas": selected_records,
        "held_personas": held_records,
        "selected_source_lane_counts": dict(sorted(lane_counts.items())),
        "selection_policy": (
            "dynamic score = matched objective terms + 3 * matched gap terms, with a small bonus for direct persona matches; ties keep config order"
            if mode == "dynamic"
            else "deterministic config order"
        ),
    }
    selected_personas = []
    selected_receipts = {record["persona_id"]: record for record in selected_records}
    for persona, _ in selected_pairs:
        row = dict(persona)
        row["_planning"] = selected_receipts[persona["id"]]
        selected_personas.append(row)
    return selected_personas, receipt


def lane_catalog(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(lane["id"]): lane for lane in config["source_lanes"]}


def target_locator(target: dict[str, Any]) -> str:
    return str(target.get("url") or target.get("query") or target.get("path") or target.get("target") or "")


def sample_targets_for_lane(lane: dict[str, Any]) -> list[dict[str, Any]]:
    targets = []
    for target in lane.get("sample_targets", []):
        if not isinstance(target, dict):
            continue
        targets.append({
            "title": str(target.get("title", "")),
            "kind": str(target.get("kind", "source")),
            "locator": target_locator(target),
            "sampling_goal": str(target.get("sampling_goal", "")),
        })
    if targets:
        return targets
    fallback = []
    for example in lane.get("examples", []):
        fallback.append({
            "title": str(example),
            "kind": "example",
            "locator": str(example),
            "sampling_goal": "Use as a lane hint; replace with a concrete source during live research.",
        })
    return fallback


def sample_plan_for_persona(config: dict[str, Any], persona: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = lane_catalog(config)
    plan = []
    for lane_id in persona.get("source_lanes", []):
        lane = lanes[lane_id]
        plan.append({
            "source_lane": lane_id,
            "lane_title": lane.get("title", lane_id),
            "evidence_role": lane.get("evidence_role", ""),
            "targets": sample_targets_for_lane(lane),
        })
    return plan


def render_sample_plan(plan: list[dict[str, Any]]) -> str:
    if not plan:
        return "- None"
    lines = []
    for lane in plan:
        lines.append(f"- {lane['source_lane']}: {lane.get('lane_title', '')}")
        if lane.get("evidence_role"):
            lines.append(f"  Evidence role: {lane['evidence_role']}")
        for target in lane.get("targets", []):
            suffix = f" - {target['sampling_goal']}" if target.get("sampling_goal") else ""
            lines.append(f"  - {target['title']} [{target['kind']}]: {target['locator']}{suffix}")
    return "\n".join(lines)


def build_assignments(
    config: dict[str, Any],
    objective: str,
    run_id: str,
    count: int | None,
    gap_hints: list[str],
    assignment_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = config.get("review_policy", {})
    assignments = []
    personas, planner_receipt = plan_personas(config, objective, count, gap_hints, assignment_mode)
    for persona in personas:
        sample_plan = sample_plan_for_persona(config, persona)
        prompt = (
            f"You are the {persona['title']} for MLX model-porting research loop {run_id}. "
            f"Objective: {objective}. Source lanes: {', '.join(persona.get('source_lanes', []))}. "
            "Return source-backed findings only. Do not execute remote model code. "
            "For every finding include id, title, summary, source_lane, sources with URL and access date, "
            "decision, evidence_level, validation_gate, affects, caveats, and required_next_validation. "
            "Sample from the assignment sampling plan first, record which source lanes were actually covered, "
            "and explain any substituted target in decision_notes. "
            "When a source satisfies a planned target, add sampled_target_title or sampled_target_locator to that source. "
            "Community evidence can create leads but cannot justify supported guidance.\n\n"
            "Sampling plan:\n"
            f"{render_sample_plan(sample_plan)}"
        )
        assignments.append({
            "persona_id": persona["id"],
            "title": persona["title"],
            "source_lanes": persona.get("source_lanes", []),
            "mission": persona.get("mission", ""),
            "sample_plan": sample_plan,
            "prompt": prompt,
            "execution": {
                "kind": "offline-scaffold",
                "state": "scaffolded_not_run",
                "executor": None,
                "log_path": None,
                "result_path": None,
            },
            "planning": persona.get("_planning", {}),
            "review_only": True,
            "policy": {
                "auto_modify_recommendations": policy.get("auto_modify_recommendations", False),
                "execute_remote_model_code": policy.get("execute_remote_model_code", False),
            },
        })
    return assignments, planner_receipt


def load_fixture(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"agents": [], "limitations": ["No offline fixture supplied; assignments only."]}
    data = load_structured(path)
    if not isinstance(data, dict):
        raise SkillError("offline fixture must be an object")
    return data


def rel_output_path(output_dir: Path, path: Path) -> str:
    return safe_relpath(output_dir, path)


def agent_output_paths(output_dir: Path, assignment: dict[str, Any]) -> dict[str, Path]:
    slug = slugify(assignment["persona_id"])
    agent_dir = output_dir / "agents"
    return {
        "assignment": agent_dir / f"{slug}.assignment.json",
        "prompt": agent_dir / f"{slug}.prompt.md",
        "result": agent_dir / f"{slug}.result.json",
        "stdout": agent_dir / f"{slug}.stdout.txt",
        "stderr": agent_dir / f"{slug}.stderr.txt",
        "generated_blog": agent_dir / f"{slug}.generated-blog.md",
        "blog": output_dir / "blogs" / f"{slug}.md",
    }


def agent_rel_paths(output_dir: Path, assignment: dict[str, Any]) -> dict[str, str]:
    paths = agent_output_paths(output_dir, assignment)
    return {name: rel_output_path(output_dir, path) for name, path in paths.items()}


def stringify_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def render_executor_prompt(assignment: dict[str, Any]) -> str:
    return "\n".join([
        f"# {assignment['title']}",
        "",
        "## Mission",
        assignment.get("mission", ""),
        "",
        "## Prompt",
        assignment["prompt"],
        "",
        "## Result Contract",
        "Write one JSON object to MLX_RESEARCH_RESULT_PATH with persona_id, decision_notes, findings, and optional limitations.",
        "Do not execute remote model code.",
        "",
    ])


def write_subagent_handoffs(
    output_dir: Path,
    assignments: list[dict[str, Any]],
    planner_receipt: dict[str, Any],
    run_id: str,
    objective: str,
    iteration: int,
    iteration_count: int,
    execution_mode: str,
    executor_command: str | None = None,
) -> dict[str, Any]:
    agent_dir = output_dir / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    agents = []
    for index, assignment in enumerate(assignments):
        paths = agent_output_paths(output_dir, assignment)
        rel_paths = agent_rel_paths(output_dir, assignment)
        paths["prompt"].write_text(render_executor_prompt(assignment), encoding="utf-8")
        handoff = {
            "assignment_index": index,
            "persona_id": assignment["persona_id"],
            "title": assignment["title"],
            "source_lanes": assignment.get("source_lanes", []),
            "assignment_path": rel_paths["assignment"],
            "prompt_path": rel_paths["prompt"],
            "result_path": rel_paths["result"],
            "blog_path": rel_paths["blog"],
            "blog_source": assignment.get("blog", {}).get("source", "pending"),
            "execution_kind": execution_mode,
            "execution_state": assignment.get("execution", {}).get("state", "scaffolded_not_run"),
            "review_only": True,
        }
        assignment["handoff"] = handoff
        packet = {
            "schema_version": 1,
            "run_id": run_id,
            "objective": objective,
            "iteration": iteration,
            "iteration_count": iteration_count,
            "review_only": True,
            "persona_id": assignment["persona_id"],
            "title": assignment["title"],
            "mission": assignment.get("mission", ""),
            "source_lanes": assignment.get("source_lanes", []),
            "sample_plan": assignment.get("sample_plan", []),
            "planning": assignment.get("planning", {}),
            "policy": assignment.get("policy", {}),
            "prompt": assignment.get("prompt", ""),
            "paths": {
                "assignment": rel_paths["assignment"],
                "prompt": rel_paths["prompt"],
                "result": rel_paths["result"],
                "blog": rel_paths["blog"],
            },
            "execution": assignment.get("execution", {}),
            "blog": assignment.get("blog", {
                "path": rel_paths["blog"],
                "source": "pending",
                "generated_blog_path": rel_paths["generated_blog"],
            }),
            "result_contract": {
                "path": rel_paths["result"],
                "format": "json",
                "required_top_level_fields": ["persona_id", "decision_notes", "findings"],
                "finding_required_fields": [
                    "id",
                    "title",
                    "summary",
                    "source_lane",
                    "sources",
                    "decision",
                    "evidence_level",
                    "validation_gate",
                    "affects",
                    "caveats",
                    "required_next_validation",
                ],
                "source_optional_fields": [
                    "sampled_target_title",
                    "sampled_target_locator",
                ],
            },
            "blog_contract": {
                "path": rel_paths["blog"],
                "sections": BLOG_REQUIRED_SECTIONS,
            },
            "constraints": [
                "Review-only: do not modify recommendation assets.",
                "Do not execute remote model code.",
                "Sample planned targets first and explain substitutions.",
                "Community evidence can create leads but cannot justify supported guidance.",
            ],
            "environment": {
                "MLX_RESEARCH_PERSONA_ID": assignment["persona_id"],
                "MLX_RESEARCH_ASSIGNMENT_PATH": rel_paths["assignment"],
                "MLX_RESEARCH_PROMPT_PATH": rel_paths["prompt"],
                "MLX_RESEARCH_RESULT_PATH": rel_paths["result"],
                "MLX_RESEARCH_BLOG_PATH": rel_paths["blog"],
                "MLX_RESEARCH_REVIEW_ONLY": "1",
            },
        }
        dump_json(packet, paths["assignment"])
        agents.append(handoff)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "objective": objective,
        "generated_at": utc_now(),
        "iteration": iteration,
        "iteration_count": iteration_count,
        "review_only": True,
        "execution_mode": execution_mode,
        "executor_command": executor_command,
        "agent_count": len(agents),
        "assignment_planner": planner_receipt,
        "agents": agents,
        "instructions": [
            "Dispatch one review-only subagent per listed assignment.",
            "Each subagent should read its assignment packet and prompt, then write the result JSON path.",
            "Do not execute remote model code or modify recommendation assets during research.",
        ],
    }
    dump_json(manifest, output_dir / "subagents.json")
    return manifest


def append_review_gate_command_args(command_args: list[str], requirements: dict[str, Any]) -> None:
    min_sampled = int(requirements.get("min_sampled_targets", 0))
    min_non_github = int(requirements.get("min_non_github_lanes", 0))
    if min_sampled:
        command_args.extend(["--min-sampled-targets", str(min_sampled)])
    if min_non_github:
        command_args.extend(["--min-non-github-lanes", str(min_non_github)])
    for lane_id in requirements.get("required_source_lanes", []):
        command_args.extend(["--require-source-lane", str(lane_id)])
    if requirements.get("require_explicit_sampling_receipts"):
        command_args.append("--require-explicit-sampling-receipts")


def append_failure_command_args(command_args: list[str], summary: dict[str, Any]) -> None:
    if summary.get("fail_on_review_gate"):
        command_args.append("--fail-on-review-gate")


def append_blog_contract_command_args(command_args: list[str], summary: dict[str, Any]) -> None:
    if summary.get("require_worker_blog_contract"):
        command_args.append("--require-worker-blog-contract")


def skill_root_command_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return safe_relpath(SKILL_ROOT, resolved)
    except SkillError:
        return str(path)


def ingest_command_args_for_wave(summary: dict[str, Any], output_dir: Path) -> list[str]:
    planner = summary.get("assignment_planner", {})
    dispatch = summary.get("subagent_dispatch", {})
    command_args = [
        "python3",
        "scripts/research_loop.py",
        "--run-id",
        str(summary["run_id"]),
        "--objective",
        str(summary["objective"]),
        "--agent-count",
        str(dispatch.get("agent_count", summary.get("agent_count", 0))),
        "--assignment-mode",
        str(planner.get("mode", "config-order")),
    ]
    for hint in summary.get("gap_hints", []):
        command_args.extend(["--gap-hint", str(hint)])
    iteration = int(summary.get("iteration", 1))
    iteration_count = int(summary.get("iteration_count", 1))
    if iteration_count > 1:
        command_args.extend([
            "--iteration-index",
            str(iteration),
            "--iteration-count-total",
            str(iteration_count),
        ])
    append_review_gate_command_args(
        command_args,
        summary.get("review_gate", {}).get("requirements", {}),
    )
    append_failure_command_args(command_args, summary)
    append_blog_contract_command_args(command_args, summary)
    command_args.extend([
        "--ingest-subagent-results",
        "--output-dir",
        skill_root_command_path(output_dir),
    ])
    return command_args


def campaign_rel_path(root: Path, wave_output_dir: Path, relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return safe_relpath(root, wave_output_dir / relative_path)


def next_iteration_run_id(run_id: str, next_iteration: int) -> str:
    if re.search(r"-i\d{2}$", run_id):
        return re.sub(r"-i\d{2}$", f"-i{next_iteration:02d}", run_id)
    return f"{run_id}-i{next_iteration:02d}"


def next_wave_output_dir(output_dir: Path, next_iteration: int) -> Path:
    if re.fullmatch(r"\d{2}", output_dir.name):
        return output_dir.parent / f"{next_iteration:02d}"
    return output_dir.parent / f"{output_dir.name}-i{next_iteration:02d}"


def build_next_wave_scaffold(
    summary: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    iteration = int(summary.get("iteration", 1))
    iteration_count = int(summary.get("iteration_count", 1))
    if iteration >= iteration_count:
        return None
    next_iteration = iteration + 1
    next_run_id = next_iteration_run_id(str(summary["run_id"]), next_iteration)
    next_output_dir = next_wave_output_dir(output_dir, next_iteration)
    gap_hints = summary.get("next_gap_hints", [])
    command_args = ["python3", "scripts/research_loop.py"]
    if str(args.config) != str(DEFAULT_CONFIG):
        command_args.extend(["--config", str(args.config)])
    command_args.extend([
        "--run-id",
        next_run_id,
        "--objective",
        str(summary["objective"]),
        "--agent-count",
        str(summary.get("agent_count", 0)),
        "--assignment-mode",
        "dynamic",
        "--iteration-index",
        str(next_iteration),
        "--iteration-count-total",
        str(iteration_count),
    ])
    for hint in gap_hints:
        command_args.extend(["--gap-hint", str(hint)])
    append_review_gate_command_args(command_args, summary.get("review_gate", {}).get("requirements", {}))
    append_failure_command_args(command_args, summary)
    append_blog_contract_command_args(command_args, summary)
    command_output_dir = skill_root_command_path(next_output_dir)
    command_args.extend(["--output-dir", command_output_dir])
    return {
        "schema_version": 1,
        "review_only": True,
        "status": "ready_after_current_wave_ingest",
        "current_iteration": iteration,
        "next_iteration": next_iteration,
        "iteration_count": iteration_count,
        "run_id": next_run_id,
        "output_dir": command_output_dir,
        "assignment_mode": "dynamic",
        "gap_hints": gap_hints,
        "command_args": command_args,
        "requires_current_wave_ingestion": True,
        "instructions": [
            "Run the current wave ingest command after worker result files are present.",
            "Then run this scaffold command to create the next wave from the derived gap hints.",
            "Do not execute remote model code or modify recommendation assets during research.",
        ],
    }


def build_campaign_agent(root: Path, wave_output_dir: Path, agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "assignment_index": agent.get("assignment_index"),
        "persona_id": agent.get("persona_id"),
        "title": agent.get("title"),
        "source_lanes": agent.get("source_lanes", []),
        "assignment_path": campaign_rel_path(root, wave_output_dir, agent.get("assignment_path")),
        "prompt_path": campaign_rel_path(root, wave_output_dir, agent.get("prompt_path")),
        "result_path": campaign_rel_path(root, wave_output_dir, agent.get("result_path")),
        "blog_path": campaign_rel_path(root, wave_output_dir, agent.get("blog_path")),
        "blog_source": agent.get("blog_source", "pending"),
        "execution_kind": agent.get("execution_kind"),
        "execution_state": agent.get("execution_state"),
        "review_only": True,
    }


def build_campaign_wave(
    root: Path,
    summary: dict[str, Any],
    wave_output_dir: Path,
    iteration_cap: int,
    until_review_gate: bool,
) -> dict[str, Any]:
    dispatch = summary.get("subagent_dispatch", {})
    agents = [
        build_campaign_agent(root, wave_output_dir, agent)
        for agent in dispatch.get("agents", [])
    ]
    output_rel = safe_relpath(root, wave_output_dir)
    return {
        "iteration": summary.get("iteration", 1),
        "run_id": summary["run_id"],
        "output_dir": output_rel,
        "subagent_manifest_path": safe_relpath(root, wave_output_dir / "subagents.json"),
        "assignments_path": safe_relpath(root, wave_output_dir / "assignments.json"),
        "synthesis_path": safe_relpath(root, wave_output_dir / "synthesis.json"),
        "execution_mode": dispatch.get("execution_mode"),
        "assignment_mode": summary.get("assignment_planner", {}).get("mode"),
        "gap_hints": summary.get("gap_hints", []),
        "next_gap_hints": summary.get("next_gap_hints", []),
        "review_gate": summary.get("review_gate", {}),
        "agent_count": len(agents),
        "agents": agents,
        "launch": {
            "parallel_safe": True,
            "max_parallel_agents": len(agents),
            "input_paths": [
                path
                for agent in agents
                for path in (agent.get("assignment_path"), agent.get("prompt_path"))
                if path
            ],
            "expected_result_paths": [
                agent["result_path"] for agent in agents if agent.get("result_path")
            ],
            "expected_blog_paths": [
                agent["blog_path"] for agent in agents if agent.get("blog_path")
            ],
            "constraints": [
                "Review-only: do not modify recommendation assets.",
                "Do not execute remote model code.",
                "Sample planned targets first and explain substitutions.",
                "Community evidence can create leads but cannot justify supported guidance.",
            ],
        },
        "ingest": {
            "command_args": ingest_command_args_for_wave(summary, wave_output_dir),
            "wait_for_result_paths": [
                agent["result_path"] for agent in agents if agent.get("result_path")
            ],
            "completion_condition": "all result files exist and each persona_id matches its assignment packet",
        },
        "wave_dependency": {
            "requires_prior_wave_ingestion": summary.get("iteration", 1) > 1,
            "requires_ingest_before_next_wave": summary.get("iteration", 1) < iteration_cap,
            "reason": (
                "Ingest this wave before scaffolding later dynamic waves when returned findings should drive gap hints."
                if until_review_gate or summary.get("iteration", 1) < iteration_cap
                else "Single-wave campaign."
            ),
        },
        **({"next_wave_scaffold": summary["next_wave_scaffold"]} if summary.get("next_wave_scaffold") else {}),
    }


def build_campaign_manifest(
    run_id: str,
    objective: str,
    root: Path,
    waves: list[tuple[dict[str, Any], Path]],
    iteration_cap: int,
    until_review_gate: bool,
    stop_reason: str,
) -> dict[str, Any]:
    wave_records = [
        build_campaign_wave(root, summary, wave_output_dir, iteration_cap, until_review_gate)
        for summary, wave_output_dir in waves
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "objective": objective,
        "generated_at": utc_now(),
        "review_only": True,
        "campaign_root": ".",
        "wave_count": len(wave_records),
        "iteration_cap": iteration_cap,
        "until_review_gate": until_review_gate,
        "stop_reason": stop_reason,
        "waves": wave_records,
        "orchestration_contract": {
            "dispatch_model": "one review-only subagent per campaign wave agent",
            "parallelism": "agents within a wave may run in parallel; dynamic waves should be sequenced through ingestion",
            "result_contract": "each subagent writes one JSON object to its result_path and may write markdown to its blog_path",
            "promotion_rule": "campaign receipts are review-only and never promote findings without source, validation, tests, and rollback evidence",
        },
        "instructions": [
            "Spawn one researcher per wave agent using the listed assignment and prompt paths.",
            "Wait for every result_path in a wave before running that wave's ingest command_args.",
            "For dynamic multi-wave campaigns, ingest a completed wave before scaffolding the next wave if returned findings should drive gap hints.",
            "Do not execute remote model code or modify recommendation assets during research.",
        ],
    }


def write_campaign_markdown(output_dir: Path, campaign: dict[str, Any]) -> None:
    lines = [
        f"# Research Campaign {campaign['run_id']}",
        "",
        f"Objective: {campaign['objective']}",
        f"Review only: {campaign['review_only']}",
        f"Waves: {campaign['wave_count']}",
        f"Iteration cap: {campaign['iteration_cap']}",
        f"Stop reason: {campaign['stop_reason']}",
        "",
        "## Orchestration",
        f"- Dispatch: {campaign['orchestration_contract']['dispatch_model']}",
        f"- Parallelism: {campaign['orchestration_contract']['parallelism']}",
        f"- Promotion rule: {campaign['orchestration_contract']['promotion_rule']}",
        "",
        "## Waves",
    ]
    for wave in campaign["waves"]:
        lines.extend([
            f"- Wave {wave['iteration']}: {wave['run_id']} ({wave['agent_count']} agents)",
            f"  - output: {wave['output_dir']}",
            f"  - subagents: {wave['subagent_manifest_path']}",
            f"  - assignment mode: {wave['assignment_mode']}",
            f"  - gap hints: {', '.join(wave.get('gap_hints', [])) or 'none'}",
            f"  - next gap hints: {', '.join(wave.get('next_gap_hints', [])) or 'none'}",
            f"  - ingest command args: {shlex.join(wave['ingest']['command_args'])}",
            f"  - dependency: {wave['wave_dependency']['reason']}",
        ])
        if wave.get("next_wave_scaffold"):
            lines.append(
                f"  - next-wave scaffold command args: {shlex.join(wave['next_wave_scaffold']['command_args'])}"
            )
        for agent in wave["agents"]:
            lines.append(
                f"  - {agent['persona_id']}: {agent['assignment_path']} -> {agent['result_path']}"
            )
    lines.append("")
    (output_dir / "campaign.md").write_text("\n".join(lines), encoding="utf-8")


def write_campaign_receipts(
    output_dir: Path,
    campaign: dict[str, Any],
) -> dict[str, Any]:
    dump_json(campaign, output_dir / "campaign.json")
    write_campaign_markdown(output_dir, campaign)
    return {
        "path": "campaign.json",
        "markdown_path": "campaign.md",
        "wave_count": campaign["wave_count"],
    }


def load_executor_result(result_path: Path, persona_id: str) -> dict[str, Any]:
    if not result_path.exists():
        raise SkillError("executor did not write result JSON")
    data = load_structured(result_path)
    if not isinstance(data, dict):
        raise SkillError("executor result must be an object")
    if data.get("persona_id") != persona_id:
        raise SkillError(
            f"executor result persona_id mismatch: expected {persona_id}, got {data.get('persona_id') or '<missing>'}"
        )
    return data


def execute_assignments(
    assignments: list[dict[str, Any]],
    command: str,
    output_dir: Path,
    run_id: str,
    timeout: float,
    workers: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if timeout <= 0:
        raise SkillError("--execution-timeout must be positive")
    if workers <= 0:
        raise SkillError("--executor-workers must be positive")
    try:
        command_parts = shlex.split(command)
    except ValueError as exc:
        raise SkillError(f"could not parse --executor-command: {exc}") from exc
    if not command_parts:
        raise SkillError("--executor-command must not be empty")

    agent_dir = output_dir / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    actual_workers = min(workers, len(assignments)) if assignments else 1

    def execute_one(index: int, assignment: dict[str, Any]) -> tuple[int, dict[str, Any] | None, dict[str, Any], str | None]:
        persona_id = assignment["persona_id"]
        paths = agent_output_paths(output_dir, assignment)
        assignment_path = paths["assignment"]
        prompt_path = paths["prompt"]
        result_path = paths["result"]
        stdout_path = paths["stdout"]
        stderr_path = paths["stderr"]
        blog_path = paths["blog"]
        prompt_path.write_text(render_executor_prompt(assignment), encoding="utf-8")

        started_at = utc_now()
        exit_code: int | None = None
        timed_out = False
        stdout = ""
        stderr = ""
        failure_reason = None
        try:
            completed = subprocess.run(
                command_parts,
                cwd=Path.cwd(),
                env={
                    **os.environ,
                    "MLX_RESEARCH_PERSONA_ID": persona_id,
                    "MLX_RESEARCH_ASSIGNMENT_PATH": str(assignment_path),
                    "MLX_RESEARCH_PROMPT_PATH": str(prompt_path),
                    "MLX_RESEARCH_RESULT_PATH": str(result_path),
                    "MLX_RESEARCH_BLOG_PATH": str(blog_path),
                    "MLX_RESEARCH_RUN_ID": run_id,
                    "MLX_RESEARCH_OUTPUT_DIR": str(output_dir),
                    "MLX_RESEARCH_REVIEW_ONLY": "1",
                },
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code = completed.returncode
            stdout = stringify_process_output(completed.stdout)
            stderr = stringify_process_output(completed.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = stringify_process_output(exc.stdout)
            stderr = stringify_process_output(exc.stderr)
            failure_reason = f"executor timed out after {timeout:g}s"

        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        state = "executor_failed"
        agent = None
        if failure_reason is None and exit_code != 0:
            failure_reason = f"executor exited with {exit_code}"
        if failure_reason is None:
            try:
                agent = load_executor_result(result_path, persona_id)
                state = "executor_completed"
            except (SkillError, OSError, json.JSONDecodeError) as exc:
                failure_reason = str(exc)

        record: dict[str, Any] = {
            "kind": "local-executor",
            "state": state,
            "executor": command,
            "assignment_index": index,
            "executor_workers": workers,
            "executor_actual_workers": actual_workers,
            "assignment_path": rel_output_path(output_dir, assignment_path),
            "prompt_path": rel_output_path(output_dir, prompt_path),
            "blog_path": rel_output_path(output_dir, blog_path),
            "blog_source": "executor-authored" if blog_path.exists() and blog_path.stat().st_size > 0 else "generated",
            "log_path": rel_output_path(output_dir, stdout_path),
            "stdout_path": rel_output_path(output_dir, stdout_path),
            "stderr_path": rel_output_path(output_dir, stderr_path),
            "result_path": rel_output_path(output_dir, result_path),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        if failure_reason:
            record["failure_reason"] = failure_reason
        failure = f"{persona_id}: {failure_reason}" if failure_reason else None
        return index, agent, record, failure

    results: list[tuple[int, dict[str, Any] | None, dict[str, Any], str | None]] = []
    if actual_workers == 1:
        for index, assignment in enumerate(assignments):
            results.append(execute_one(index, assignment))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as pool:
            future_to_index = {
                pool.submit(execute_one, index, assignment): index
                for index, assignment in enumerate(assignments)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    results.append(future.result())
                except OSError as exc:
                    index = future_to_index[future]
                    assignment = assignments[index]
                    persona_id = assignment["persona_id"]
                    rel_paths = agent_rel_paths(output_dir, assignment)
                    results.append((index, None, {
                        "kind": "local-executor",
                        "state": "executor_failed",
                        "executor": command,
                        "assignment_index": index,
                        "executor_workers": workers,
                        "executor_actual_workers": actual_workers,
                        "assignment_path": rel_paths["assignment"],
                        "prompt_path": rel_paths["prompt"],
                        "blog_path": rel_paths["blog"],
                        "blog_source": "generated",
                        "result_path": rel_paths["result"],
                        "failure_reason": str(exc),
                    }, f"{persona_id}: {exc}"))

    results.sort(key=lambda row: row[0])
    agents = []
    executions: dict[str, dict[str, Any]] = {}
    failures = []
    for index, agent, record, failure in results:
        persona_id = assignments[index]["persona_id"]
        if agent is not None:
            agents.append(agent)
        executions[persona_id] = record
        if failure:
            failures.append(failure)

    if failures:
        raise SkillError("executor failed for " + "; ".join(failures))
    return {
        "schema_version": 1,
        "agents": agents,
        "executor_workers": workers,
        "executor_actual_workers": actual_workers,
        "limitations": [
            "Findings came from an explicit local executor command and remain review-only until promoted separately."
        ],
    }, executions


def ingest_subagent_results(
    assignments: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    agents = []
    executions: dict[str, dict[str, Any]] = {}
    failures = []
    for index, assignment in enumerate(assignments):
        persona_id = assignment["persona_id"]
        paths = agent_output_paths(output_dir, assignment)
        rel_paths = agent_rel_paths(output_dir, assignment)
        result_path = paths["result"]
        blog_path = paths["blog"]
        generated_title = f"# {assignment['title']} Research Blog"
        worker_authored_blog = False
        if blog_path.exists() and blog_path.stat().st_size > 0:
            first_line = blog_path.read_text(encoding="utf-8").splitlines()[0:1]
            worker_authored_blog = not first_line or first_line[0].strip() != generated_title
        state = "subagent_result_missing"
        failure_reason = None
        agent = None
        try:
            agent = load_executor_result(result_path, persona_id)
            state = "subagent_result_ingested"
        except (SkillError, OSError, json.JSONDecodeError) as exc:
            failure_reason = str(exc)
        record: dict[str, Any] = {
            "kind": "external-subagent",
            "state": state,
            "assignment_index": index,
            "assignment_path": rel_paths["assignment"],
            "prompt_path": rel_paths["prompt"],
            "blog_path": rel_paths["blog"],
            "blog_source": "executor-authored" if worker_authored_blog else "generated",
            "result_path": rel_paths["result"],
            "ingested_at": utc_now(),
        }
        if failure_reason:
            record["failure_reason"] = failure_reason
            failures.append(f"{persona_id}: {failure_reason} at {rel_paths['result']}")
        else:
            agents.append(agent)
        executions[persona_id] = record
    if failures:
        raise SkillError("subagent result ingestion failed for " + "; ".join(failures))
    return {
        "schema_version": 1,
        "agents": agents,
        "limitations": [
            "Findings came from externally written subagent result files and remain review-only until promoted separately."
        ],
    }, executions


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


def normalize_locator(value: Any) -> str:
    return str(value or "").strip().rstrip("/").lower()


def source_match_keys(source: dict[str, Any]) -> set[str]:
    return {
        normalize_locator(source.get("url")),
        normalize_locator(source.get("title")),
    } - {""}


def source_explicit_target_keys(source: dict[str, Any]) -> set[str]:
    return {
        normalize_locator(source.get("sampled_target_locator")),
        normalize_locator(source.get("sampled_target_title")),
    } - {""}


def source_has_explicit_sampling_receipt(source: dict[str, Any]) -> bool:
    return bool(source_explicit_target_keys(source))


def target_match_keys(target: dict[str, Any]) -> set[str]:
    return {
        normalize_locator(target.get("locator")),
        normalize_locator(target.get("title")),
    } - {""}


def sources_for_agent(agent: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for finding in agent.get("findings", []):
        for source in finding.get("sources", []):
            key = (
                normalize_locator(source.get("url")),
                normalize_locator(source.get("title")),
                finding.get("id", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "finding_id": finding.get("id", ""),
                "source_lane": finding.get("source_lane", ""),
                "title": source.get("title", source.get("url", "")),
                "url": source.get("url", ""),
                "kind": source.get("kind", ""),
                "accessed": source.get("accessed", ""),
                "sampled_target_title": source.get("sampled_target_title", ""),
                "sampled_target_locator": source.get("sampled_target_locator", ""),
            })
    return rows


def planned_targets_for_assignment(assignment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for lane in assignment.get("sample_plan", []):
        for target in lane.get("targets", []):
            rows.append({
                "source_lane": lane.get("source_lane", ""),
                "lane_title": lane.get("lane_title", ""),
                "title": target.get("title", ""),
                "kind": target.get("kind", ""),
                "locator": target.get("locator", ""),
            })
    return rows


def assignment_sampling_coverage(assignment: dict[str, Any], agent: dict[str, Any]) -> dict[str, Any]:
    planned_targets = planned_targets_for_assignment(assignment)
    returned_sources = sources_for_agent(agent)
    matched_source_indexes: set[int] = set()
    sampled_targets = []
    unsampled_targets = []
    valid_explicit_receipt_source_indexes: set[int] = set()
    invalid_sampling_receipts = []
    for index, source in enumerate(returned_sources):
        explicit_keys = source_explicit_target_keys(source)
        if not explicit_keys:
            continue
        valid = any(
            target.get("source_lane") == source.get("source_lane")
            and explicit_keys & target_match_keys(target)
            for target in planned_targets
        )
        if valid:
            valid_explicit_receipt_source_indexes.add(index)
        else:
            invalid_sampling_receipts.append({
                "finding_id": source.get("finding_id", ""),
                "source_lane": source.get("source_lane", ""),
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "sampled_target_title": source.get("sampled_target_title", ""),
                "sampled_target_locator": source.get("sampled_target_locator", ""),
            })
    for target in planned_targets:
        matches = []
        explicit_receipts = []
        keys = target_match_keys(target)
        for index, source in enumerate(returned_sources):
            source_keys = source_match_keys(source)
            explicit_keys = source_explicit_target_keys(source)
            if target.get("source_lane") == source.get("source_lane") and keys & (source_keys | explicit_keys):
                matches.append(source)
                matched_source_indexes.add(index)
                if explicit_keys and keys & explicit_keys:
                    explicit_receipts.append(source)
        row = dict(target)
        row["matched_sources"] = matches
        row["explicit_receipts"] = explicit_receipts
        row["missing_explicit_sampling_receipt"] = bool(matches and not explicit_receipts)
        if matches:
            sampled_targets.append(row)
        else:
            unsampled_targets.append(row)
    unplanned_sources = [
        source for index, source in enumerate(returned_sources) if index not in matched_source_indexes
    ]
    planned_count = len(planned_targets)
    sampled_count = len(sampled_targets)
    missing_explicit_count = sum(1 for target in sampled_targets if target["missing_explicit_sampling_receipt"])
    explicit_receipt_count = sum(1 for source in returned_sources if source_has_explicit_sampling_receipt(source))
    return {
        "persona_id": assignment.get("persona_id", ""),
        "planned_target_count": planned_count,
        "sampled_target_count": sampled_count,
        "unsampled_target_count": len(unsampled_targets),
        "returned_source_count": len(returned_sources),
        "unplanned_source_count": len(unplanned_sources),
        "explicit_sampling_receipt_count": explicit_receipt_count,
        "valid_explicit_sampling_receipt_count": len(valid_explicit_receipt_source_indexes),
        "invalid_explicit_sampling_receipt_count": len(invalid_sampling_receipts),
        "sampled_target_missing_explicit_receipt_count": missing_explicit_count,
        "coverage_ratio": (sampled_count / planned_count) if planned_count else None,
        "sampled_targets": sampled_targets,
        "unsampled_targets": unsampled_targets,
        "unplanned_sources": unplanned_sources,
        "invalid_sampling_receipts": invalid_sampling_receipts,
    }


def sampling_coverage_summary(assignments: list[dict[str, Any]], fixture: dict[str, Any]) -> dict[str, Any]:
    by_persona = {agent.get("persona_id"): agent for agent in fixture.get("agents", [])}
    assignments_coverage = []
    totals = Counter()
    for assignment in assignments:
        coverage = assignment_sampling_coverage(assignment, by_persona.get(assignment["persona_id"], {}))
        assignment["sampling_coverage"] = coverage
        assignments_coverage.append(coverage)
        totals["planned_target_count"] += coverage["planned_target_count"]
        totals["sampled_target_count"] += coverage["sampled_target_count"]
        totals["unsampled_target_count"] += coverage["unsampled_target_count"]
        totals["returned_source_count"] += coverage["returned_source_count"]
        totals["unplanned_source_count"] += coverage["unplanned_source_count"]
        totals["explicit_sampling_receipt_count"] += coverage["explicit_sampling_receipt_count"]
        totals["valid_explicit_sampling_receipt_count"] += coverage["valid_explicit_sampling_receipt_count"]
        totals["invalid_explicit_sampling_receipt_count"] += coverage["invalid_explicit_sampling_receipt_count"]
        totals["sampled_target_missing_explicit_receipt_count"] += coverage["sampled_target_missing_explicit_receipt_count"]
    planned = totals["planned_target_count"]
    sampled = totals["sampled_target_count"]
    return {
        "planned_target_count": planned,
        "sampled_target_count": sampled,
        "unsampled_target_count": totals["unsampled_target_count"],
        "returned_source_count": totals["returned_source_count"],
        "unplanned_source_count": totals["unplanned_source_count"],
        "explicit_sampling_receipt_count": totals["explicit_sampling_receipt_count"],
        "valid_explicit_sampling_receipt_count": totals["valid_explicit_sampling_receipt_count"],
        "invalid_explicit_sampling_receipt_count": totals["invalid_explicit_sampling_receipt_count"],
        "sampled_target_missing_explicit_receipt_count": totals["sampled_target_missing_explicit_receipt_count"],
        "coverage_ratio": (sampled / planned) if planned else None,
        "assignments": assignments_coverage,
    }


def planned_sampling_summary(assignments: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    planned_lanes = {lane["id"]: 0 for lane in config["source_lanes"]}
    planned_targets = {lane["id"]: 0 for lane in config["source_lanes"]}
    non_github_targets = []
    seen: set[tuple[str, str, str]] = set()
    for assignment in assignments:
        for lane in assignment.get("sample_plan", []):
            lane_id = lane["source_lane"]
            planned_lanes[lane_id] = planned_lanes.get(lane_id, 0) + 1
            targets = lane.get("targets", [])
            planned_targets[lane_id] = planned_targets.get(lane_id, 0) + len(targets)
            if lane_id == "repositories":
                continue
            for target in targets:
                key = (lane_id, target.get("title", ""), target.get("locator", ""))
                if key in seen:
                    continue
                seen.add(key)
                non_github_targets.append({
                    "source_lane": lane_id,
                    "title": target.get("title", ""),
                    "kind": target.get("kind", ""),
                    "locator": target.get("locator", ""),
                })
    return {
        "planned_source_lane_counts": planned_lanes,
        "planned_sample_target_counts": planned_targets,
        "planned_non_github_sample_targets": non_github_targets,
        "planned_non_github_sample_target_count": len(non_github_targets),
    }


def evidence_source_key(source: dict[str, Any]) -> str:
    return normalize_locator(source.get("url")) or normalize_locator(source.get("title"))


def build_evidence_matrix(
    config: dict[str, Any],
    findings: list[dict[str, Any]],
    sampling_coverage: dict[str, Any],
    planned_summary: dict[str, Any],
) -> dict[str, Any]:
    lane_ids = [lane["id"] for lane in config["source_lanes"]]
    source_entries: dict[str, dict[str, Any]] = {}
    lane_citation_counts: Counter[str] = Counter()
    lane_unique_sources: dict[str, set[str]] = {lane_id: set() for lane_id in lane_ids}
    lane_finding_counts: Counter[str] = Counter()
    lane_sampled_targets: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    seen_source_citations: set[tuple[str, str, str]] = set()

    for coverage in sampling_coverage.get("assignments", []):
        for target in coverage.get("sampled_targets", []):
            lane_sampled_targets[str(target.get("source_lane", ""))] += 1

    for finding in findings:
        finding_id = str(finding.get("id", ""))
        persona_id = str(finding.get("persona_id", ""))
        source_lane = str(finding.get("source_lane", ""))
        decision = str(finding.get("decision", ""))
        lane_finding_counts[source_lane] += 1
        decision_counts[decision] += 1
        for source in finding.get("sources", []):
            if not isinstance(source, dict):
                continue
            key = evidence_source_key(source)
            if not key:
                continue
            citation_key = (key, finding_id, persona_id)
            if citation_key in seen_source_citations:
                continue
            seen_source_citations.add(citation_key)
            lane_citation_counts[source_lane] += 1
            lane_unique_sources.setdefault(source_lane, set()).add(key)
            entry = source_entries.setdefault(
                key,
                {
                    "key": key,
                    "title": source.get("title") or source.get("url") or key,
                    "url": source.get("url", ""),
                    "kinds": [],
                    "accessed_dates": [],
                    "source_lanes": [],
                    "source_lane_counts": {},
                    "decision_counts": {},
                    "persona_ids": [],
                    "finding_ids": [],
                    "findings": [],
                    "sampled_targets": [],
                    "citation_count": 0,
                },
            )
            entry["citation_count"] += 1
            if source.get("kind"):
                entry["kinds"].append(str(source["kind"]))
            if source.get("accessed"):
                entry["accessed_dates"].append(str(source["accessed"]))
            entry["source_lanes"].append(source_lane)
            entry["source_lane_counts"][source_lane] = entry["source_lane_counts"].get(source_lane, 0) + 1
            entry["decision_counts"][decision] = entry["decision_counts"].get(decision, 0) + 1
            entry["persona_ids"].append(persona_id)
            entry["finding_ids"].append(finding_id)
            entry["findings"].append({
                "id": finding_id,
                "title": finding.get("title", ""),
                "decision": decision,
                "source_lane": source_lane,
                "persona_id": persona_id,
            })
            sampled_target = {
                "title": source.get("sampled_target_title", ""),
                "locator": source.get("sampled_target_locator", ""),
            }
            if sampled_target["title"] or sampled_target["locator"]:
                entry["sampled_targets"].append(sampled_target)

    sources = []
    for entry in source_entries.values():
        entry["kinds"] = unique_ordered(entry["kinds"])
        entry["accessed_dates"] = unique_ordered(entry["accessed_dates"])
        entry["source_lanes"] = unique_ordered(entry["source_lanes"])
        entry["persona_ids"] = unique_ordered(entry["persona_ids"])
        entry["finding_ids"] = unique_ordered(entry["finding_ids"])
        entry["sampled_targets"] = [
            dict(row)
            for row in {
                (target.get("title", ""), target.get("locator", "")): target
                for target in entry["sampled_targets"]
            }.values()
        ]
        entry["finding_count"] = len(entry["finding_ids"])
        sources.append(entry)
    sources.sort(key=lambda row: (-int(row.get("citation_count", 0)), row.get("key", "")))

    planned_targets = planned_summary.get("planned_sample_target_counts", {})
    lane_rows = []
    thin_lanes = []
    uncited_lanes = []
    for lane_id in lane_ids:
        planned_count = int(planned_targets.get(lane_id, 0))
        sampled_count = int(lane_sampled_targets.get(lane_id, 0))
        citation_count = int(lane_citation_counts.get(lane_id, 0))
        unique_count = len(lane_unique_sources.get(lane_id, set()))
        if planned_count and citation_count == 0:
            status = "uncited"
        elif planned_count and sampled_count < planned_count:
            status = "thin"
        elif citation_count:
            status = "covered"
        else:
            status = "not_planned"
        lane_row = {
            "source_lane": lane_id,
            "planned_target_count": planned_count,
            "sampled_target_count": sampled_count,
            "finding_count": int(lane_finding_counts.get(lane_id, 0)),
            "source_citation_count": citation_count,
            "unique_source_count": unique_count,
            "status": status,
        }
        lane_rows.append(lane_row)
        if status == "thin":
            thin_lanes.append(lane_row)
        elif status == "uncited":
            uncited_lanes.append(lane_row)

    return {
        "schema_version": 1,
        "review_only": True,
        "citation_policy": "Repeated source citation is corroboration context only; it does not promote guidance without validation gates.",
        "unique_source_count": len(sources),
        "source_citation_count": len(seen_source_citations),
        "source_lane_counts": {lane_id: int(lane_citation_counts.get(lane_id, 0)) for lane_id in lane_ids},
        "finding_decision_counts": {state["id"]: int(decision_counts.get(state["id"], 0)) for state in config["decision_states"]},
        "source_lanes": lane_rows,
        "thin_source_lanes": thin_lanes,
        "uncited_source_lanes": uncited_lanes,
        "top_sources": [
            {
                "key": row["key"],
                "title": row["title"],
                "url": row["url"],
                "citation_count": row["citation_count"],
                "finding_count": row["finding_count"],
                "source_lanes": row["source_lanes"],
                "finding_ids": row["finding_ids"],
                "decision_counts": row["decision_counts"],
            }
            for row in sources[:10]
        ],
        "sources": sources,
    }


def review_check(name: str, observed: int, required: int, detail: str) -> dict[str, Any]:
    status = "pass" if observed >= required else "fail"
    return {
        "name": name,
        "status": status,
        "comparison": "at_least",
        "observed": observed,
        "required": required,
        "detail": detail,
    }


def review_check_at_most(name: str, observed: int, required: int, detail: str) -> dict[str, Any]:
    status = "pass" if observed <= required else "fail"
    return {
        "name": name,
        "status": status,
        "comparison": "at_most",
        "observed": observed,
        "required": required,
        "detail": detail,
    }


def build_review_gate(
    sampling_coverage: dict[str, Any],
    source_lane_counts: dict[str, int],
    non_github_lanes_covered: list[str],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        review_check(
            "sampled_planned_targets",
            int(sampling_coverage.get("sampled_target_count", 0)),
            int(requirements.get("min_sampled_targets", 0)),
            "Matched planned sampling targets returned by agents or fixtures.",
        ),
        review_check(
            "non_github_lanes_covered",
            len(non_github_lanes_covered),
            int(requirements.get("min_non_github_lanes", 0)),
            "Non-GitHub source lanes with at least one returned finding.",
        ),
    ]
    for lane_id in requirements.get("required_source_lanes", []):
        checks.append(
            review_check(
                f"required_source_lane:{lane_id}",
                int(source_lane_counts.get(lane_id, 0)),
                1,
                f"Returned findings from required source lane `{lane_id}`.",
            )
        )
    if requirements.get("require_explicit_sampling_receipts"):
        checks.append(
            review_check(
                "explicit_sampling_receipts",
                int(sampling_coverage.get("sampled_target_count", 0))
                - int(sampling_coverage.get("sampled_target_missing_explicit_receipt_count", 0)),
                int(sampling_coverage.get("sampled_target_count", 0)),
                "Matched planned sampling targets backed by explicit worker-declared target receipts.",
            )
        )
        checks.append(
            review_check_at_most(
                "invalid_explicit_sampling_receipts",
                int(sampling_coverage.get("invalid_explicit_sampling_receipt_count", 0)),
                0,
                "Returned sources that claimed a planned sample target outside the assignment sample plan.",
            )
        )
    blocked_reasons = [
        (
            f"{check['name']} observed {check['observed']}, "
            f"required {'<= ' if check.get('comparison') == 'at_most' else ''}{check['required']}"
        )
        for check in checks
        if check["status"] != "pass"
    ]
    status = "pass" if not blocked_reasons else "fail"
    return {
        "status": status,
        "ready_for_skill_update": status == "pass",
        "requirements": {
            "min_sampled_targets": int(requirements.get("min_sampled_targets", 0)),
            "min_non_github_lanes": int(requirements.get("min_non_github_lanes", 0)),
            "required_source_lanes": list(requirements.get("required_source_lanes", [])),
            "require_explicit_sampling_receipts": bool(requirements.get("require_explicit_sampling_receipts")),
        },
        "checks": checks,
        "blocked_reasons": blocked_reasons,
    }


def _has_non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return bool(value)


def promotion_blockers(finding: dict[str, Any]) -> list[str]:
    blockers = []
    sources = finding.get("sources", [])
    if not isinstance(sources, list) or not sources:
        blockers.append("missing source provenance")
    elif any(not isinstance(source, dict) or not source.get("url") or not source.get("accessed") for source in sources):
        blockers.append("source provenance lacks url/accessed")
    if not _has_non_empty(finding.get("affects")):
        blockers.append("missing affected skill asset or runbook")
    if not _has_non_empty(finding.get("validation_gate")):
        blockers.append("missing validation gate")
    if not _has_non_empty(finding.get("required_next_validation")):
        blockers.append("missing required next validation")
    if not (
        _has_non_empty(finding.get("rollback_condition"))
        or _has_non_empty(finding.get("rollback_conditions"))
        or _has_non_empty(finding.get("hold_condition"))
        or _has_non_empty(finding.get("caveats"))
    ):
        blockers.append("missing rollback or caveat metadata")
    if finding.get("source_lane") == "community_discussions":
        blockers.append("community-only evidence cannot be promoted")
    return blockers


def promotion_review_entry(
    finding: dict[str, Any],
    promotion_status: str,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    entry = {
        "id": finding["id"],
        "title": finding["title"],
        "summary": finding["summary"],
        "decision": finding["decision"],
        "promotion_status": promotion_status,
        "source_lane": finding["source_lane"],
        "persona_id": finding.get("persona_id"),
        "evidence_level": finding.get("evidence_level"),
        "source_count": len(finding.get("sources", [])),
        "sources": finding.get("sources", []),
        "affects": finding.get("affects", []),
        "validation_gate": finding.get("validation_gate"),
        "required_next_validation": finding.get("required_next_validation"),
        "caveats": finding.get("caveats", []),
        "promotion_blockers": blockers or [],
    }
    for optional in ("rollback_condition", "rollback_conditions", "hold_condition"):
        if optional in finding:
            entry[optional] = finding[optional]
    return entry


def build_promotion_review(config: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    promotion_ready = []
    validation_backlog = []
    rejected = []
    for finding in findings:
        decision = finding["decision"]
        if decision == "adopted":
            blockers = promotion_blockers(finding)
            if blockers:
                validation_backlog.append(
                    promotion_review_entry(finding, "adopted-needs-promotion-metadata", blockers)
                )
            else:
                promotion_ready.append(promotion_review_entry(finding, "promotion-ready"))
        elif decision in {"held", "needs-validation"}:
            validation_backlog.append(
                promotion_review_entry(
                    finding,
                    "validation-backlog",
                    [f"decision is {decision}"],
                )
            )
        elif decision == "rejected":
            rejected.append(promotion_review_entry(finding, "rejected"))
    review_policy = config.get("review_policy", {})
    return {
        "schema_version": 1,
        "review_only": True,
        "auto_modify_recommendations": bool(review_policy.get("auto_modify_recommendations", False)),
        "auto_promote_sources": bool(review_policy.get("auto_promote_sources", False)),
        "promotion_ready_count": len(promotion_ready),
        "validation_backlog_count": len(validation_backlog),
        "rejected_count": len(rejected),
        "promotion_ready": promotion_ready,
        "validation_backlog": validation_backlog,
        "rejected": rejected,
        "promotion_ready_requires": [
            "source URL and access date",
            "affected skill asset or runbook",
            "validation gate",
            "required next validation",
            "rollback or caveat metadata",
            "non-community-only evidence",
        ],
    }


def summarize(
    config: dict[str, Any],
    assignments: list[dict[str, Any]],
    planner_receipt: dict[str, Any],
    fixture: dict[str, Any],
    run_id: str,
    objective: str,
    executions: dict[str, dict[str, Any]] | None = None,
    review_requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    findings = flatten_findings(fixture)
    decision_counts = {state["id"]: 0 for state in config["decision_states"]}
    lanes: dict[str, int] = {lane["id"]: 0 for lane in config["source_lanes"]}
    for finding in findings:
        decision_counts[finding["decision"]] += 1
        lanes[finding["source_lane"]] += 1
    non_github_lanes = sorted(lane for lane, count in lanes.items() if count and lane != "repositories")
    returned_personas = {agent.get("persona_id") for agent in fixture.get("agents", [])}
    execution_counts = {
        "scaffolded_not_run": 0,
        "fixture_ingested": 0,
        "executor_completed": 0,
        "executor_failed": 0,
        "subagent_result_ingested": 0,
        "subagent_result_missing": 0,
    }
    executions = executions or {}
    for assignment in assignments:
        if assignment["persona_id"] in executions:
            assignment["execution"] = executions[assignment["persona_id"]]
            state = assignment["execution"].get("state", "executor_failed")
            execution_counts[state] = execution_counts.get(state, 0) + 1
        elif assignment["persona_id"] in returned_personas:
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
    sampling_coverage = sampling_coverage_summary(assignments, fixture)
    review_gate = build_review_gate(
        sampling_coverage,
        lanes,
        non_github_lanes,
        review_requirements or {},
    )
    promotion_review = build_promotion_review(config, findings)
    planned_summary = planned_sampling_summary(assignments, config)
    evidence_matrix = build_evidence_matrix(config, findings, sampling_coverage, planned_summary)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "objective": objective,
        "generated_at": utc_now(),
        "review_only": True,
        "agent_count": len(assignments),
        "gap_hints": planner_receipt.get("gap_hints", []),
        "execution_counts": execution_counts,
        "finding_count": len(findings),
        "source_lane_counts": lanes,
        "non_github_lanes_covered": non_github_lanes,
        "sampling_coverage": sampling_coverage,
        "review_gate": review_gate,
        "promotion_review": promotion_review,
        "evidence_matrix": evidence_matrix,
        "assignment_planner": planner_receipt,
        **planned_summary,
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


def unique_ordered(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def review_gate_requirements(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.min_sampled_targets < 0:
        raise SkillError("--min-sampled-targets must be non-negative")
    if args.min_non_github_lanes < 0:
        raise SkillError("--min-non-github-lanes must be non-negative")
    required_source_lanes = unique_ordered(args.require_source_lane)
    known_lanes = {lane["id"] for lane in config["source_lanes"]}
    unknown = sorted(lane_id for lane_id in required_source_lanes if lane_id not in known_lanes)
    if unknown:
        raise SkillError(f"--require-source-lane references unknown lanes: {', '.join(unknown)}")
    return {
        "min_sampled_targets": args.min_sampled_targets,
        "min_non_github_lanes": args.min_non_github_lanes,
        "required_source_lanes": required_source_lanes,
        "require_explicit_sampling_receipts": bool(args.require_explicit_sampling_receipts),
    }


def add_terms(counter: Counter[str], values: Any, weight: int, previous_terms: set[str]) -> None:
    for term in collect_terms(values):
        if term in previous_terms:
            continue
        counter[term] += weight


def derive_next_gap_hints(summary: dict[str, Any], previous_hints: list[str], limit: int = 10) -> list[str]:
    previous_terms = set()
    for hint in previous_hints:
        previous_terms.update(tokenize(hint))
    counter: Counter[str] = Counter()
    findings = list(summary.get("needs_validation", [])) + list(summary.get("held", []))
    for finding in findings:
        source_lane = str(finding.get("source_lane", ""))
        if source_lane:
            counter[source_lane] += 4
        title_fields: list[Any] = [
            finding.get("id", ""),
            finding.get("title", ""),
        ]
        detail_fields: list[Any] = [
            finding.get("summary", ""),
            finding.get("validation_gate", ""),
            finding.get("required_next_validation", ""),
            finding.get("affects", []),
            finding.get("caveats", []),
        ]
        for term in collect_terms(title_fields):
            if term in previous_terms:
                continue
            counter[term] += 2
        for term in collect_terms(detail_fields):
            if term in previous_terms:
                continue
            counter[term] += 1
    review_gate = summary.get("review_gate", {})
    if review_gate.get("status") != "pass":
        for reason in review_gate.get("blocked_reasons", []):
            add_terms(counter, reason, 3, previous_terms)
        for check in review_gate.get("checks", []):
            if check.get("status") == "pass":
                continue
            name = str(check.get("name", ""))
            if name.startswith("required_source_lane:"):
                lane_id = name.split(":", 1)[1]
                if lane_id not in previous_hints:
                    counter[lane_id] += 5
            add_terms(counter, [name, check.get("detail", "")], 2, previous_terms)
        for assignment in summary.get("sampling_coverage", {}).get("assignments", []):
            for target in assignment.get("unsampled_targets", []):
                lane_id = str(target.get("source_lane", ""))
                if lane_id and lane_id not in previous_hints:
                    counter[lane_id] += 4
                add_terms(
                    counter,
                    [target.get("title", ""), target.get("kind", ""), target.get("locator", "")],
                    2,
                    previous_terms,
                )
    if not counter:
        return []
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def merge_gap_hints(existing: list[str], derived: list[str]) -> list[str]:
    return unique_ordered(existing + derived)


def markdown_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def blog_required_sections(config: dict[str, Any]) -> list[str]:
    configured = [
        str(section)
        for section in config.get("blog_sections", [])
        if str(section).strip()
    ]
    return unique_ordered(BLOG_REQUIRED_SECTIONS + configured)


def markdown_heading_sections(text: str) -> set[str]:
    sections = set()
    for line in text.splitlines():
        match = re.match(r"^\s*#{2,6}\s+(.+?)\s*$", line)
        if match:
            sections.add(match.group(1).strip().lower())
    return sections


def blog_contract_receipt(text: str, required_sections: list[str]) -> dict[str, Any]:
    present_headings = markdown_heading_sections(text)
    present = [section for section in required_sections if section.lower() in present_headings]
    missing = [section for section in required_sections if section.lower() not in present_headings]
    return {
        "required_sections": required_sections,
        "present_sections": present,
        "missing_sections": missing,
        "contract_status": "pass" if not missing else "fail",
    }


def summarize_blog_contract(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [receipt for receipt in receipts if receipt.get("contract_status") != "pass"]
    worker_failed = [
        receipt
        for receipt in failed
        if receipt.get("source") != "generated"
    ]
    return {
        "schema_version": 1,
        "required_sections": receipts[0].get("required_sections", []) if receipts else [],
        "blog_count": len(receipts),
        "passing_count": len(receipts) - len(failed),
        "failed_count": len(failed),
        "worker_authored_failed_count": len(worker_failed),
        "failed_blogs": [
            {
                "persona_id": receipt["persona_id"],
                "path": receipt["path"],
                "source": receipt["source"],
                "missing_sections": receipt.get("missing_sections", []),
            }
            for receipt in failed
        ],
    }


def blog_contract_failed(receipt: dict[str, Any]) -> bool:
    return any(
        blog.get("source") != "generated"
        for blog in receipt.get("blog_contract", {}).get("failed_blogs", [])
    )


def blog_contract_failure_message(receipt: dict[str, Any]) -> str:
    failures = [
        f"{blog['persona_id']} missing {', '.join(blog.get('missing_sections', [])) or 'required sections'}"
        for blog in receipt.get("blog_contract", {}).get("failed_blogs", [])
        if blog.get("source") != "generated"
    ]
    return "worker blog contract failed: " + "; ".join(failures or ["worker-authored blog contract did not pass"])


def write_blogs(
    output_dir: Path,
    assignments: list[dict[str, Any]],
    fixture: dict[str, Any],
    required_sections: list[str],
) -> list[dict[str, Any]]:
    by_persona = {agent.get("persona_id"): agent for agent in fixture.get("agents", [])}
    blog_dir = output_dir / "blogs"
    blog_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
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
        planned_sampling = []
        for lane in assignment.get("sample_plan", []):
            for target in lane.get("targets", []):
                planned_sampling.append(
                    f"{lane['source_lane']}: {target['title']} [{target['kind']}] - {target['locator']}"
                )
        coverage = assignment.get("sampling_coverage", {})
        sampled_target_lines = [
            f"{target['source_lane']}: {target['title']} -> {', '.join(source['url'] for source in target.get('matched_sources', []))}"
            for target in coverage.get("sampled_targets", [])
        ]
        unsampled_target_lines = [
            f"{target['source_lane']}: {target['title']} [{target['kind']}] - {target['locator']}"
            for target in coverage.get("unsampled_targets", [])
        ]
        unplanned_source_lines = [
            f"{source.get('finding_id')}: {source.get('title') or source.get('url')} ({source.get('url')})"
            for source in coverage.get("unplanned_sources", [])
        ]
        invalid_receipt_lines = [
            (
                f"{receipt.get('finding_id')}: claimed "
                f"{receipt.get('sampled_target_title') or receipt.get('sampled_target_locator') or '<missing>'} "
                f"from {receipt.get('url') or receipt.get('title')}"
            )
            for receipt in coverage.get("invalid_sampling_receipts", [])
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
            "## Planned sampling",
            markdown_list(planned_sampling),
            "",
            "## Sources sampled",
            markdown_list(sampled),
            "",
            "## Sampling coverage",
            f"Planned targets: {coverage.get('planned_target_count', 0)}",
            f"Matched targets: {coverage.get('sampled_target_count', 0)}",
            f"Unplanned returned sources: {coverage.get('unplanned_source_count', 0)}",
            f"Explicit sampling receipts: {coverage.get('valid_explicit_sampling_receipt_count', 0)} valid, {coverage.get('invalid_explicit_sampling_receipt_count', 0)} invalid",
            f"Matched targets missing explicit receipts: {coverage.get('sampled_target_missing_explicit_receipt_count', 0)}",
            "",
            "### Matched planned targets",
            markdown_list(sampled_target_lines),
            "",
            "### Unmatched planned targets",
            markdown_list(unsampled_target_lines),
            "",
            "### Unplanned returned sources",
            markdown_list(unplanned_source_lines),
            "",
            "### Invalid explicit sampling receipts",
            markdown_list(invalid_receipt_lines),
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
        paths = agent_output_paths(output_dir, assignment)
        blog_path = paths["blog"]
        generated_blog_path = paths["generated_blog"]
        execution = assignment.get("execution", {})
        worker_blog_present = (
            execution.get("kind") in {"local-executor", "external-subagent"}
            and execution.get("blog_source") == "executor-authored"
            and blog_path.exists()
            and blog_path.stat().st_size > 0
        )
        if worker_blog_present:
            generated_blog_path.write_text(text, encoding="utf-8")
            receipt_text = blog_path.read_text(encoding="utf-8")
            source = "executor-authored"
            generated_rel = rel_output_path(output_dir, generated_blog_path)
        else:
            blog_path.write_text(text, encoding="utf-8")
            receipt_text = text
            source = "generated"
            generated_rel = rel_output_path(output_dir, blog_path)
        contract = blog_contract_receipt(receipt_text, required_sections)
        receipt = {
            "persona_id": assignment["persona_id"],
            "path": rel_output_path(output_dir, blog_path),
            "source": source,
            "generated_blog_path": generated_rel,
            "preserved_executor_blog": worker_blog_present,
            **contract,
        }
        execution["blog_path"] = receipt["path"]
        execution["blog_source"] = source
        assignment["blog"] = receipt
        if "handoff" in assignment:
            assignment["handoff"]["blog_source"] = source
        receipts.append(receipt)
    return receipts


def write_markdown_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    review_gate = summary.get("review_gate", {})
    promotion_review = summary.get("promotion_review", {})
    blog_contract = summary.get("blog_contract", {})
    evidence_matrix = summary.get("evidence_matrix", {})
    lines = [
        f"# Research Loop {summary['run_id']}",
        "",
        f"Objective: {summary['objective']}",
        "",
        f"Review only: {summary['review_only']}",
        f"Iteration: {summary.get('iteration', 1)} of {summary.get('iteration_count', 1)}",
        f"Gap hints used: {', '.join(summary.get('gap_hints', [])) or 'none'}",
        f"Next gap hints: {', '.join(summary.get('next_gap_hints', [])) or 'none'}",
        f"Findings: {summary['finding_count']}",
        f"Sampling coverage: {summary['sampling_coverage']['sampled_target_count']}/{summary['sampling_coverage']['planned_target_count']} planned targets",
        f"Unplanned returned sources: {summary['sampling_coverage']['unplanned_source_count']}",
        (
            "Explicit sampling receipts: "
            f"{summary['sampling_coverage'].get('valid_explicit_sampling_receipt_count', 0)} valid, "
            f"{summary['sampling_coverage'].get('invalid_explicit_sampling_receipt_count', 0)} invalid, "
            f"{summary['sampling_coverage'].get('sampled_target_missing_explicit_receipt_count', 0)} matched targets missing receipts"
        ),
        f"Review gate: {review_gate.get('status', 'unknown')}",
        f"Assignment planner: {summary['assignment_planner']['mode']}",
        f"Non-GitHub lanes covered: {', '.join(summary['non_github_lanes_covered']) or 'none'}",
        f"Planned non-GitHub sample targets: {summary['planned_non_github_sample_target_count']}",
        "",
        "## Selected Agents",
    ]
    for selected in summary["assignment_planner"].get("selected_personas", []):
        lines.append(
            f"- {selected['persona_id']}: score {selected['score']} - {'; '.join(selected.get('reasons', []))}"
        )
    lines.extend([
        "",
        "## Decision Counts",
    ])
    for decision, count in summary["decision_counts"].items():
        lines.append(f"- {decision}: {count}")
    lines.extend(["", "## Review Gate"])
    lines.append(f"- status: {review_gate.get('status', 'unknown')}")
    lines.append(
        f"- ready for skill update: {str(review_gate.get('ready_for_skill_update', False)).lower()}"
    )
    lines.append("- blocked reasons:")
    for reason in review_gate.get("blocked_reasons", []) or ["None"]:
        lines.append(f"  - {reason}")
    lines.extend([
        "",
        "## Evidence Matrix",
        "- review-only: true",
        f"- unique sources: {evidence_matrix.get('unique_source_count', 0)}",
        f"- source citations: {evidence_matrix.get('source_citation_count', 0)}",
        f"- citation policy: {evidence_matrix.get('citation_policy', 'Repeated source citation is review context only.')}",
        "",
        "### Source Lanes",
    ])
    lane_rows = []
    for lane in evidence_matrix.get("source_lanes", []):
        lane_rows.append(
            (
                f"{lane['source_lane']}: {lane['unique_source_count']} unique sources, "
                f"{lane['source_citation_count']} citations, "
                f"{lane['sampled_target_count']}/{lane['planned_target_count']} sampled targets "
                f"({lane['status']})"
            )
        )
    lines.append(markdown_list(lane_rows))
    lines.extend(["", "### Top Cited Sources"])
    top_rows = []
    for source in evidence_matrix.get("top_sources", []):
        top_rows.append(
            (
                f"{source['title']} - {source['citation_count']} citation(s), "
                f"findings: {', '.join(source.get('finding_ids', [])) or 'none'}"
            )
        )
    lines.append(markdown_list(top_rows))
    lines.extend(["", "### Thin Source Lanes"])
    thin_rows = []
    for lane in evidence_matrix.get("thin_source_lanes", []) + evidence_matrix.get("uncited_source_lanes", []):
        thin_rows.append(
            (
                f"{lane['source_lane']}: {lane['sampled_target_count']}/"
                f"{lane['planned_target_count']} sampled targets, "
                f"{lane['source_citation_count']} source citations ({lane['status']})"
            )
        )
    lines.append(markdown_list(thin_rows))
    lines.extend([
        "",
        "## Promotion Review",
        "- review-only: true",
        f"- auto modify recommendations: {str(promotion_review.get('auto_modify_recommendations', False)).lower()}",
        f"- auto promote sources: {str(promotion_review.get('auto_promote_sources', False)).lower()}",
        f"- promotion ready: {promotion_review.get('promotion_ready_count', 0)}",
        f"- validation backlog: {promotion_review.get('validation_backlog_count', 0)}",
        f"- rejected: {promotion_review.get('rejected_count', 0)}",
        "",
        "### Promotion Ready",
    ])
    ready_rows = []
    for finding in promotion_review.get("promotion_ready", []):
        ready_rows.append(
            f"{finding['id']}: {finding['title']} - gate: {finding.get('validation_gate') or 'missing'}"
        )
    lines.append(markdown_list(ready_rows))
    lines.extend(["", "### Validation Backlog"])
    backlog_rows = []
    for finding in promotion_review.get("validation_backlog", []):
        blockers = ", ".join(finding.get("promotion_blockers", [])) or "needs review"
        backlog_rows.append(f"{finding['id']} ({finding['decision']}): {finding['title']} - {blockers}")
    lines.append(markdown_list(backlog_rows))
    lines.extend(["", "### Rejected"])
    rejected_rows = []
    for finding in promotion_review.get("rejected", []):
        rejected_rows.append(f"{finding['id']}: {finding['title']} - {finding['summary']}")
    lines.append(markdown_list(rejected_rows))
    lines.extend(["", "## Blog Receipts"])
    lines.append(
        f"- contract: {blog_contract.get('passing_count', 0)}/"
        f"{blog_contract.get('blog_count', 0)} passing; "
        f"{blog_contract.get('worker_authored_failed_count', 0)} worker-authored failed"
    )
    for receipt in summary.get("blog_receipts", []):
        lines.append(
            f"- {receipt['persona_id']}: {receipt['source']} at {receipt['path']} "
            f"({receipt.get('contract_status', 'unknown')})"
        )
        if receipt.get("missing_sections"):
            lines.append(f"  - missing sections: {', '.join(receipt['missing_sections'])}")
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


def run_iteration(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_id: str,
    output_dir: Path,
    gap_hints: list[str],
    iteration: int,
    iteration_count: int,
    review_requirements: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments, planner_receipt = build_assignments(
        config,
        args.objective,
        run_id,
        args.agent_count,
        gap_hints,
        args.assignment_mode,
    )
    if args.executor_command:
        execution_mode = "local-executor"
    elif args.ingest_subagent_results:
        execution_mode = "external-subagent"
    elif args.offline_fixture:
        execution_mode = "offline-fixture"
    else:
        execution_mode = "offline-scaffold"
    write_subagent_handoffs(
        output_dir,
        assignments,
        planner_receipt,
        run_id,
        args.objective,
        iteration,
        iteration_count,
        execution_mode,
        args.executor_command,
    )
    executions = None
    if args.executor_command:
        fixture, executions = execute_assignments(
            assignments,
            args.executor_command,
            output_dir,
            run_id,
            args.execution_timeout,
            args.executor_workers,
        )
    elif args.ingest_subagent_results:
        fixture, executions = ingest_subagent_results(assignments, output_dir)
    else:
        fixture = load_fixture(args.offline_fixture)
    validate_findings(config, fixture)
    summary = summarize(
        config,
        assignments,
        planner_receipt,
        fixture,
        run_id,
        args.objective,
        executions,
        review_requirements,
    )
    summary["iteration"] = iteration
    summary["iteration_count"] = iteration_count
    summary["gap_hints"] = gap_hints
    summary["next_gap_hints"] = derive_next_gap_hints(summary, gap_hints)
    summary["next_wave_expected"] = iteration < iteration_count
    if args.offline_fixture:
        fixture_path = Path(args.offline_fixture)
        summary["offline_fixture"] = safe_relpath(Path.cwd(), fixture_path) if fixture_path.exists() else args.offline_fixture
    if args.executor_command:
        summary["executor_command"] = args.executor_command
        summary["executor_workers"] = args.executor_workers
        summary["executor_actual_workers"] = fixture.get("executor_actual_workers", args.executor_workers)
    if args.ingest_subagent_results:
        summary["ingested_subagent_results"] = True
    summary["fail_on_review_gate"] = bool(args.fail_on_review_gate)
    summary["require_worker_blog_contract"] = bool(args.require_worker_blog_contract)
    summary["blog_receipts"] = write_blogs(output_dir, assignments, fixture, blog_required_sections(config))
    summary["blog_contract"] = summarize_blog_contract(summary["blog_receipts"])
    next_wave = build_next_wave_scaffold(summary, args, output_dir)
    if next_wave:
        summary["next_wave_scaffold"] = next_wave
    summary["subagent_dispatch"] = write_subagent_handoffs(
        output_dir,
        assignments,
        planner_receipt,
        run_id,
        args.objective,
        iteration,
        iteration_count,
        execution_mode,
        args.executor_command,
    )
    summary["campaign_manifest"] = write_campaign_receipts(
        output_dir,
        build_campaign_manifest(
            run_id,
            args.objective,
            output_dir,
            [(summary, output_dir)],
            iteration_count,
            args.until_review_gate,
            "single_iteration" if iteration_count == 1 else "wave_ready",
        ),
    )
    dump_json({
        "schema_version": 1,
        "run_id": run_id,
        "objective": args.objective,
        "generated_at": summary["generated_at"],
        "iteration": iteration,
        "iteration_count": iteration_count,
        "gap_hints": gap_hints,
        "next_gap_hints": summary["next_gap_hints"],
        "assignment_planner": planner_receipt,
        "assignments": assignments,
    }, output_dir / "assignments.json")
    dump_json(summary, output_dir / "synthesis.json")
    write_markdown_summary(output_dir, summary)
    return summary


def loop_iteration_record(root: Path, summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    promotion_review = summary.get("promotion_review", {})
    evidence_matrix = summary.get("evidence_matrix", {})
    return {
        "iteration": summary["iteration"],
        "run_id": summary["run_id"],
        "output_dir": safe_relpath(root, output_dir),
        "gap_hints": summary.get("gap_hints", []),
        "next_gap_hints": summary.get("next_gap_hints", []),
        "assignment_mode": summary.get("assignment_planner", {}).get("mode"),
        "sampling_coverage": {
            "planned_target_count": summary.get("sampling_coverage", {}).get("planned_target_count", 0),
            "sampled_target_count": summary.get("sampling_coverage", {}).get("sampled_target_count", 0),
            "unplanned_source_count": summary.get("sampling_coverage", {}).get("unplanned_source_count", 0),
            "explicit_sampling_receipt_count": summary.get("sampling_coverage", {}).get("explicit_sampling_receipt_count", 0),
            "valid_explicit_sampling_receipt_count": summary.get("sampling_coverage", {}).get("valid_explicit_sampling_receipt_count", 0),
            "invalid_explicit_sampling_receipt_count": summary.get("sampling_coverage", {}).get("invalid_explicit_sampling_receipt_count", 0),
            "sampled_target_missing_explicit_receipt_count": summary.get("sampling_coverage", {}).get("sampled_target_missing_explicit_receipt_count", 0),
        },
        "executor_workers": summary.get("executor_workers"),
        "executor_actual_workers": summary.get("executor_actual_workers"),
        "selected_personas": [
            row.get("persona_id") for row in summary.get("assignment_planner", {}).get("selected_personas", [])
        ],
        "finding_count": summary.get("finding_count", 0),
        "decision_counts": summary.get("decision_counts", {}),
        "non_github_lanes_covered": summary.get("non_github_lanes_covered", []),
        "execution_counts": summary.get("execution_counts", {}),
        "review_gate": summary.get("review_gate", {}),
        "blog_contract": summary.get("blog_contract", {}),
        "evidence_matrix": {
            "schema_version": evidence_matrix.get("schema_version", 1),
            "review_only": True,
            "unique_source_count": int(evidence_matrix.get("unique_source_count", 0)),
            "source_citation_count": int(evidence_matrix.get("source_citation_count", 0)),
            "source_lane_counts": evidence_matrix.get("source_lane_counts", {}),
            "finding_decision_counts": evidence_matrix.get("finding_decision_counts", {}),
            "thin_source_lanes": evidence_matrix.get("thin_source_lanes", []),
            "uncited_source_lanes": evidence_matrix.get("uncited_source_lanes", []),
            "top_sources": evidence_matrix.get("top_sources", []),
        },
        "promotion_review": {
            "schema_version": promotion_review.get("schema_version", 1),
            "review_only": True,
            "auto_modify_recommendations": bool(promotion_review.get("auto_modify_recommendations", False)),
            "auto_promote_sources": bool(promotion_review.get("auto_promote_sources", False)),
            "promotion_ready_count": int(promotion_review.get("promotion_ready_count", 0)),
            "validation_backlog_count": int(promotion_review.get("validation_backlog_count", 0)),
            "rejected_count": int(promotion_review.get("rejected_count", 0)),
            "promotion_ready": promotion_review.get("promotion_ready", []),
            "validation_backlog": promotion_review.get("validation_backlog", []),
            "rejected": promotion_review.get("rejected", []),
        },
    }


def aggregate_review_gate(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode == "final_iteration":
        final = records[-1] if records else {}
        final_gate = final.get("review_gate", {})
        final_passed = final_gate.get("status") == "pass"
        return {
            "status": "pass" if final_passed else "fail",
            "ready_for_skill_update": final_passed,
            "requirements": {
                "final_iteration_pass": True,
            },
            "checks": [
                review_check(
                    "final_iteration_review_gate",
                    1 if final_passed else 0,
                    1,
                    "Adaptive loops require the final completed iteration to pass its review gate.",
                )
            ],
            "blocked_reasons": [
                f"final iteration {final.get('iteration')}: {reason}"
                for reason in final_gate.get("blocked_reasons", []) or ["review gate did not pass"]
            ] if not final_passed else [],
            "iteration_statuses": [
                {
                    "iteration": record.get("iteration"),
                    "run_id": record.get("run_id"),
                    "status": record.get("review_gate", {}).get("status", "fail"),
                    "ready_for_skill_update": record.get("review_gate", {}).get("ready_for_skill_update", False),
                }
                for record in records
            ],
        }

    passed = 0
    blocked_reasons = []
    iteration_statuses = []
    for record in records:
        gate = record.get("review_gate", {})
        status = gate.get("status", "fail")
        if status == "pass":
            passed += 1
        else:
            reasons = gate.get("blocked_reasons", []) or ["review gate did not pass"]
            for reason in reasons:
                blocked_reasons.append(f"iteration {record.get('iteration')}: {reason}")
        iteration_statuses.append({
            "iteration": record.get("iteration"),
            "run_id": record.get("run_id"),
            "status": status,
            "ready_for_skill_update": gate.get("ready_for_skill_update", False),
        })
    status = "pass" if passed == len(records) else "fail"
    return {
        "status": status,
        "ready_for_skill_update": status == "pass",
        "requirements": {
            "all_iterations_pass": True,
        },
        "checks": [
            review_check(
                "iterations_passing_review_gate",
                passed,
                len(records),
                "All iterations must pass their per-run review gate.",
            )
        ],
        "blocked_reasons": blocked_reasons,
        "iteration_statuses": iteration_statuses,
    }


def build_loop_summary(
    run_id: str,
    objective: str,
    output_dir: Path,
    records: list[dict[str, Any]],
    final_gap_hints: list[str],
    iteration_cap: int,
    stop_reason: str,
    until_review_gate: bool,
) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    total_findings = 0
    sampling_counts: Counter[str] = Counter()
    for record in records:
        total_findings += int(record.get("finding_count", 0))
        decision_counts.update(record.get("decision_counts", {}))
        execution_counts.update(record.get("execution_counts", {}))
        sampling_counts.update(record.get("sampling_coverage", {}))
    sampling_coverage = dict(sorted(sampling_counts.items()))
    review_gate = aggregate_review_gate(records, "final_iteration" if until_review_gate else "all_iterations")
    blog_contract = aggregate_loop_blog_contract(records)
    promotion_review = aggregate_loop_promotion_review(records)
    learning_dossier = aggregate_loop_learning_dossier(
        records,
        sampling_coverage,
        review_gate,
        blog_contract,
        promotion_review,
        final_gap_hints,
        stop_reason,
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "objective": objective,
        "generated_at": utc_now(),
        "review_only": True,
        "iteration_count": len(records),
        "iteration_cap": iteration_cap,
        "until_review_gate": until_review_gate,
        "stop_reason": stop_reason,
        "stopped_after_review_gate": stop_reason == "review_gate_passed",
        "iteration_cap_exhausted": stop_reason == "iteration_cap_exhausted",
        "total_finding_count": total_findings,
        "decision_counts": dict(sorted(decision_counts.items())),
        "execution_counts": dict(sorted(execution_counts.items())),
        "sampling_coverage": sampling_coverage,
        "review_gate": review_gate,
        "blog_contract": blog_contract,
        "promotion_review": promotion_review,
        "learning_dossier": learning_dossier,
        "iterations": records,
        "final_gap_hints": final_gap_hints,
        "instructions": [
            "This loop receipt summarizes review-only research iterations.",
            "Promote findings only through explicit source, validation, tests, and manifest updates.",
            "Do not execute remote model code while investigating candidates.",
        ],
    }


def aggregate_loop_blog_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    failed_blogs = []
    blog_count = 0
    passing_count = 0
    failed_count = 0
    worker_authored_failed_count = 0
    required_sections = []
    for record in records:
        contract = record.get("blog_contract", {})
        blog_count += int(contract.get("blog_count", 0))
        passing_count += int(contract.get("passing_count", 0))
        failed_count += int(contract.get("failed_count", 0))
        worker_authored_failed_count += int(contract.get("worker_authored_failed_count", 0))
        if not required_sections:
            required_sections = contract.get("required_sections", [])
        for blog in contract.get("failed_blogs", []):
            row = dict(blog)
            row["iteration"] = record.get("iteration")
            row["iteration_run_id"] = record.get("run_id")
            row["iteration_output_dir"] = record.get("output_dir")
            failed_blogs.append(row)
    return {
        "schema_version": 1,
        "required_sections": required_sections,
        "blog_count": blog_count,
        "passing_count": passing_count,
        "failed_count": failed_count,
        "worker_authored_failed_count": worker_authored_failed_count,
        "failed_blogs": failed_blogs,
    }


def with_iteration_context(entry: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    row = dict(entry)
    row["iteration"] = record.get("iteration")
    row["iteration_run_id"] = record.get("run_id")
    row["iteration_output_dir"] = record.get("output_dir")
    return row


def aggregate_loop_promotion_review(records: list[dict[str, Any]]) -> dict[str, Any]:
    promotion_ready = []
    validation_backlog = []
    rejected = []
    auto_modify = False
    auto_promote = False
    for record in records:
        review = record.get("promotion_review", {})
        auto_modify = auto_modify or bool(review.get("auto_modify_recommendations", False))
        auto_promote = auto_promote or bool(review.get("auto_promote_sources", False))
        promotion_ready.extend(with_iteration_context(entry, record) for entry in review.get("promotion_ready", []))
        validation_backlog.extend(with_iteration_context(entry, record) for entry in review.get("validation_backlog", []))
        rejected.extend(with_iteration_context(entry, record) for entry in review.get("rejected", []))
    return {
        "schema_version": 1,
        "review_only": True,
        "auto_modify_recommendations": auto_modify,
        "auto_promote_sources": auto_promote,
        "promotion_ready_count": len(promotion_ready),
        "validation_backlog_count": len(validation_backlog),
        "rejected_count": len(rejected),
        "promotion_ready": promotion_ready,
        "validation_backlog": validation_backlog,
        "rejected": rejected,
        "aggregation_rule": "Entries are copied from per-iteration synthesis promotion_review ledgers and tagged with iteration context.",
    }


def learning_action(action_id: str, kind: str, action: str, reason: str, priority: str = "medium") -> dict[str, Any]:
    return {
        "id": action_id,
        "kind": kind,
        "priority": priority,
        "action": action,
        "reason": reason,
    }


def aggregate_loop_learning_dossier(
    records: list[dict[str, Any]],
    sampling_coverage: dict[str, Any],
    review_gate: dict[str, Any],
    blog_contract: dict[str, Any],
    promotion_review: dict[str, Any],
    final_gap_hints: list[str],
    stop_reason: str,
) -> dict[str, Any]:
    lane_citations: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    unique_sources: set[str] = set()
    unique_source_observation_count = 0
    source_citation_count = 0
    top_sources_by_key: dict[str, dict[str, Any]] = {}
    thin_lanes_by_id: dict[str, dict[str, Any]] = {}
    uncited_lanes_by_id: dict[str, dict[str, Any]] = {}
    non_github_lanes = set()

    for record in records:
        non_github_lanes.update(record.get("non_github_lanes_covered", []))
        matrix = record.get("evidence_matrix", {})
        lane_citations.update(matrix.get("source_lane_counts", {}))
        decision_counts.update(matrix.get("finding_decision_counts", {}))
        unique_source_observation_count += int(matrix.get("unique_source_count", 0))
        source_citation_count += int(matrix.get("source_citation_count", 0))
        for source in matrix.get("top_sources", []):
            key = source.get("key")
            if key:
                unique_sources.add(str(key))
                existing = top_sources_by_key.setdefault(str(key), {
                    "key": key,
                    "title": source.get("title", key),
                    "url": source.get("url", ""),
                    "citation_count": 0,
                    "finding_ids": [],
                    "source_lanes": [],
                })
                existing["citation_count"] += int(source.get("citation_count", 0))
                existing["finding_ids"] = unique_ordered(existing["finding_ids"] + source.get("finding_ids", []))
                existing["source_lanes"] = unique_ordered(existing["source_lanes"] + source.get("source_lanes", []))
        for field, target in (
            ("thin_source_lanes", thin_lanes_by_id),
            ("uncited_source_lanes", uncited_lanes_by_id),
        ):
            for lane in matrix.get(field, []):
                lane_id = str(lane.get("source_lane", ""))
                if not lane_id:
                    continue
                current = target.setdefault(lane_id, {
                    "source_lane": lane_id,
                    "planned_target_count": 0,
                    "sampled_target_count": 0,
                    "source_citation_count": 0,
                    "iterations": [],
                    "status": lane.get("status", field.replace("_source_lanes", "")),
                })
                current["planned_target_count"] += int(lane.get("planned_target_count", 0))
                current["sampled_target_count"] += int(lane.get("sampled_target_count", 0))
                current["source_citation_count"] += int(lane.get("source_citation_count", 0))
                current["iterations"] = unique_ordered(current["iterations"] + [str(record.get("iteration"))])

    top_sources = sorted(
        top_sources_by_key.values(),
        key=lambda row: (-int(row.get("citation_count", 0)), str(row.get("key", ""))),
    )[:10]
    validation_backlog = promotion_review.get("validation_backlog", [])
    actions: list[dict[str, Any]] = []
    seen_actions: set[str] = set()

    def add_action(row: dict[str, Any]) -> None:
        if row["id"] in seen_actions:
            return
        seen_actions.add(row["id"])
        actions.append(row)

    for hint in final_gap_hints[:10]:
        add_action(learning_action(
            f"gap:{slugify(hint)}",
            "gap_hint",
            f"Schedule a dynamic follow-up worker pass for `{hint}`.",
            "Final loop gap hints identify unresolved research terms.",
            "high",
        ))
    for lane in list(thin_lanes_by_id.values())[:10]:
        add_action(learning_action(
            f"thin-lane:{slugify(lane['source_lane'])}",
            "thin_source_lane",
            f"Sample more planned targets in `{lane['source_lane']}`.",
            f"{lane['sampled_target_count']}/{lane['planned_target_count']} planned targets were sampled across iterations.",
            "high",
        ))
    for lane in list(uncited_lanes_by_id.values())[:10]:
        add_action(learning_action(
            f"uncited-lane:{slugify(lane['source_lane'])}",
            "uncited_source_lane",
            f"Dispatch a researcher to collect evidence for `{lane['source_lane']}`.",
            "The lane had planned targets but no returned source citations.",
            "high",
        ))
    for entry in validation_backlog[:10]:
        add_action(learning_action(
            f"validation:{slugify(str(entry.get('id', 'finding')))}",
            "validation_backlog",
            f"Validate `{entry.get('id')}` before editing skill guidance.",
            ", ".join(entry.get("promotion_blockers", [])) or "Finding remains in validation backlog.",
            "medium",
        ))
    for blog in blog_contract.get("failed_blogs", [])[:10]:
        add_action(learning_action(
            f"blog:{slugify(str(blog.get('iteration', 'i')))}:{slugify(str(blog.get('persona_id', 'persona')))}",
            "blog_contract",
            f"Repair or rerun the blog for `{blog.get('persona_id')}`.",
            f"Missing sections: {', '.join(blog.get('missing_sections', [])) or 'required sections'}.",
            "medium",
        ))
    for reason in review_gate.get("blocked_reasons", [])[:10]:
        add_action(learning_action(
            f"gate:{slugify(reason)}",
            "review_gate",
            "Run another research wave to satisfy the review gate.",
            reason,
            "high",
        ))

    return {
        "schema_version": 1,
        "review_only": True,
        "auto_modify_recommendations": bool(promotion_review.get("auto_modify_recommendations", False)),
        "auto_promote_sources": bool(promotion_review.get("auto_promote_sources", False)),
        "stop_reason": stop_reason,
        "learned_findings": {
            "total_count": sum(int(record.get("finding_count", 0)) for record in records),
            "decision_counts": dict(sorted(decision_counts.items())),
            "promotion_ready_count": int(promotion_review.get("promotion_ready_count", 0)),
            "validation_backlog_count": int(promotion_review.get("validation_backlog_count", 0)),
            "rejected_count": int(promotion_review.get("rejected_count", 0)),
            "promotion_ready": promotion_review.get("promotion_ready", [])[:10],
        },
        "coverage_overview": {
            "iteration_count": len(records),
            "sampled_target_count": int(sampling_coverage.get("sampled_target_count", 0)),
            "planned_target_count": int(sampling_coverage.get("planned_target_count", 0)),
            "unplanned_source_count": int(sampling_coverage.get("unplanned_source_count", 0)),
            "valid_explicit_sampling_receipt_count": int(sampling_coverage.get("valid_explicit_sampling_receipt_count", 0)),
            "invalid_explicit_sampling_receipt_count": int(sampling_coverage.get("invalid_explicit_sampling_receipt_count", 0)),
            "unique_source_observation_count": unique_source_observation_count,
            "source_citation_count": source_citation_count,
            "unique_top_source_count": len(unique_sources),
            "non_github_lanes_covered": sorted(non_github_lanes),
            "source_lane_citation_counts": dict(sorted(lane_citations.items())),
            "top_sources": top_sources,
        },
        "blog_health": {
            "blog_count": int(blog_contract.get("blog_count", 0)),
            "passing_count": int(blog_contract.get("passing_count", 0)),
            "failed_count": int(blog_contract.get("failed_count", 0)),
            "worker_authored_failed_count": int(blog_contract.get("worker_authored_failed_count", 0)),
            "failed_blogs": blog_contract.get("failed_blogs", []),
        },
        "evidence_gaps": {
            "final_gap_hints": final_gap_hints,
            "thin_source_lanes": sorted(thin_lanes_by_id.values(), key=lambda row: row["source_lane"]),
            "uncited_source_lanes": sorted(uncited_lanes_by_id.values(), key=lambda row: row["source_lane"]),
            "review_gate_blocked_reasons": review_gate.get("blocked_reasons", []),
        },
        "validation_backlog": validation_backlog[:20],
        "next_research_actions": actions[:30],
        "instructions": [
            "This dossier is a review-only campaign handoff.",
            "Use next_research_actions to plan more sampling or validation before editing skill assets.",
            "Repeated citations, passed gates, or promotion-ready status still require explicit tests and rollback evidence before skill changes.",
        ],
    }


def promotion_review_markdown_rows(entries: list[dict[str, Any]], limit: int = 10) -> list[str]:
    rows = []
    for entry in entries[:limit]:
        blockers = ", ".join(entry.get("promotion_blockers", []))
        suffix = f" - {blockers}" if blockers else ""
        rows.append(
            f"i{entry.get('iteration')}: {entry.get('id')}: {entry.get('title')}{suffix}"
        )
    remaining = len(entries) - limit
    if remaining > 0:
        rows.append(f"... {remaining} more")
    return rows


def write_loop_markdown(output_dir: Path, loop_summary: dict[str, Any]) -> None:
    review_gate = loop_summary.get("review_gate", {})
    promotion_review = loop_summary.get("promotion_review", {})
    blog_contract = loop_summary.get("blog_contract", {})
    learning_dossier = loop_summary.get("learning_dossier", {})
    lines = [
        f"# Research Loop {loop_summary['run_id']}",
        "",
        f"Objective: {loop_summary['objective']}",
        f"Review only: {loop_summary['review_only']}",
        f"Iterations: {loop_summary['iteration_count']}",
        f"Iteration cap: {loop_summary.get('iteration_cap', loop_summary['iteration_count'])}",
        f"Adaptive until review gate: {str(loop_summary.get('until_review_gate', False)).lower()}",
        f"Stop reason: {loop_summary.get('stop_reason', 'fixed_iterations_complete')}",
        f"Total findings: {loop_summary['total_finding_count']}",
        f"Sampling coverage: {loop_summary.get('sampling_coverage', {}).get('sampled_target_count', 0)}/{loop_summary.get('sampling_coverage', {}).get('planned_target_count', 0)} planned targets",
        f"Unplanned returned sources: {loop_summary.get('sampling_coverage', {}).get('unplanned_source_count', 0)}",
        f"Review gate: {review_gate.get('status', 'unknown')}",
        f"Final gap hints: {', '.join(loop_summary.get('final_gap_hints', [])) or 'none'}",
        "",
        "## Iterations",
    ]
    for record in loop_summary["iterations"]:
        lines.append(
            f"- {record['iteration']}: {record['run_id']} in {record['output_dir']} "
            f"({record['assignment_mode']}, {record['finding_count']} findings)"
        )
        lines.append(f"  - gap hints: {', '.join(record.get('gap_hints', [])) or 'none'}")
        lines.append(f"  - next gap hints: {', '.join(record.get('next_gap_hints', [])) or 'none'}")
        lines.append(f"  - selected personas: {', '.join(record.get('selected_personas', [])) or 'none'}")
        coverage = record.get("sampling_coverage", {})
        lines.append(
            f"  - sampling coverage: {coverage.get('sampled_target_count', 0)}/"
            f"{coverage.get('planned_target_count', 0)} planned targets; "
            f"{coverage.get('unplanned_source_count', 0)} unplanned sources"
        )
        lines.append(
            f"  - explicit receipts: {coverage.get('valid_explicit_sampling_receipt_count', 0)} valid; "
            f"{coverage.get('invalid_explicit_sampling_receipt_count', 0)} invalid; "
            f"{coverage.get('sampled_target_missing_explicit_receipt_count', 0)} matched targets missing receipts"
        )
        iteration_gate = record.get("review_gate", {})
        lines.append(f"  - review gate: {iteration_gate.get('status', 'unknown')}")
    learned = learning_dossier.get("learned_findings", {})
    coverage_overview = learning_dossier.get("coverage_overview", {})
    evidence_gaps = learning_dossier.get("evidence_gaps", {})
    blog_health = learning_dossier.get("blog_health", {})
    lane_counts = coverage_overview.get("source_lane_citation_counts", {})
    lane_rows = [f"{lane}: {count}" for lane, count in lane_counts.items()]
    top_source_rows = [
        f"{source.get('title') or source.get('key')} - {source.get('citation_count', 0)} citation(s)"
        for source in coverage_overview.get("top_sources", [])[:5]
    ]
    thin_lane_rows = [
        f"{lane.get('source_lane')}: {lane.get('sampled_target_count', 0)}/"
        f"{lane.get('planned_target_count', 0)} sampled; "
        f"{lane.get('source_citation_count', 0)} citation(s)"
        for lane in evidence_gaps.get("thin_source_lanes", [])[:10]
    ]
    uncited_lane_rows = [
        f"{lane.get('source_lane')}: {lane.get('sampled_target_count', 0)}/"
        f"{lane.get('planned_target_count', 0)} sampled"
        for lane in evidence_gaps.get("uncited_source_lanes", [])[:10]
    ]
    failed_blog_rows = []
    for blog in blog_health.get("failed_blogs", [])[:10]:
        failed_blog_rows.append(
            f"i{blog.get('iteration')}: {blog.get('persona_id')} missing "
            f"{', '.join(blog.get('missing_sections', [])) or 'required sections'}"
        )
    validation_rows = promotion_review_markdown_rows(learning_dossier.get("validation_backlog", []))
    action_rows = [
        f"[{action.get('priority')}] {action.get('kind')}: {action.get('action')} - {action.get('reason')}"
        for action in learning_dossier.get("next_research_actions", [])[:15]
    ]
    lines.extend([
        "",
        "## Learning Dossier",
        f"- review-only: {str(learning_dossier.get('review_only', True)).lower()}",
        f"- learned findings: {learned.get('total_count', 0)}",
        f"- promotion ready: {learned.get('promotion_ready_count', 0)}",
        f"- validation backlog: {learned.get('validation_backlog_count', 0)}",
        f"- rejected: {learned.get('rejected_count', 0)}",
        f"- source citations: {coverage_overview.get('source_citation_count', 0)}",
        f"- unique source observations: {coverage_overview.get('unique_source_observation_count', 0)}",
        f"- sampled coverage: {coverage_overview.get('sampled_target_count', 0)}/{coverage_overview.get('planned_target_count', 0)} planned targets",
        "",
        "### Key Learnings",
        markdown_list(promotion_review_markdown_rows(learned.get("promotion_ready", []))),
        "",
        "### Source Coverage",
        markdown_list(lane_rows),
        "",
        "### Top Sources",
        markdown_list(top_source_rows),
        "",
        "### Evidence Gaps",
        markdown_list(thin_lane_rows + uncited_lane_rows + evidence_gaps.get("review_gate_blocked_reasons", [])),
        "",
        "### Failed Blogs",
        markdown_list(failed_blog_rows),
        "",
        "### Validation Backlog",
        markdown_list(validation_rows),
        "",
        "### Next Research Actions",
        markdown_list(action_rows),
    ])
    lines.extend(["", "## Review Gate"])
    lines.append(f"- status: {review_gate.get('status', 'unknown')}")
    lines.append(
        f"- ready for skill update: {str(review_gate.get('ready_for_skill_update', False)).lower()}"
    )
    lines.append("- blocked reasons:")
    for reason in review_gate.get("blocked_reasons", []) or ["None"]:
        lines.append(f"  - {reason}")
    lines.extend([
        "",
        "## Blog Contract",
        f"- passing: {blog_contract.get('passing_count', 0)}/{blog_contract.get('blog_count', 0)}",
        f"- worker-authored failed: {blog_contract.get('worker_authored_failed_count', 0)}",
        "- failed blogs:",
    ])
    for blog in blog_contract.get("failed_blogs", []) or ["None"]:
        if isinstance(blog, str):
            lines.append(f"  - {blog}")
        else:
            lines.append(
                f"  - i{blog.get('iteration')}: {blog.get('persona_id')} at {blog.get('path')} "
                f"missing {', '.join(blog.get('missing_sections', [])) or 'required sections'}"
            )
    lines.extend([
        "",
        "## Promotion Review",
        "- review-only: true",
        f"- auto modify recommendations: {str(promotion_review.get('auto_modify_recommendations', False)).lower()}",
        f"- auto promote sources: {str(promotion_review.get('auto_promote_sources', False)).lower()}",
        f"- promotion ready: {promotion_review.get('promotion_ready_count', 0)}",
        f"- validation backlog: {promotion_review.get('validation_backlog_count', 0)}",
        f"- rejected: {promotion_review.get('rejected_count', 0)}",
        "",
        "### Promotion Ready",
        markdown_list(promotion_review_markdown_rows(promotion_review.get("promotion_ready", []))),
        "",
        "### Validation Backlog",
        markdown_list(promotion_review_markdown_rows(promotion_review.get("validation_backlog", []))),
        "",
        "### Rejected",
        markdown_list(promotion_review_markdown_rows(promotion_review.get("rejected", []))),
    ])
    lines.extend(["", "## Decision Counts"])
    for decision, count in loop_summary.get("decision_counts", {}).items():
        lines.append(f"- {decision}: {count}")
    lines.extend(["", "## Execution Counts"])
    for state, count in loop_summary.get("execution_counts", {}).items():
        lines.append(f"- {state}: {count}")
    lines.append("")
    (output_dir / "loop.md").write_text("\n".join(lines), encoding="utf-8")


def write_loop_receipts(output_dir: Path, loop_summary: dict[str, Any]) -> None:
    dump_json(loop_summary, output_dir / "loop.json")
    write_loop_markdown(output_dir, loop_summary)


def review_gate_failed(receipt: dict[str, Any]) -> bool:
    return receipt.get("review_gate", {}).get("status") != "pass"


def review_gate_failure_message(receipt: dict[str, Any]) -> str:
    gate = receipt.get("review_gate", {})
    reasons = gate.get("blocked_reasons", []) or ["review gate did not pass"]
    return "review gate failed: " + "; ".join(reasons)


def failure_messages(args: argparse.Namespace, receipt: dict[str, Any]) -> list[str]:
    messages = []
    if args.require_worker_blog_contract and blog_contract_failed(receipt):
        messages.append(blog_contract_failure_message(receipt))
    if args.fail_on_review_gate and review_gate_failed(receipt):
        messages.append(review_gate_failure_message(receipt))
    return messages


def main() -> int:
    args = parse_args()
    try:
        if args.offline_fixture and args.executor_command:
            raise SkillError("--offline-fixture and --executor-command are mutually exclusive")
        if args.ingest_subagent_results and args.executor_command:
            raise SkillError("--ingest-subagent-results and --executor-command are mutually exclusive")
        if args.ingest_subagent_results and args.offline_fixture:
            raise SkillError("--ingest-subagent-results and --offline-fixture are mutually exclusive")
        if args.iterations <= 0:
            raise SkillError("--iterations must be positive")
        if args.executor_workers <= 0:
            raise SkillError("--executor-workers must be positive")
        if (args.iteration_index is None) != (args.iteration_count_total is None):
            raise SkillError("--iteration-index and --iteration-count-total must be provided together")
        if args.iteration_index is not None:
            if args.iteration_index <= 0 or args.iteration_count_total <= 0:
                raise SkillError("--iteration-index and --iteration-count-total must be positive")
            if args.iteration_index > args.iteration_count_total:
                raise SkillError("--iteration-index cannot exceed --iteration-count-total")
            if args.iterations != 1:
                raise SkillError("--iteration-index is only valid for single-wave scaffolds")
        config = load_config(args.config)
        review_requirements = review_gate_requirements(args, config)
        run_id = args.run_id or default_run_id()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.iterations == 1 and not args.until_review_gate:
            summary = run_iteration(
                args,
                config,
                run_id,
                output_dir,
                unique_ordered(args.gap_hint),
                args.iteration_index or 1,
                args.iteration_count_total or 1,
                review_requirements,
            )
            print(f"wrote research loop {run_id} to {output_dir}: {summary['finding_count']} findings")
            messages = failure_messages(args, summary)
            if messages:
                print(f"error: {'; '.join(messages)}", file=sys.stderr)
                return 2
            return 0

        gap_hints = unique_ordered(args.gap_hint)
        records = []
        campaign_waves = []
        stop_reason = "fixed_iterations_complete"
        for iteration in range(1, args.iterations + 1):
            iteration_run_id = f"{run_id}-i{iteration:02d}"
            iteration_dir = output_dir / "iterations" / f"{iteration:02d}"
            summary = run_iteration(
                args,
                config,
                iteration_run_id,
                iteration_dir,
                gap_hints,
                iteration,
                args.iterations,
                review_requirements,
            )
            records.append(loop_iteration_record(output_dir, summary, iteration_dir))
            campaign_waves.append((summary, iteration_dir))
            gap_hints = merge_gap_hints(gap_hints, summary.get("next_gap_hints", []))
            if args.until_review_gate and not review_gate_failed(summary):
                stop_reason = "review_gate_passed"
                break
        else:
            if args.until_review_gate:
                stop_reason = "iteration_cap_exhausted"
        loop_summary = build_loop_summary(
            run_id,
            args.objective,
            output_dir,
            records,
            gap_hints,
            args.iterations,
            stop_reason,
            args.until_review_gate,
        )
        loop_summary["campaign_manifest"] = write_campaign_receipts(
            output_dir,
            build_campaign_manifest(
                run_id,
                args.objective,
                output_dir,
                campaign_waves,
                args.iterations,
                args.until_review_gate,
                stop_reason,
            ),
        )
        write_loop_receipts(output_dir, loop_summary)
        print(
            f"wrote research loop {run_id} to {output_dir}: "
            f"{loop_summary['total_finding_count']} findings across {loop_summary['iteration_count']} iterations"
        )
        messages = failure_messages(args, loop_summary)
        if messages:
            print(f"error: {'; '.join(messages)}", file=sys.stderr)
            return 2
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
