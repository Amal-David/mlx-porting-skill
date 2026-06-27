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
        slug = slugify(persona_id)
        prompt_path = agent_dir / f"{slug}.prompt.md"
        result_path = agent_dir / f"{slug}.result.json"
        stdout_path = agent_dir / f"{slug}.stdout.txt"
        stderr_path = agent_dir / f"{slug}.stderr.txt"
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
                    "MLX_RESEARCH_PROMPT_PATH": str(prompt_path),
                    "MLX_RESEARCH_RESULT_PATH": str(result_path),
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
            "prompt_path": rel_output_path(output_dir, prompt_path),
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
                    persona_id = assignments[index]["persona_id"]
                    results.append((index, None, {
                        "kind": "local-executor",
                        "state": "executor_failed",
                        "executor": command,
                        "assignment_index": index,
                        "executor_workers": workers,
                        "executor_actual_workers": actual_workers,
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


def summarize(
    config: dict[str, Any],
    assignments: list[dict[str, Any]],
    planner_receipt: dict[str, Any],
    fixture: dict[str, Any],
    run_id: str,
    objective: str,
    executions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    findings = flatten_findings(fixture)
    decision_counts = {state["id"]: 0 for state in config["decision_states"]}
    lanes: dict[str, int] = {lane["id"]: 0 for lane in config["source_lanes"]}
    for finding in findings:
        decision_counts[finding["decision"]] += 1
        lanes[finding["source_lane"]] += 1
    non_github_lanes = sorted(lane for lane, count in lanes.items() if count and lane != "repositories")
    returned_personas = {agent.get("persona_id") for agent in fixture.get("agents", [])}
    execution_counts = {"scaffolded_not_run": 0, "fixture_ingested": 0, "executor_completed": 0, "executor_failed": 0}
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
        "assignment_planner": planner_receipt,
        **planned_sampling_summary(assignments, config),
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
        planned_sampling = []
        for lane in assignment.get("sample_plan", []):
            for target in lane.get("targets", []):
                planned_sampling.append(
                    f"{lane['source_lane']}: {target['title']} [{target['kind']}] - {target['locator']}"
                )
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
        f"Iteration: {summary.get('iteration', 1)} of {summary.get('iteration_count', 1)}",
        f"Gap hints used: {', '.join(summary.get('gap_hints', [])) or 'none'}",
        f"Next gap hints: {', '.join(summary.get('next_gap_hints', [])) or 'none'}",
        f"Findings: {summary['finding_count']}",
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
    else:
        fixture = load_fixture(args.offline_fixture)
    validate_findings(config, fixture)
    summary = summarize(config, assignments, planner_receipt, fixture, run_id, args.objective, executions)
    summary["iteration"] = iteration
    summary["iteration_count"] = iteration_count
    summary["gap_hints"] = gap_hints
    summary["next_gap_hints"] = derive_next_gap_hints(summary, gap_hints)
    if args.offline_fixture:
        fixture_path = Path(args.offline_fixture)
        summary["offline_fixture"] = safe_relpath(Path.cwd(), fixture_path) if fixture_path.exists() else args.offline_fixture
    if args.executor_command:
        summary["executor_command"] = args.executor_command
        summary["executor_workers"] = args.executor_workers
        summary["executor_actual_workers"] = fixture.get("executor_actual_workers", args.executor_workers)
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
    write_blogs(output_dir, assignments, fixture)
    write_markdown_summary(output_dir, summary)
    return summary


def loop_iteration_record(root: Path, summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "iteration": summary["iteration"],
        "run_id": summary["run_id"],
        "output_dir": safe_relpath(root, output_dir),
        "gap_hints": summary.get("gap_hints", []),
        "next_gap_hints": summary.get("next_gap_hints", []),
        "assignment_mode": summary.get("assignment_planner", {}).get("mode"),
        "executor_workers": summary.get("executor_workers"),
        "executor_actual_workers": summary.get("executor_actual_workers"),
        "selected_personas": [
            row.get("persona_id") for row in summary.get("assignment_planner", {}).get("selected_personas", [])
        ],
        "finding_count": summary.get("finding_count", 0),
        "decision_counts": summary.get("decision_counts", {}),
        "non_github_lanes_covered": summary.get("non_github_lanes_covered", []),
        "execution_counts": summary.get("execution_counts", {}),
    }


def build_loop_summary(
    run_id: str,
    objective: str,
    output_dir: Path,
    records: list[dict[str, Any]],
    final_gap_hints: list[str],
) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    total_findings = 0
    for record in records:
        total_findings += int(record.get("finding_count", 0))
        decision_counts.update(record.get("decision_counts", {}))
        execution_counts.update(record.get("execution_counts", {}))
    return {
        "schema_version": 1,
        "run_id": run_id,
        "objective": objective,
        "generated_at": utc_now(),
        "review_only": True,
        "iteration_count": len(records),
        "total_finding_count": total_findings,
        "decision_counts": dict(sorted(decision_counts.items())),
        "execution_counts": dict(sorted(execution_counts.items())),
        "iterations": records,
        "final_gap_hints": final_gap_hints,
        "instructions": [
            "This loop receipt summarizes review-only research iterations.",
            "Promote findings only through explicit source, validation, tests, and manifest updates.",
            "Do not execute remote model code while investigating candidates.",
        ],
    }


def write_loop_markdown(output_dir: Path, loop_summary: dict[str, Any]) -> None:
    lines = [
        f"# Research Loop {loop_summary['run_id']}",
        "",
        f"Objective: {loop_summary['objective']}",
        f"Review only: {loop_summary['review_only']}",
        f"Iterations: {loop_summary['iteration_count']}",
        f"Total findings: {loop_summary['total_finding_count']}",
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


def main() -> int:
    args = parse_args()
    try:
        if args.offline_fixture and args.executor_command:
            raise SkillError("--offline-fixture and --executor-command are mutually exclusive")
        if args.iterations <= 0:
            raise SkillError("--iterations must be positive")
        if args.executor_workers <= 0:
            raise SkillError("--executor-workers must be positive")
        config = load_config(args.config)
        run_id = args.run_id or default_run_id()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.iterations == 1:
            summary = run_iteration(args, config, run_id, output_dir, unique_ordered(args.gap_hint), 1, 1)
            print(f"wrote research loop {run_id} to {output_dir}: {summary['finding_count']} findings")
            return 0

        gap_hints = unique_ordered(args.gap_hint)
        records = []
        for iteration in range(1, args.iterations + 1):
            iteration_run_id = f"{run_id}-i{iteration:02d}"
            iteration_dir = output_dir / "iterations" / f"{iteration:02d}"
            summary = run_iteration(args, config, iteration_run_id, iteration_dir, gap_hints, iteration, args.iterations)
            records.append(loop_iteration_record(output_dir, summary, iteration_dir))
            gap_hints = merge_gap_hints(gap_hints, summary.get("next_gap_hints", []))
        loop_summary = build_loop_summary(run_id, args.objective, output_dir, records, gap_hints)
        write_loop_receipts(output_dir, loop_summary)
        print(
            f"wrote research loop {run_id} to {output_dir}: "
            f"{loop_summary['total_finding_count']} findings across {args.iterations} iterations"
            )
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
