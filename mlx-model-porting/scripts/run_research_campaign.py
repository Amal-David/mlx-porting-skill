#!/usr/bin/env python3
"""Run review-only research campaign waves from a campaign.json receipt."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured, safe_relpath, slugify

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a review-only MLX research campaign")
    parser.add_argument("--campaign", required=True, help="Path to a campaign.json receipt")
    parser.add_argument(
        "--agent-command",
        help="Explicit local command to run once per campaign agent",
    )
    parser.add_argument("--workers", type=int, default=1, help="Maximum agents to run concurrently per wave")
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--skip-ingest", action="store_true", help="Run agents but do not invoke ingest commands")
    parser.add_argument("--dry-run", action="store_true", help="Write a run plan without executing agents or ingest")
    parser.add_argument("--output", help="Receipt path; defaults to <campaign-dir>/campaign-run.json")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stringify_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def campaign_path(root: Path, value: str | None, label: str) -> Path:
    if not value:
        raise SkillError(f"campaign agent is missing {label}")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SkillError(f"{label} escapes campaign root: {value}") from exc
    return candidate


def parse_command(value: str | None, flag: str) -> list[str]:
    if value is None:
        return []
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise SkillError(f"could not parse {flag}: {exc}") from exc
    if not parts:
        raise SkillError(f"{flag} must not be empty")
    return parts


def load_campaign(path: Path) -> dict[str, Any]:
    data = load_structured(path)
    if not isinstance(data, dict):
        raise SkillError("campaign receipt must be an object")
    if data.get("review_only") is not True:
        raise SkillError("campaign receipt must be review_only=true")
    waves = data.get("waves")
    if not isinstance(waves, list) or not waves:
        raise SkillError("campaign receipt must contain at least one wave")
    return data


def run_log_path(log_dir: Path, wave: dict[str, Any], suffix: str, extension: str) -> Path:
    iteration = int(wave.get("iteration") or 1)
    return log_dir / f"wave-{iteration:02d}-{suffix}.{extension}"


def agent_log_path(log_dir: Path, wave: dict[str, Any], agent: dict[str, Any], extension: str) -> Path:
    suffix = slugify(str(agent.get("persona_id") or f"agent-{agent.get('assignment_index', 0)}"))
    return run_log_path(log_dir, wave, suffix, extension)


def execute_agent(
    root: Path,
    campaign_receipt: Path,
    campaign: dict[str, Any],
    wave: dict[str, Any],
    agent: dict[str, Any],
    command: list[str],
    command_label: str,
    timeout: float,
    workers_requested: int,
    workers_actual: int,
    log_dir: Path,
) -> tuple[dict[str, Any], str | None]:
    persona_id = str(agent.get("persona_id") or "")
    if not persona_id:
        raise SkillError("campaign agent is missing persona_id")
    assignment_path = campaign_path(root, agent.get("assignment_path"), "assignment_path")
    prompt_path = campaign_path(root, agent.get("prompt_path"), "prompt_path")
    result_path = campaign_path(root, agent.get("result_path"), "result_path")
    blog_path = campaign_path(root, agent.get("blog_path"), "blog_path")
    stdout_path = agent_log_path(log_dir, wave, agent, "stdout.txt")
    stderr_path = agent_log_path(log_dir, wave, agent, "stderr.txt")
    log_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    exit_code: int | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    failure_reason = None
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env={
                **os.environ,
                "MLX_RESEARCH_PERSONA_ID": persona_id,
                "MLX_RESEARCH_ASSIGNMENT_PATH": str(assignment_path),
                "MLX_RESEARCH_PROMPT_PATH": str(prompt_path),
                "MLX_RESEARCH_RESULT_PATH": str(result_path),
                "MLX_RESEARCH_BLOG_PATH": str(blog_path),
                "MLX_RESEARCH_RUN_ID": str(wave.get("run_id") or campaign.get("run_id") or ""),
                "MLX_RESEARCH_OUTPUT_DIR": str(campaign_path(root, wave.get("output_dir"), "output_dir")),
                "MLX_RESEARCH_CAMPAIGN_PATH": str(campaign_receipt),
                "MLX_RESEARCH_CAMPAIGN_WAVE": str(wave.get("iteration") or 1),
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
        failure_reason = f"agent command timed out after {timeout:g}s"
    except OSError as exc:
        failure_reason = str(exc)

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    if failure_reason is None and exit_code != 0:
        failure_reason = f"agent command exited with {exit_code}"
    result_exists = result_path.exists()
    if failure_reason is None and not result_exists:
        failure_reason = f"agent did not write expected result file {safe_relpath(root, result_path)}"
    state = "agent_completed" if failure_reason is None else "agent_failed"
    record: dict[str, Any] = {
        "persona_id": persona_id,
        "assignment_index": agent.get("assignment_index"),
        "state": state,
        "agent_command": command_label,
        "workers_requested": workers_requested,
        "workers_actual": workers_actual,
        "assignment_path": safe_relpath(root, assignment_path),
        "prompt_path": safe_relpath(root, prompt_path),
        "result_path": safe_relpath(root, result_path),
        "blog_path": safe_relpath(root, blog_path),
        "stdout_path": safe_relpath(root, stdout_path),
        "stderr_path": safe_relpath(root, stderr_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "result_exists": result_exists,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    if failure_reason:
        record["failure_reason"] = failure_reason
    return record, f"{persona_id}: {failure_reason}" if failure_reason else None


def dry_run_agent_record(root: Path, agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "persona_id": agent.get("persona_id"),
        "assignment_index": agent.get("assignment_index"),
        "state": "dry_run",
        "assignment_path": agent.get("assignment_path"),
        "prompt_path": agent.get("prompt_path"),
        "result_path": agent.get("result_path"),
        "blog_path": agent.get("blog_path"),
        "result_exists": bool(agent.get("result_path") and campaign_path(root, agent.get("result_path"), "result_path").exists()),
    }


def run_agents(
    root: Path,
    campaign_receipt: Path,
    campaign: dict[str, Any],
    wave: dict[str, Any],
    command: list[str],
    command_label: str,
    timeout: float,
    workers: int,
    dry_run: bool,
    log_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    agents = wave.get("agents")
    if not isinstance(agents, list):
        raise SkillError("campaign wave agents must be a list")
    if dry_run:
        return [dry_run_agent_record(root, agent) for agent in agents], []
    actual_workers = min(workers, len(agents)) if agents else 1
    results: list[tuple[int, dict[str, Any], str | None]] = []
    if actual_workers == 1:
        for index, agent in enumerate(agents):
            record, failure = execute_agent(
                root, campaign_receipt, campaign, wave, agent, command, command_label, timeout, workers, actual_workers, log_dir
            )
            results.append((index, record, failure))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as pool:
            future_to_index = {
                pool.submit(
                    execute_agent,
                    root,
                    campaign_receipt,
                    campaign,
                    wave,
                    agent,
                    command,
                    command_label,
                    timeout,
                    workers,
                    actual_workers,
                    log_dir,
                ): index
                for index, agent in enumerate(agents)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                record, failure = future.result()
                results.append((index, record, failure))
    results.sort(key=lambda row: row[0])
    return [record for _, record, _ in results], [failure for _, _, failure in results if failure]


def run_ingest(
    root: Path,
    wave: dict[str, Any],
    timeout: float,
    skip_ingest: bool,
    dry_run: bool,
    log_dir: Path,
) -> tuple[dict[str, Any], str | None]:
    command = wave.get("ingest", {}).get("command_args")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise SkillError("campaign wave ingest.command_args must be a string list")
    stdout_path = run_log_path(log_dir, wave, "ingest", "stdout.txt")
    stderr_path = run_log_path(log_dir, wave, "ingest", "stderr.txt")
    if dry_run:
        return {"state": "dry_run", "command_args": command}, None
    if skip_ingest:
        return {"state": "skipped", "command_args": command}, None
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    exit_code: int | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    failure_reason = None
    try:
        completed = subprocess.run(
            command,
            cwd=SKILL_ROOT,
            env={**os.environ, "MLX_RESEARCH_REVIEW_ONLY": "1"},
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
        failure_reason = f"ingest command timed out after {timeout:g}s"
    except OSError as exc:
        failure_reason = str(exc)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    if failure_reason is None and exit_code != 0:
        failure_reason = f"ingest command exited with {exit_code}"
    state = "ingest_completed" if failure_reason is None else "ingest_failed"
    record: dict[str, Any] = {
        "state": state,
        "command_args": command,
        "stdout_path": safe_relpath(root, stdout_path),
        "stderr_path": safe_relpath(root, stderr_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    if failure_reason:
        record["failure_reason"] = failure_reason
    return record, failure_reason


def run_wave(
    root: Path,
    campaign_receipt: Path,
    campaign: dict[str, Any],
    wave: dict[str, Any],
    command: list[str],
    command_label: str,
    timeout: float,
    workers: int,
    skip_ingest: bool,
    dry_run: bool,
    log_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    started_at = utc_now()
    agent_records, failures = run_agents(
        root, campaign_receipt, campaign, wave, command, command_label, timeout, workers, dry_run, log_dir
    )
    ingest_record = {"state": "not_started"}
    if not failures:
        ingest_record, ingest_failure = run_ingest(root, wave, timeout, skip_ingest, dry_run, log_dir)
        if ingest_failure:
            failures.append(f"wave {wave.get('iteration', 1)} ingest: {ingest_failure}")
    state = "wave_completed" if not failures else "wave_failed"
    return {
        "iteration": wave.get("iteration"),
        "run_id": wave.get("run_id"),
        "state": "dry_run" if dry_run else state,
        "output_dir": wave.get("output_dir"),
        "agent_count": len(agent_records),
        "agents": agent_records,
        "ingest": ingest_record,
        "started_at": started_at,
        "finished_at": utc_now(),
    }, failures


def build_receipt(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    if args.workers <= 0:
        raise SkillError("--workers must be positive")
    if args.execution_timeout <= 0:
        raise SkillError("--execution-timeout must be positive")
    if not args.dry_run and not args.agent_command:
        raise SkillError("--agent-command is required unless --dry-run is set")
    command = parse_command(args.agent_command, "--agent-command")
    campaign_receipt = Path(args.campaign).resolve()
    root = campaign_receipt.parent
    campaign = load_campaign(campaign_receipt)
    log_dir = root / "campaign-run-logs"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "campaign_path": safe_relpath(root, campaign_receipt),
        "campaign_run_id": campaign.get("run_id"),
        "objective": campaign.get("objective"),
        "generated_at": utc_now(),
        "review_only": True,
        "dry_run": bool(args.dry_run),
        "skip_ingest": bool(args.skip_ingest),
        "agent_command": args.agent_command,
        "workers_requested": args.workers,
        "execution_timeout": args.execution_timeout,
        "state": "running",
        "waves": [],
        "instructions": [
            "This receipt records review-only campaign execution.",
            "Promote findings only through explicit source, validation, tests, and manifest updates.",
            "Do not execute remote model code while investigating candidates.",
        ],
    }
    all_failures: list[str] = []
    for wave in campaign["waves"]:
        wave_record, failures = run_wave(
            root,
            campaign_receipt,
            campaign,
            wave,
            command,
            args.agent_command or "",
            args.execution_timeout,
            args.workers,
            args.skip_ingest,
            args.dry_run,
            log_dir,
        )
        receipt["waves"].append(wave_record)
        all_failures.extend(failures)
        if failures:
            break
    receipt["state"] = "dry_run" if args.dry_run else ("failed" if all_failures else "completed")
    receipt["failure_count"] = len(all_failures)
    receipt["failures"] = all_failures
    return receipt, all_failures


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        f"# Research Campaign Run {receipt.get('campaign_run_id')}",
        "",
        f"Objective: {receipt.get('objective')}",
        f"State: {receipt['state']}",
        f"Review only: {receipt['review_only']}",
        f"Dry run: {receipt['dry_run']}",
        f"Skip ingest: {receipt['skip_ingest']}",
        f"Failures: {receipt['failure_count']}",
        "",
        "## Waves",
    ]
    for wave in receipt.get("waves", []):
        lines.append(f"- Wave {wave.get('iteration')}: {wave.get('state')} ({wave.get('agent_count')} agents)")
        lines.append(f"  - ingest: {wave.get('ingest', {}).get('state', 'unknown')}")
        for agent in wave.get("agents", []):
            lines.append(
                f"  - {agent.get('persona_id')}: {agent.get('state')} -> {agent.get('result_path')}"
            )
    if receipt.get("failures"):
        lines.extend(["", "## Failures"])
        for failure in receipt["failures"]:
            lines.append(f"- {failure}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_receipts(output_path: Path, receipt: dict[str, Any]) -> None:
    dump_json(receipt, output_path)
    markdown_path = output_path.with_suffix(".md")
    write_markdown(markdown_path, receipt)


def main() -> int:
    args = parse_args()
    try:
        output_path = Path(args.output).resolve() if args.output else Path(args.campaign).resolve().parent / "campaign-run.json"
        receipt, failures = build_receipt(args)
        write_receipts(output_path, receipt)
        if failures:
            print(f"error: campaign run failed: {'; '.join(failures)}", file=sys.stderr)
            return 2
        print(f"wrote campaign run receipt to {output_path}: {receipt['state']}")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
