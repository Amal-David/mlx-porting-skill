#!/usr/bin/env python3
"""Run review-only research campaign waves from a campaign.json receipt."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import (
    SkillError,
    atomic_write_text,
    dump_json,
    load_structured,
    redact_secret_text,
    run_process_capture,
    safe_relpath,
    slugify,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
MAX_CAMPAIGN_WAVES = 32
MAX_AGENTS_PER_WAVE = 64
MAX_CAMPAIGN_AGENTS = 512
CAMPAIGN_AGENT_KEYS = {
    "assignment_index",
    "persona_id",
    "title",
    "source_lanes",
    "assignment_path",
    "prompt_path",
    "result_path",
    "blog_path",
    "blog_source",
    "execution_kind",
    "execution_state",
    "review_only",
}


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
    parser.add_argument(
        "--follow-next-wave-scaffold",
        action="store_true",
        help="After successful ingest, run next_wave_scaffold commands and continue generated waves",
    )
    parser.add_argument(
        "--max-followed-waves",
        type=int,
        default=8,
        help="Maximum generated follow-up waves to run when following next_wave_scaffold",
    )
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
    if not isinstance(value, str) or Path(value).is_absolute():
        raise SkillError(f"{label} must be a relative path under the campaign root")
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"campaign root contains a symlink cycle or invalid link: {root}") from exc
    candidate = Path(os.path.abspath(root / value))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SkillError(f"{label} escapes campaign root: {value}") from exc
    cursor = root
    for part in candidate.relative_to(root).parts:
        cursor /= part
        try:
            cursor.lstat()
        except FileNotFoundError:
            break
        except (OSError, RuntimeError) as exc:
            raise SkillError(f"{label} contains a symlink cycle or invalid parent: {value}") from exc
        if cursor.is_symlink():
            try:
                resolved_link = cursor.resolve(strict=True)
                resolved_link.relative_to(root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise SkillError(f"{label} contains a symlink cycle or invalid link: {value}") from exc
    try:
        candidate.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SkillError(f"{label} resolves outside campaign root: {value}") from exc
    return candidate


def result_file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SkillError(f"research result must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def quarantine_preexisting_result(root: Path, result_path: Path) -> str | None:
    """Move an old result aside so a no-op process cannot inherit success."""
    if not result_path.exists() and not result_path.is_symlink():
        return None
    root = root.resolve()
    lexical = result_path.parent.resolve() / result_path.name
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise SkillError(f"result path escapes campaign root: {result_path}") from exc
    quarantine_root = root / ".stale-results"
    destination = quarantine_root / relative.parent / f"{relative.name}.{time.time_ns()}.stale"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(lexical, destination)
    return safe_relpath(root, destination)


def validate_campaign_schema(campaign: dict[str, Any], root: Path) -> None:
    """Validate bounded campaign topology and all agent output destinations."""
    if campaign.get("schema_version") != 1:
        raise SkillError("campaign schema_version must be 1")
    for field in ("run_id", "objective"):
        if not isinstance(campaign.get(field), str) or not campaign[field].strip():
            raise SkillError(f"campaign {field} must be a non-empty string")
    if campaign.get("review_only") is not True:
        raise SkillError("campaign receipt must be review_only=true")
    if campaign.get("campaign_root") != ".":
        raise SkillError("campaign_root must be '.'")
    iteration_cap = campaign.get("iteration_cap")
    if (
        isinstance(iteration_cap, bool)
        or not isinstance(iteration_cap, int)
        or not 1 <= iteration_cap <= MAX_CAMPAIGN_WAVES
    ):
        raise SkillError(f"campaign iteration_cap must be between 1 and {MAX_CAMPAIGN_WAVES}")
    if not isinstance(campaign.get("until_review_gate"), bool):
        raise SkillError("campaign until_review_gate must be boolean")
    waves = campaign.get("waves")
    if not isinstance(waves, list) or not waves:
        raise SkillError("campaign receipt must contain at least one wave")
    if len(waves) > MAX_CAMPAIGN_WAVES:
        raise SkillError(f"campaign wave limit exceeded: {len(waves)} > {MAX_CAMPAIGN_WAVES}")
    if campaign.get("wave_count") != len(waves):
        raise SkillError("campaign wave_count does not match waves")

    wave_run_ids: set[str] = set()
    all_result_paths: set[str] = set()
    total_agents = 0
    for wave_index, wave in enumerate(waves, start=1):
        if not isinstance(wave, dict):
            raise SkillError(f"campaign wave {wave_index} must be an object")
        run_id = wave.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise SkillError(f"campaign wave {wave_index} is missing run_id")
        if run_id in wave_run_ids:
            raise SkillError(f"campaign contains duplicate wave run_id: {run_id}")
        wave_run_ids.add(run_id)
        iteration = wave.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
            raise SkillError(f"campaign wave {run_id} iteration must be a positive integer")
        output_root = campaign_path(root, wave.get("output_dir"), "wave output_dir")
        agents = wave.get("agents")
        if not isinstance(agents, list) or not agents:
            raise SkillError(f"campaign wave {run_id} agents must be a non-empty list")
        if len(agents) > MAX_AGENTS_PER_WAVE:
            raise SkillError(
                f"campaign wave {run_id} agent limit exceeded: {len(agents)} > {MAX_AGENTS_PER_WAVE}"
            )
        if wave.get("agent_count") != len(agents):
            raise SkillError(f"campaign wave {run_id} agent_count does not match agents")
        total_agents += len(agents)
        if total_agents > MAX_CAMPAIGN_AGENTS:
            raise SkillError(f"campaign total agent limit exceeded: {total_agents} > {MAX_CAMPAIGN_AGENTS}")

        persona_ids: set[str] = set()
        persona_log_ids: set[str] = set()
        assignment_indices: set[int] = set()
        for agent_index, agent in enumerate(agents):
            if not isinstance(agent, dict):
                raise SkillError(f"campaign wave {run_id} agent {agent_index} must be an object")
            keys = set(agent)
            missing = sorted(CAMPAIGN_AGENT_KEYS - keys)
            unknown = sorted(keys - CAMPAIGN_AGENT_KEYS)
            if missing:
                raise SkillError(
                    f"campaign wave {run_id} agent {agent_index} missing required keys: {', '.join(missing)}"
                )
            if unknown:
                raise SkillError(
                    f"campaign wave {run_id} agent {agent_index} has unsupported keys: {', '.join(unknown)}"
                )
            persona_id = agent.get("persona_id")
            if not isinstance(persona_id, str) or not persona_id.strip():
                raise SkillError(f"campaign wave {run_id} agent {agent_index} has invalid persona_id")
            if persona_id in persona_ids:
                raise SkillError(f"campaign wave {run_id} has duplicate persona_id: {persona_id}")
            persona_ids.add(persona_id)
            persona_log_id = slugify(persona_id)
            if persona_log_id in persona_log_ids:
                raise SkillError(
                    f"campaign wave {run_id} has duplicate persona log identity: {persona_log_id}"
                )
            persona_log_ids.add(persona_log_id)
            assignment_index = agent.get("assignment_index")
            if isinstance(assignment_index, bool) or not isinstance(assignment_index, int) or assignment_index < 0:
                raise SkillError(f"campaign wave {run_id} agent {persona_id} has invalid assignment_index")
            if assignment_index in assignment_indices:
                raise SkillError(f"campaign wave {run_id} has duplicate assignment_index: {assignment_index}")
            assignment_indices.add(assignment_index)
            if agent.get("review_only") is not True:
                raise SkillError(f"campaign wave {run_id} agent {persona_id} must be review_only=true")
            if not isinstance(agent.get("title"), str) or not agent["title"].strip():
                raise SkillError(f"campaign wave {run_id} agent {persona_id} has invalid title")
            lanes = agent.get("source_lanes")
            if (
                not isinstance(lanes, list)
                or len(lanes) > 32
                or not all(isinstance(lane, str) and lane for lane in lanes)
                or len(lanes) != len(set(lanes))
            ):
                raise SkillError(f"campaign wave {run_id} agent {persona_id} has invalid source_lanes")
            for field in ("assignment_path", "prompt_path", "result_path", "blog_path"):
                path = campaign_path(root, agent.get(field), field)
                try:
                    path.resolve(strict=False).relative_to(output_root.resolve(strict=False))
                except ValueError as exc:
                    raise SkillError(
                        f"campaign wave {run_id} agent {persona_id} {field} is outside wave output_dir"
                    ) from exc
                if field == "result_path":
                    normalized = safe_relpath(root, path)
                    if normalized.startswith(".stale-results/"):
                        raise SkillError("campaign result_path cannot use the stale-results quarantine")
                    if normalized in all_result_paths:
                        raise SkillError(f"campaign contains duplicate result_path: {normalized}")
                    all_result_paths.add(normalized)
            for field in ("blog_source",):
                if not isinstance(agent.get(field), str):
                    raise SkillError(f"campaign wave {run_id} agent {persona_id} has invalid {field}")
            for field in ("execution_kind", "execution_state"):
                if agent.get(field) is not None and not isinstance(agent.get(field), str):
                    raise SkillError(f"campaign wave {run_id} agent {persona_id} has invalid {field}")


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


def command_arg_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    if index + 1 >= len(command):
        raise SkillError(f"{flag} requires a value")
    return command[index + 1]


RESEARCH_OPERATION_TYPES = {"research_loop_ingest", "research_loop_scaffold"}
RESEARCH_OPERATION_ARGUMENTS = {
    "run_id",
    "objective",
    "agent_count",
    "assignment_mode",
    "gap_hints",
    "iteration_index",
    "iteration_count_total",
    "min_sampled_targets",
    "min_non_github_lanes",
    "required_source_lanes",
    "require_explicit_sampling_receipts",
    "fail_on_review_gate",
    "require_worker_blog_contract",
    "ingest_subagent_results",
    "output_dir",
    "config",
}


def _operation_string(arguments: dict[str, Any], key: str, *, required: bool = False) -> str | None:
    value = arguments.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise SkillError(f"typed research operation argument {key} must be a non-empty string")
    return value


def _operation_integer(arguments: dict[str, Any], key: str, *, minimum: int) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SkillError(f"typed research operation argument {key} must be an integer >= {minimum}")
    return value


def _operation_strings(arguments: dict[str, Any], key: str) -> list[str]:
    value = arguments.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SkillError(f"typed research operation argument {key} must be a string list")
    return value


def _operation_bool(arguments: dict[str, Any], key: str) -> bool:
    value = arguments.get(key, False)
    if not isinstance(value, bool):
        raise SkillError(f"typed research operation argument {key} must be a boolean")
    return value


def typed_research_loop_command(
    operation: Any,
    allowed_root: Path,
    *,
    expected_type: str,
) -> tuple[list[str], Path]:
    """Build the only executable command shape accepted from campaign receipts."""
    if not isinstance(operation, dict):
        raise SkillError("campaign receipt must select a typed operation")
    operation_type = operation.get("type")
    if operation_type not in RESEARCH_OPERATION_TYPES or operation_type != expected_type:
        raise SkillError(f"campaign typed operation must be {expected_type}")
    arguments = operation.get("arguments")
    if not isinstance(arguments, dict):
        raise SkillError("campaign typed operation arguments must be an object")
    unknown = sorted(set(arguments) - RESEARCH_OPERATION_ARGUMENTS)
    if unknown:
        raise SkillError(f"campaign typed operation has unsupported arguments: {', '.join(unknown)}")

    run_id = _operation_string(arguments, "run_id", required=True)
    objective = _operation_string(arguments, "objective", required=True)
    agent_count = _operation_integer(arguments, "agent_count", minimum=1)
    assignment_mode = _operation_string(arguments, "assignment_mode") or "config-order"
    if assignment_mode not in {"auto", "config-order", "dynamic"}:
        raise SkillError("typed research operation assignment_mode is invalid")
    output_value = _operation_string(arguments, "output_dir", required=True)
    assert output_value is not None
    output_dir = Path(output_value)
    if output_dir.is_absolute():
        output_dir = output_dir.resolve()
    else:
        base = allowed_root.resolve()
        try:
            allowed_root.resolve().relative_to(SKILL_ROOT.resolve())
            base = SKILL_ROOT.resolve()
        except ValueError:
            pass
        output_dir = (base / output_dir).resolve()
    try:
        output_dir.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise SkillError(f"typed research operation output escapes allowed root: {output_dir}") from exc

    command = [
        sys.executable,
        str(SCRIPT_DIR / "research_loop.py"),
        "--run-id",
        str(run_id),
        "--objective",
        str(objective),
    ]
    if agent_count is not None:
        command.extend(["--agent-count", str(agent_count)])
    command.extend(["--assignment-mode", assignment_mode])
    config = _operation_string(arguments, "config")
    if config:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = (SKILL_ROOT / config_path).resolve()
        else:
            config_path = config_path.resolve()
        try:
            config_path.relative_to(SKILL_ROOT.resolve())
        except ValueError as exc:
            raise SkillError("typed research operation config escapes skill root") from exc
        command.extend(["--config", str(config_path)])
    for hint in _operation_strings(arguments, "gap_hints"):
        command.extend(["--gap-hint", hint])
    for key, flag, minimum in (
        ("iteration_index", "--iteration-index", 1),
        ("iteration_count_total", "--iteration-count-total", 1),
        ("min_sampled_targets", "--min-sampled-targets", 0),
        ("min_non_github_lanes", "--min-non-github-lanes", 0),
    ):
        value = _operation_integer(arguments, key, minimum=minimum)
        if value is not None:
            command.extend([flag, str(value)])
    for lane in _operation_strings(arguments, "required_source_lanes"):
        command.extend(["--require-source-lane", lane])
    for key, flag in (
        ("require_explicit_sampling_receipts", "--require-explicit-sampling-receipts"),
        ("fail_on_review_gate", "--fail-on-review-gate"),
        ("require_worker_blog_contract", "--require-worker-blog-contract"),
        ("ingest_subagent_results", "--ingest-subagent-results"),
    ):
        if _operation_bool(arguments, key):
            command.append(flag)
    if expected_type == "research_loop_ingest" and "--ingest-subagent-results" not in command:
        raise SkillError("research_loop_ingest operation must ingest subagent results")
    if expected_type == "research_loop_scaffold" and "--ingest-subagent-results" in command:
        raise SkillError("research_loop_scaffold operation cannot ingest subagent results")
    command.extend(["--output-dir", str(output_dir)])
    return command, output_dir


def ingest_command_for_wave(root: Path, wave: dict[str, Any]) -> tuple[list[str], Path]:
    ingest = wave.get("ingest")
    if not isinstance(ingest, dict):
        raise SkillError("campaign wave is missing ingest typed operation")
    return typed_research_loop_command(
        ingest.get("operation"),
        root,
        expected_type="research_loop_ingest",
    )


def load_campaign(path: Path) -> dict[str, Any]:
    data = load_structured(path)
    if not isinstance(data, dict):
        raise SkillError("campaign receipt must be an object")
    if data.get("review_only") is not True:
        raise SkillError("campaign receipt must be review_only=true")
    try:
        campaign_root = path.resolve(strict=True).parent
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"campaign path contains a symlink cycle or invalid link: {path}") from exc
    validate_campaign_schema(data, campaign_root)
    return data


def validate_scaffold_command(command: Any, allowed_root: Path) -> tuple[list[str], Path]:
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise SkillError("next_wave_scaffold.command_args must be a string list")
    if len(command) < 2:
        raise SkillError("next_wave_scaffold.command_args is too short")
    executable = Path(command[0]).name
    if executable not in {"python", "python3"} and Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise SkillError("next_wave_scaffold must use a local Python executable")
    script = command[1]
    script_path = Path(script)
    if script_path.is_absolute():
        try:
            script_path.resolve().relative_to(SCRIPT_DIR.resolve())
        except ValueError as exc:
            raise SkillError("next_wave_scaffold must run scripts/research_loop.py") from exc
        if script_path.resolve() != (SCRIPT_DIR / "research_loop.py").resolve():
            raise SkillError("next_wave_scaffold must run scripts/research_loop.py")
    elif script.replace("\\", "/") != "scripts/research_loop.py":
        raise SkillError("next_wave_scaffold must run scripts/research_loop.py")
    disallowed = {
        "--executor-command",
        "--ingest-subagent-results",
        "--offline-fixture",
    }
    present = sorted(flag for flag in disallowed if flag in command)
    if present:
        raise SkillError(f"next_wave_scaffold command cannot include {', '.join(present)}")
    output_value = command_arg_value(command, "--output-dir")
    if not output_value:
        raise SkillError("next_wave_scaffold command must include --output-dir")
    output_dir = Path(output_value)
    if not output_dir.is_absolute():
        output_dir = (SKILL_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise SkillError(f"next_wave_scaffold output escapes allowed campaign run root: {output_dir}") from exc
    return command, output_dir


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
    stale_result_quarantined_to = quarantine_preexisting_result(root, result_path)

    started_at = utc_now()
    exit_code: int | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    failure_reason = None
    try:
        completed, timed_out = run_process_capture(
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
            timeout=timeout,
        )
        exit_code = completed.returncode
        stdout = stringify_process_output(completed.stdout)
        stderr = stringify_process_output(completed.stderr)
        if timed_out:
            failure_reason = f"agent command timed out after {timeout:g}s"
    except OSError as exc:
        failure_reason = str(exc)

    atomic_write_text(stdout_path, redact_secret_text(stdout))
    atomic_write_text(stderr_path, redact_secret_text(stderr))
    if failure_reason is None and exit_code != 0:
        failure_reason = f"agent command exited with {exit_code}"
    result_exists = result_path.is_file() and not result_path.is_symlink()
    result_identity = result_file_identity(result_path) if result_exists else None
    if failure_reason is None and not result_exists:
        failure_reason = f"agent did not write expected result file {safe_relpath(root, result_path)}"
    state = "agent_completed" if failure_reason is None else "agent_failed"
    record: dict[str, Any] = {
        "persona_id": persona_id,
        "assignment_index": agent.get("assignment_index"),
        "state": state,
        "agent_command": redact_secret_text(command_label),
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
        "result_identity": result_identity,
        "stale_result_quarantined_to": stale_result_quarantined_to,
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
    command, operation_output = ingest_command_for_wave(root, wave)
    expected_output = campaign_path(root, wave.get("output_dir"), "output_dir")
    if operation_output != expected_output:
        raise SkillError("campaign ingest typed operation output_dir must match its wave output_dir")
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
        completed, timed_out = run_process_capture(
            command,
            cwd=SKILL_ROOT,
            env={**os.environ, "MLX_RESEARCH_REVIEW_ONLY": "1"},
            timeout=timeout,
        )
        exit_code = completed.returncode
        stdout = stringify_process_output(completed.stdout)
        stderr = stringify_process_output(completed.stderr)
        if timed_out:
            failure_reason = f"ingest command timed out after {timeout:g}s"
    except OSError as exc:
        failure_reason = str(exc)
    atomic_write_text(stdout_path, redact_secret_text(stdout))
    atomic_write_text(stderr_path, redact_secret_text(stderr))
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


def refreshed_wave_after_ingest(root: Path, wave: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    output_root = campaign_path(root, wave.get("output_dir"), "output_dir")
    refreshed_campaign_path = output_root / "campaign.json"
    if not refreshed_campaign_path.exists():
        return None, None
    refreshed = load_campaign(refreshed_campaign_path)
    run_id = wave.get("run_id")
    iteration = wave.get("iteration")
    for candidate in refreshed["waves"]:
        if run_id and candidate.get("run_id") == run_id:
            return candidate, refreshed_campaign_path
        if iteration and candidate.get("iteration") == iteration:
            return candidate, refreshed_campaign_path
    return refreshed["waves"][0], refreshed_campaign_path


def run_next_wave_scaffold(
    receipt_root: Path,
    allowed_root: Path,
    current_root: Path,
    wave: dict[str, Any],
    timeout: float,
    dry_run: bool,
    skip_ingest: bool,
    cap_reached: bool,
    log_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    refreshed_wave, refreshed_campaign_path = refreshed_wave_after_ingest(current_root, wave)
    scaffold_source = refreshed_wave or wave
    scaffold = scaffold_source.get("next_wave_scaffold") if isinstance(scaffold_source, dict) else None
    if not scaffold:
        return None, [], None
    command, output_dir = typed_research_loop_command(
        scaffold.get("operation"),
        allowed_root,
        expected_type="research_loop_scaffold",
    )
    record: dict[str, Any] = {
        "state": "not_started",
        "command_args": command,
        "source": "refreshed_campaign" if refreshed_wave else "loaded_campaign",
        "source_campaign_path": (
            safe_relpath(receipt_root, refreshed_campaign_path)
            if refreshed_campaign_path
            else None
        ),
        "output_dir": safe_relpath(receipt_root, output_dir),
    }
    if cap_reached:
        record["state"] = "skipped"
        record["skip_reason"] = "--max-followed-waves reached"
        return record, [], None
    if dry_run:
        record["state"] = "dry_run"
        return record, [], None
    if skip_ingest:
        record["state"] = "skipped"
        record["skip_reason"] = "ingest was skipped; next-wave scaffolds require completed ingest"
        return record, [], None

    stdout_path = run_log_path(log_dir, wave, "next-wave-scaffold", "stdout.txt")
    stderr_path = run_log_path(log_dir, wave, "next-wave-scaffold", "stderr.txt")
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    exit_code: int | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    failure_reason = None
    try:
        completed, timed_out = run_process_capture(
            command,
            cwd=SKILL_ROOT,
            env={**os.environ, "MLX_RESEARCH_REVIEW_ONLY": "1"},
            timeout=timeout,
        )
        exit_code = completed.returncode
        stdout = stringify_process_output(completed.stdout)
        stderr = stringify_process_output(completed.stderr)
        if timed_out:
            failure_reason = f"next-wave scaffold command timed out after {timeout:g}s"
    except OSError as exc:
        failure_reason = str(exc)

    atomic_write_text(stdout_path, redact_secret_text(stdout))
    atomic_write_text(stderr_path, redact_secret_text(stderr))
    if failure_reason is None and exit_code != 0:
        failure_reason = f"next-wave scaffold command exited with {exit_code}"
    generated_campaign_path = output_dir / "campaign.json"
    if failure_reason is None and not generated_campaign_path.exists():
        failure_reason = f"next-wave scaffold did not write {generated_campaign_path}"

    record.update({
        "state": "scaffold_completed" if failure_reason is None else "scaffold_failed",
        "stdout_path": safe_relpath(receipt_root, stdout_path),
        "stderr_path": safe_relpath(receipt_root, stderr_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "started_at": started_at,
        "finished_at": utc_now(),
        "generated_campaign_path": safe_relpath(receipt_root, generated_campaign_path),
        "generated_campaign_exists": generated_campaign_path.exists(),
    })
    if failure_reason:
        record["failure_reason"] = failure_reason
        return record, [], failure_reason
    generated_campaign = load_campaign(generated_campaign_path)
    record["generated_wave_count"] = len(generated_campaign["waves"])
    generated_items = [
        {
            "root": output_dir,
            "campaign_receipt": generated_campaign_path,
            "campaign": generated_campaign,
            "wave": generated_wave,
            "source": "next_wave_scaffold",
        }
        for generated_wave in generated_campaign["waves"]
    ]
    return record, generated_items, None


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
    if args.max_followed_waves <= 0:
        raise SkillError("--max-followed-waves must be positive")
    if args.max_followed_waves > MAX_CAMPAIGN_WAVES:
        raise SkillError(f"--max-followed-waves must be <= {MAX_CAMPAIGN_WAVES}")
    if not args.dry_run and not args.agent_command:
        raise SkillError("--agent-command is required unless --dry-run is set")
    command = parse_command(args.agent_command, "--agent-command")
    campaign_receipt = Path(args.campaign).resolve()
    root = campaign_receipt.parent
    receipt_root = root.parent if args.follow_next_wave_scaffold else root
    allowed_scaffold_root = receipt_root
    campaign = load_campaign(campaign_receipt)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "campaign_path": safe_relpath(receipt_root, campaign_receipt),
        "campaign_run_id": campaign.get("run_id"),
        "objective": campaign.get("objective"),
        "generated_at": utc_now(),
        "review_only": True,
        "dry_run": bool(args.dry_run),
        "skip_ingest": bool(args.skip_ingest),
        "follow_next_wave_scaffold": bool(args.follow_next_wave_scaffold),
        "max_followed_waves": args.max_followed_waves,
        "followed_scaffold_count": 0,
        "agent_command": args.agent_command,
        "workers_requested": args.workers,
        "execution_timeout": args.execution_timeout,
        "state": "running",
        "waves": [],
        "scaffold_followups": [],
        "instructions": [
            "This receipt records review-only campaign execution.",
            "Promote findings only through explicit source, validation, tests, and manifest updates.",
            "Do not execute remote model code while investigating candidates.",
        ],
    }
    all_failures: list[str] = []
    pending = [
        {
            "root": root,
            "campaign_receipt": campaign_receipt,
            "campaign": campaign,
            "wave": wave,
            "source": "initial_campaign",
        }
        for wave in campaign["waves"]
    ]
    while pending:
        item = pending.pop(0)
        wave_root = item["root"]
        wave_campaign_receipt = item["campaign_receipt"]
        wave_campaign = item["campaign"]
        wave = item["wave"]
        log_dir = wave_root / "campaign-run-logs"
        wave_record, failures = run_wave(
            wave_root,
            wave_campaign_receipt,
            wave_campaign,
            wave,
            command,
            args.agent_command or "",
            args.execution_timeout,
            args.workers,
            args.skip_ingest,
            args.dry_run,
            log_dir,
        )
        wave_record["campaign_path"] = safe_relpath(receipt_root, wave_campaign_receipt)
        wave_record["campaign_root"] = safe_relpath(receipt_root, wave_root)
        wave_record["source"] = item["source"]
        receipt["waves"].append(wave_record)
        all_failures.extend(failures)
        if failures:
            break
        if not args.follow_next_wave_scaffold:
            continue
        followup, generated_items, scaffold_failure = run_next_wave_scaffold(
            receipt_root,
            allowed_scaffold_root,
            wave_root,
            wave,
            args.execution_timeout,
            args.dry_run,
            args.skip_ingest,
            receipt["followed_scaffold_count"] >= args.max_followed_waves,
            log_dir,
        )
        if followup:
            followup["wave_iteration"] = wave.get("iteration")
            followup["wave_run_id"] = wave.get("run_id")
            wave_record["next_wave_scaffold"] = followup
            receipt["scaffold_followups"].append(followup)
        if scaffold_failure:
            all_failures.append(f"wave {wave.get('iteration', 1)} scaffold: {scaffold_failure}")
            break
        if generated_items:
            receipt["followed_scaffold_count"] += 1
            pending.extend(generated_items)
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
        f"Follow next-wave scaffolds: {receipt.get('follow_next_wave_scaffold', False)}",
        f"Followed scaffolds: {receipt.get('followed_scaffold_count', 0)}",
        f"Failures: {receipt['failure_count']}",
        "",
        "## Waves",
    ]
    for wave in receipt.get("waves", []):
        lines.append(f"- Wave {wave.get('iteration')}: {wave.get('state')} ({wave.get('agent_count')} agents)")
        lines.append(f"  - campaign: {wave.get('campaign_path')}")
        lines.append(f"  - ingest: {wave.get('ingest', {}).get('state', 'unknown')}")
        if wave.get("next_wave_scaffold"):
            followup = wave["next_wave_scaffold"]
            lines.append(
                f"  - next-wave scaffold: {followup.get('state')} -> {followup.get('generated_campaign_path') or followup.get('output_dir')}"
            )
        for agent in wave.get("agents", []):
            lines.append(
                f"  - {agent.get('persona_id')}: {agent.get('state')} -> {agent.get('result_path')}"
            )
    if receipt.get("scaffold_followups"):
        lines.extend(["", "## Scaffold Followups"])
        for followup in receipt["scaffold_followups"]:
            lines.append(
                f"- {followup.get('wave_run_id') or followup.get('wave_iteration')}: {followup.get('state')} -> {followup.get('generated_campaign_path') or followup.get('output_dir')}"
            )
    if receipt.get("failures"):
        lines.extend(["", "## Failures"])
        for failure in receipt["failures"]:
            lines.append(f"- {failure}")
    lines.append("")
    atomic_write_text(path, "\n".join(lines))


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
