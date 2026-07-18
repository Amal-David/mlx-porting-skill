#!/usr/bin/env python3
"""Nightly review-only MLX knowledge curator orchestration."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import (
    SkillError,
    dump_json,
    load_structured,
    redact_secret_text,
    redact_secrets,
    run_process_capture,
    slugify,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SKILL_ROOT.parent
DEFAULT_COMMAND_TIMEOUT_SECONDS = 1800.0
RECEIPT_OUTPUT_BYTES = 4000
ERROR_OUTPUT_BYTES = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly MLX knowledge curator")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=str(SKILL_ROOT / "research-runs"))
    parser.add_argument("--agent-count", type=int, default=6)
    parser.add_argument("--skip-live-collectors", action="store_true")
    parser.add_argument("--offline-update-fixture", help="Fixture for update_sources.py")
    parser.add_argument("--offline-contributor-fixture", help="Fixture for collect_contributors.py")
    parser.add_argument("--update-output", default=str(SKILL_ROOT / "assets" / "update-candidates.json"))
    parser.add_argument("--contributor-output", default=str(SKILL_ROOT / "assets" / "contributor-refresh.json"))
    parser.add_argument("--graph-output", default=str(SKILL_ROOT / "assets" / "knowledge_graph.json"))
    parser.add_argument("--previous-graph", default=str(SKILL_ROOT / "assets" / "knowledge_graph.json"))
    parser.add_argument("--research-backlog", default=str(SKILL_ROOT / "assets" / "research_backlog.json"))
    parser.add_argument("--no-research-loop", action="store_true")
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help="Per-command timeout in seconds (default: 1800)",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-nightly-knowledge-curator")


def receipt_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return redact_secret_text(str(path))
    try:
        return redact_secret_text(str(path.resolve().relative_to(REPO_ROOT)))
    except ValueError:
        return redact_secret_text(f"<external>/{path.name}")


def receipt_command(command: list[str]) -> list[str]:
    portable: list[str] = []
    for arg in command:
        if arg == sys.executable:
            portable.append("python3")
        elif Path(arg).is_absolute():
            portable.append(receipt_path(arg))
        else:
            portable.append(arg)
    redacted = redact_secrets(portable)
    return [str(item) for item in redacted]


def receipt_text(value: str) -> str:
    return redact_secret_text(value.replace(str(REPO_ROOT), "."))


def bounded_receipt_output(value: str, limit: int = RECEIPT_OUTPUT_BYTES) -> str:
    """Redact a bounded capture before retaining its byte-bounded tail."""
    safe = receipt_text(value)
    payload = safe.encode("utf-8", errors="replace")
    if len(payload) > limit:
        payload = payload[-limit:]
    return payload.decode("utf-8", errors="replace")


def run_command(command: list[str], cwd: Path, *, timeout: float) -> dict[str, Any]:
    started_at = utc_now()
    completed, timed_out = run_process_capture(command, cwd=cwd, timeout=timeout)
    safe_command = receipt_command(command)
    stdout = bounded_receipt_output(completed.stdout)
    stderr = bounded_receipt_output(completed.stderr)
    record = {
        "command": safe_command,
        "cwd": receipt_path(cwd),
        "started_at": started_at,
        "finished_at": utc_now(),
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }
    command_display = shlex.join(safe_command)
    error_tail = bounded_receipt_output(stderr, ERROR_OUTPUT_BYTES)
    if timed_out:
        raise SkillError(f"command timed out after {timeout:g}s: {command_display}\n{error_tail}")
    if completed.returncode != 0:
        raise SkillError(f"command failed ({completed.returncode}): {command_display}\n{error_tail}")
    return record


def build_update_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(SCRIPT_DIR / "update_sources.py"), "--output", args.update_output]
    if args.offline_update_fixture:
        command.extend(["--offline-fixture", args.offline_update_fixture])
    elif not args.skip_live_collectors:
        command.append("--fail-on-network-error")
    return command


def build_contributor_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "collect_contributors.py"),
        "--repo", "ml-explore/mlx",
        "--requested-count", "1000",
        "--output", args.contributor_output,
    ]
    if args.offline_contributor_fixture:
        command.extend(["--offline-fixture", args.offline_contributor_fixture])
    return command


def build_backlog_reconcile_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "knowledge_curator.py"),
        "--reconcile-backlog",
        "--previous-graph", args.graph_output,
        "--update-candidates", args.update_output,
        "--research-backlog", args.research_backlog,
    ]


def read_gap_hints(delta_path: Path) -> list[str]:
    try:
        delta = load_structured(delta_path)
    except SkillError:
        return []
    hints = delta.get("gap_hints")
    return [str(item) for item in hints] if isinstance(hints, list) else []


def research_loop_command(run_id: str, run_dir: Path, gap_hints: list[str], agent_count: int) -> list[str]:
    output_dir = run_dir / "research-loop"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "research_loop.py"),
        "--run-id", f"{run_id}-research-loop",
        "--objective", "Nightly MLX knowledge curator: top contributors, papers, blogs, package releases, model outcomes, speedup ranges, and app/CLI skill deltas",
        "--assignment-mode", "dynamic",
        "--agent-count", str(agent_count),
        "--min-sampled-targets", "6",
        "--min-non-github-lanes", "4",
        "--require-source-lane", "papers",
        "--require-source-lane", "repositories",
        "--require-source-lane", "repo_local_audit",
        "--output-dir", str(output_dir),
    ]
    for hint in gap_hints:
        command.extend(["--gap-hint", hint])
    return command


def build_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "# Nightly MLX Knowledge Curator",
        "",
        f"- Run id: `{receipt['run_id']}`",
        f"- Started: `{receipt['started_at']}`",
        f"- Finished: `{receipt['finished_at']}`",
        f"- Graph: `{receipt['graph_output']}`",
        f"- Delta: `{receipt['delta_output']}`",
        f"- Research loop: `{receipt.get('research_loop_output', '')}`",
        "",
        "## Commands",
        "",
    ]
    for command in receipt["commands"]:
        command_text = " ".join(command["command"])
        lines.append(f"- `{command_text}` -> {command['returncode']}")
    lines.extend([
        "",
        "## Gap Hints",
        "",
        ", ".join(f"`{hint}`" for hint in receipt.get("gap_hints", [])) or "None.",
        "",
        "## Policy",
        "",
        "- Review-only: candidate evidence may update the graph and research receipts.",
        "- Do not auto-promote skill/app/CLI guidance without source provenance, validation gate, rollback condition, and tests.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        run_id = redact_secret_text(args.run_id or default_run_id())
        run_dir = Path(args.output_root) / slugify(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        delta_output = run_dir / "knowledge-delta.json"
        markdown_output = run_dir / "knowledge-delta.md"
        receipt_output = run_dir / "nightly-run.json"
        receipt_markdown = run_dir / "nightly-run.md"
        commands: list[dict[str, Any]] = []
        started_at = utc_now()
        if args.command_timeout <= 0:
            raise SkillError("--command-timeout must be positive")

        if args.offline_contributor_fixture or not args.skip_live_collectors:
            commands.append(run_command(build_contributor_command(args), REPO_ROOT, timeout=args.command_timeout))
        if args.offline_update_fixture or not args.skip_live_collectors:
            commands.append(run_command(build_update_command(args), REPO_ROOT, timeout=args.command_timeout))

        curator_command = [
            sys.executable,
            str(SCRIPT_DIR / "knowledge_curator.py"),
            "--run-id", run_id,
            "--update-candidates", args.update_output,
            "--contributor-refresh", args.contributor_output,
            "--research-backlog", args.research_backlog,
            "--previous-graph", args.previous_graph,
            "--graph-output", args.graph_output,
            "--delta-output", str(delta_output),
            "--markdown-output", str(markdown_output),
        ]
        commands.append(run_command(curator_command, REPO_ROOT, timeout=args.command_timeout))
        commands.append(
            run_command(build_backlog_reconcile_command(args), REPO_ROOT, timeout=args.command_timeout)
        )
        gap_hints = read_gap_hints(delta_output)
        research_loop_output = ""
        if not args.no_research_loop:
            loop_command = research_loop_command(run_id, run_dir, gap_hints, args.agent_count)
            commands.append(run_command(loop_command, REPO_ROOT, timeout=args.command_timeout))
            research_loop_output = receipt_path(run_dir / "research-loop")

        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "review_only": True,
            "graph_output": receipt_path(args.graph_output),
            "delta_output": receipt_path(delta_output),
            "delta_markdown": receipt_path(markdown_output),
            "research_loop_output": research_loop_output,
            "gap_hints": gap_hints,
            "commands": commands,
            "next_actions": [
                "Review knowledge-delta.md for already-read, unread, and updated sources.",
                "Review the graph-derived research_backlog.json reconciliation and keep its check mode green.",
                "Dispatch or inspect research-loop/campaign.json before turning new leads into skill/app/CLI edits.",
                "If a finding is promotion-ready, make a separate review PR with tests and rollback conditions.",
            ],
        }
        safe_receipt = redact_secrets(receipt)
        if not isinstance(safe_receipt, dict):  # pragma: no cover - receipt is constructed above
            raise SkillError("nightly receipt redaction produced an invalid payload")
        dump_json(safe_receipt, receipt_output)
        receipt_markdown.write_text(build_markdown(safe_receipt), encoding="utf-8")
        print(f"wrote {receipt_output}")
        print(f"wrote {receipt_markdown}")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {redact_secret_text(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
