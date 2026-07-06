#!/usr/bin/env python3
"""Nightly review-only MLX knowledge curator orchestration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured, slugify

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SKILL_ROOT.parent


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
    parser.add_argument("--no-research-loop", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-nightly-knowledge-curator")


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started_at = utc_now()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    record = {
        "command": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": utc_now(),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise SkillError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-1200:]}")
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


def read_gap_hints(delta_path: Path) -> list[str]:
    try:
        delta = load_structured(delta_path)
    except SkillError:
        return []
    hints = delta.get("gap_hints")
    return [str(item) for item in hints[:8]] if isinstance(hints, list) else []


def unread_source_targets(delta_path: Path) -> dict[str, list[dict[str, str]]]:
    delta = load_structured(delta_path)
    kind_to_lane = {
        "paper": "papers",
        "repository": "repositories",
        "package": "packages",
        "hugging_face": "hugging_face",
        "model": "hugging_face",
        "blog": "technical_blogs",
    }
    targets_by_lane: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for source in delta.get("new_unread_sources", []):
        if not isinstance(source, dict):
            continue
        lane_id = kind_to_lane.get(str(source.get("kind", "")))
        label = str(source.get("label") or source.get("id") or "").strip()
        locator = str(source.get("locator") or "").strip()
        if not lane_id or not label or not locator:
            continue
        key = (lane_id, locator)
        if key in seen:
            continue
        seen.add(key)
        targets_by_lane.setdefault(lane_id, []).append({
            "title": label,
            "kind": f"nightly-{source.get('kind', 'source')}",
            "url": locator,
            "sampling_goal": (
                "Review this new unread source from the nightly knowledge delta; "
                "keep any resulting approach experimental until provenance, validation gate, "
                "rollback condition, and tests exist."
            ),
        })
    return targets_by_lane


def write_research_loop_config(delta_path: Path, run_dir: Path) -> Path | None:
    targets_by_lane = unread_source_targets(delta_path)
    if not targets_by_lane:
        return None
    config = load_structured(SKILL_ROOT / "assets" / "research_loop_config.json")
    if not isinstance(config, dict) or not isinstance(config.get("source_lanes"), list):
        raise SkillError("research loop config must contain source_lanes")
    config = json.loads(json.dumps(config))
    for lane in config["source_lanes"]:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id", ""))
        extra_targets = targets_by_lane.get(lane_id, [])
        if not extra_targets:
            continue
        existing_locators = {
            str(target.get("url") or target.get("query") or target.get("path") or target.get("target") or "")
            for target in lane.get("sample_targets", [])
            if isinstance(target, dict)
        }
        additions = [target for target in extra_targets if target["url"] not in existing_locators]
        if additions:
            lane["sample_targets"] = additions + list(lane.get("sample_targets", []))
    output = run_dir / "research-loop-config.json"
    dump_json(config, output)
    return output


def research_loop_command(
    run_id: str,
    run_dir: Path,
    gap_hints: list[str],
    agent_count: int,
    config_path: Path | None = None,
) -> list[str]:
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
    if config_path is not None:
        command.extend(["--config", str(config_path)])
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
        f"- Research loop config: `{receipt.get('research_loop_config', '')}`",
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
        run_id = args.run_id or default_run_id()
        run_dir = Path(args.output_root) / slugify(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        delta_output = run_dir / "knowledge-delta.json"
        markdown_output = run_dir / "knowledge-delta.md"
        receipt_output = run_dir / "nightly-run.json"
        receipt_markdown = run_dir / "nightly-run.md"
        commands: list[dict[str, Any]] = []
        started_at = utc_now()

        if args.offline_contributor_fixture or not args.skip_live_collectors:
            commands.append(run_command(build_contributor_command(args), REPO_ROOT))
        if args.offline_update_fixture or not args.skip_live_collectors:
            commands.append(run_command(build_update_command(args), REPO_ROOT))

        curator_command = [
            sys.executable,
            str(SCRIPT_DIR / "knowledge_curator.py"),
            "--run-id", run_id,
            "--update-candidates", args.update_output,
            "--previous-graph", args.previous_graph,
            "--graph-output", args.graph_output,
            "--delta-output", str(delta_output),
            "--markdown-output", str(markdown_output),
        ]
        commands.append(run_command(curator_command, REPO_ROOT))
        gap_hints = read_gap_hints(delta_output)
        research_loop_config = write_research_loop_config(delta_output, run_dir)
        research_loop_output = ""
        if not args.no_research_loop:
            loop_command = research_loop_command(run_id, run_dir, gap_hints, args.agent_count, research_loop_config)
            commands.append(run_command(loop_command, REPO_ROOT))
            research_loop_output = str(run_dir / "research-loop")

        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "review_only": True,
            "graph_output": str(Path(args.graph_output)),
            "delta_output": str(delta_output),
            "delta_markdown": str(markdown_output),
            "research_loop_config": str(research_loop_config or ""),
            "research_loop_output": research_loop_output,
            "gap_hints": gap_hints,
            "commands": commands,
            "next_actions": [
                "Review knowledge-delta.md for already-read, unread, and updated sources.",
                "Dispatch or inspect research-loop/campaign.json before turning new leads into skill/app/CLI edits.",
                "If a finding is promotion-ready, make a separate review PR with tests and rollback conditions.",
            ],
        }
        dump_json(receipt, receipt_output)
        receipt_markdown.write_text(build_markdown(receipt), encoding="utf-8")
        print(f"wrote {receipt_output}")
        print(f"wrote {receipt_markdown}")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
