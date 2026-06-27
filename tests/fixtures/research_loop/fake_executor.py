#!/usr/bin/env python3
"""Deterministic local researcher used by research_loop.py executor tests."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    if os.environ.get("MLX_FAKE_EXECUTOR_FAIL"):
        print("forced fake executor failure", file=sys.stderr)
        return 3
    persona_id = os.environ["MLX_RESEARCH_PERSONA_ID"]
    result_path = Path(os.environ["MLX_RESEARCH_RESULT_PATH"])
    blog_path = Path(os.environ["MLX_RESEARCH_BLOG_PATH"])
    slug = persona_id.replace("_", "-")
    sleep_seconds = float(os.environ.get("MLX_FAKE_EXECUTOR_SLEEP", "0") or "0")
    marker_dir = os.environ.get("MLX_FAKE_EXECUTOR_CONCURRENCY_DIR")
    marker_path = None
    active_count = None
    if marker_dir:
        marker_root = Path(marker_dir)
        marker_root.mkdir(parents=True, exist_ok=True)
        marker_path = marker_root / f"{slug}.active"
        marker_path.write_text(str(time.time()), encoding="utf-8")
        time.sleep(min(max(sleep_seconds, 0.1), 0.25))
        active_count = len(list(marker_root.glob("*.active")))
        (marker_root / f"{slug}.json").write_text(json.dumps({
            "persona_id": persona_id,
            "active_count": active_count,
        }, indent=2), encoding="utf-8")
    elif sleep_seconds > 0:
        time.sleep(sleep_seconds)
    result_path.write_text(json.dumps({
        "persona_id": persona_id,
        "decision_notes": [
            "Fake executor returned deterministic review-only findings for receipt tests."
        ],
        "findings": [
            {
                "id": f"fake-executor-{slug}",
                "title": f"Executor receipt for {persona_id}",
                "summary": "A local executor can return structured review findings without network or remote model execution.",
                "source_lane": "official_docs",
                "sources": [
                    {
                        "title": "MLX documentation index",
                        "url": "https://ml-explore.github.io/mlx/build/html/index.html",
                        "accessed": "2026-06-27",
                        "kind": "official-doc"
                    }
                ],
                "decision": "held",
                "evidence_level": "deterministic-test-fixture",
                "validation_gate": "Executor receipts must be inspected before any finding is promoted.",
                "affects": [
                    "references/deep-research-loop.md"
                ],
                "caveats": [
                    "This fixture proves harness behavior, not a real MLX recommendation."
                ],
                "required_next_validation": "Run a live researcher loop and review sources before promotion."
            }
        ]
    }, indent=2), encoding="utf-8")
    if os.environ.get("MLX_FAKE_EXECUTOR_WRITE_BLOG"):
        blog_path.parent.mkdir(parents=True, exist_ok=True)
        blog_path.write_text(
            "\n".join([
                f"# Worker-authored research blog for {persona_id}",
                "",
                "## Assignment",
                "The fake executor wrote this blog through MLX_RESEARCH_BLOG_PATH.",
                "",
                "## Open validation",
                "This proves blog ingestion behavior only.",
                "",
            ]),
            encoding="utf-8",
        )
    if marker_path:
        marker_path.unlink(missing_ok=True)
    print(f"wrote fake finding for {persona_id}")
    if active_count is not None:
        print(f"active workers observed: {active_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
