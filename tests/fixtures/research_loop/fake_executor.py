#!/usr/bin/env python3
"""Deterministic local researcher used by research_loop.py executor tests."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if os.environ.get("MLX_FAKE_EXECUTOR_FAIL"):
        print("forced fake executor failure", file=sys.stderr)
        return 3
    persona_id = os.environ["MLX_RESEARCH_PERSONA_ID"]
    result_path = Path(os.environ["MLX_RESEARCH_RESULT_PATH"])
    slug = persona_id.replace("_", "-")
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
    print(f"wrote fake finding for {persona_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
