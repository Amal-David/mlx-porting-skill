#!/usr/bin/env python3
"""Render EVIDENCE_INDEX.md from the canonical source registry.

The source registry remains in author-curated order. Rendering uses sorted copies for
stable review diffs and never writes back to the registry. ``--check`` validates pinned
synthesized GitHub evidence and fails when the committed index differs from a fresh
render.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlsplit

from _common import SkillError, atomic_write_text, load_structured


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SKILL_ROOT.parent
DEFAULT_SOURCES = SKILL_ROOT / "assets" / "sources.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "EVIDENCE_INDEX.md"
GITHUB_REF_KINDS = {"blob", "tree"}
MARKDOWN_LINK_SAFE = ":/?#@!$&'*+,;=%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or check the canonical evidence index")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES), help="Canonical sources registry")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Rendered Markdown path")
    parser.add_argument("--check", action="store_true", help="Fail if the rendered index has drifted")
    return parser.parse_args()


def markdown_cell(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "—"
    return (
        text.replace("\\", "\\\\")
        .replace("<", "&lt;")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
    )


def markdown_link_destination(
    value: object,
    *,
    allow_relative: bool = False,
    label: str,
) -> str:
    url = str(value).strip()
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise SkillError(f"{label} is not a valid URL: {url}") from exc
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        is_relative = not parsed.scheme and not parsed.netloc and not url.startswith(("/", "\\"))
        if not allow_relative or not is_relative:
            raise SkillError(f"{label} must use https: {url}")
    return quote(url, safe=MARKDOWN_LINK_SAFE)


def code_list(values: Iterable[object]) -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    return ", ".join(f"`{markdown_cell(value)}`" for value in normalized) or "—"


def count_rows(counter: collections.Counter[str]) -> list[str]:
    return [f"| `{markdown_cell(label)}` | {count} |" for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def github_pin_error(source: dict[str, Any]) -> str | None:
    """Return an error when synthesized GitHub repository/blob/tree evidence moves."""
    if source.get("review_depth") != "synthesized":
        return None
    url = str(source.get("url", ""))
    parsed = urlsplit(url)
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None

    source_id = source.get("id", "<missing-id>")
    snapshot = str(source.get("snapshot", "")).strip()
    if len(parts) == 2:
        return f"synthesized source {source_id} has moving GitHub URL {url}; pin repository root to /tree/{snapshot}"

    if len(parts) >= 4 and parts[2] in GITHUB_REF_KINDS:
        url_ref = parts[3]
        if not snapshot or url_ref != snapshot:
            return (
                f"synthesized source {source_id} has moving GitHub URL {url}; "
                f"the {parts[2]} ref must equal recorded snapshot {snapshot!r}"
            )
    return None


def validate_registry(registry: dict[str, Any]) -> None:
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise SkillError("sources registry must contain a sources list")
    if registry.get("count") != len(sources):
        raise SkillError("sources registry count does not match the source list")

    ids: list[str] = []
    pin_errors: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            raise SkillError("every source record must be an object")
        source_id = str(source.get("id", "")).strip()
        if not source_id:
            raise SkillError("source record is missing id")
        ids.append(source_id)
        pin_error = github_pin_error(source)
        if pin_error:
            pin_errors.append(pin_error)
    duplicates = sorted(source_id for source_id, count in collections.Counter(ids).items() if count > 1)
    if duplicates:
        raise SkillError(f"duplicate source IDs: {', '.join(duplicates)}")
    if pin_errors:
        raise SkillError("moving GitHub URL validation failed:\n- " + "\n- ".join(pin_errors))


def definitions_table(
    definitions: dict[str, str],
    counts: collections.Counter[str],
    *,
    label: str,
) -> list[str]:
    rows = [f"| {label} | Records | Meaning |", "|---|---:|---|"]
    for identifier in sorted(definitions):
        rows.append(
            f"| `{markdown_cell(identifier)}` | {counts.get(identifier, 0)} | "
            f"{markdown_cell(definitions[identifier])} |"
        )
    return rows


def render_index(registry: dict[str, Any]) -> str:
    validate_registry(registry)
    sources: list[dict[str, Any]] = registry["sources"]
    reviewed = markdown_cell(registry.get("reviewed", "unknown"))
    depth_definitions = registry.get("review_depth_definitions", {})
    scope_definitions = registry.get("support_scope_definitions", {})
    claim_definitions = registry.get("claim_type_definitions", {})
    boundary_policy = registry.get("claim_boundary_policy", [])
    benchmark_assessment = str(registry.get("benchmark_assessment", "")).strip()
    if not isinstance(depth_definitions, dict) or not depth_definitions:
        raise SkillError("review_depth_definitions must be a non-empty object")
    if not isinstance(scope_definitions, dict) or not scope_definitions:
        raise SkillError("support_scope_definitions must be a non-empty object")
    if not isinstance(claim_definitions, dict) or not claim_definitions:
        raise SkillError("claim_type_definitions must be a non-empty object")
    if not isinstance(boundary_policy, list) or not boundary_policy:
        raise SkillError("claim_boundary_policy must be a non-empty list")
    if not benchmark_assessment:
        raise SkillError("benchmark_assessment must link the benchmark assessment")
    benchmark_assessment = markdown_link_destination(
        benchmark_assessment,
        allow_relative=True,
        label="benchmark_assessment",
    )

    kind_counts = collections.Counter(str(source.get("kind", "unclassified")) for source in sources)
    depth_counts = collections.Counter(str(source.get("review_depth", "unclassified")) for source in sources)
    scope_counts = collections.Counter(str(source.get("support_scope", "not-classified")) for source in sources)
    evidence_counts = collections.Counter(str(source.get("evidence_class", "not-classified")) for source in sources)
    topic_counts = collections.Counter(str(topic) for source in sources for topic in source.get("topics", []))
    claim_counts = collections.Counter(str(claim) for source in sources for claim in source.get("claim_types", []))
    scope_render_definitions = {
        **scope_definitions,
        "not-classified": (
            "No support scope is encoded. The record is citation evidence only and grants no implicit support claim."
        ),
    }

    lines = [
        "# Evidence index",
        "",
        "<!-- Generated by mlx-model-porting/scripts/generate_evidence_index.py; edit sources.yaml, not this file. -->",
        "",
        f"**Snapshot date:** {reviewed}",
        f"**Total records:** {len(sources)}",
        "",
        "This is the deterministic, navigable rendering of the canonical "
        "`mlx-model-porting/assets/sources.yaml` registry. The registry order is author-curated; "
        "this view sorts records by ID without rewriting that order.",
        "",
        f"Local workload receipts and their limitations are evaluated in the "
        f"[Benchmark assessment]({benchmark_assessment}). A catalogue citation or source-reported "
        "number does not prove target-workload performance.",
        "",
        "## Evidence semantics",
        "",
        "### Review depth",
        "",
        "Review depth states how far a source was reviewed and used by this project. It is not an MLX reproduction status.",
        "",
        *definitions_table(depth_definitions, depth_counts, label="Review depth"),
        "",
        "### Support scope",
        "",
        "Support scope states whose implementation or receipt the evidence covers. It never widens a source beyond its recorded snapshot.",
        "",
        *definitions_table(scope_render_definitions, scope_counts, label="Support scope"),
        "",
        "### Claim boundary",
        "",
        "Every source is constrained by the following claim-boundary rules:",
        "",
        *(f"- {markdown_cell(rule)}" for rule in boundary_policy),
        "",
        "Claim types describe what a classified record may support; absent claim types grant no implicit claim.",
        "",
        *definitions_table(claim_definitions, claim_counts, label="Claim type"),
        "",
        "## Corpus summary",
        "",
        "### By source kind",
        "",
        "| Kind | Count |",
        "|---|---:|",
        *count_rows(kind_counts),
        "",
        "### By evidence class",
        "",
        "| Evidence class | Count |",
        "|---|---:|",
        *count_rows(evidence_counts),
        "",
        "### By topic",
        "",
        "| Topic | Records |",
        "|---|---:|",
        *count_rows(topic_counts),
        "",
        "## Full source list",
        "",
        "Records are sorted by stable source ID for review. The canonical registry remains in its original author-curated order.",
        "",
        "| ID | Depth | Kind | Evidence class | Support scope | Claim types | Owner | Topics | Snapshot | Source | Claim boundary / note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for source in sorted(sources, key=lambda item: str(item["id"])):
        title = markdown_cell(source.get("title", source["id"]))
        url = str(source.get("url", "")).strip()
        link = title
        if url:
            destination = markdown_link_destination(url, label=f"source {source['id']} URL")
            link = f"[{title}]({destination})"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{markdown_cell(source['id'])}`",
                    f"`{markdown_cell(source.get('review_depth', 'not-classified'))}`",
                    f"`{markdown_cell(source.get('kind', 'not-classified'))}`",
                    f"`{markdown_cell(source.get('evidence_class', 'not-classified'))}`",
                    f"`{markdown_cell(source.get('support_scope', 'not-classified'))}`",
                    code_list(source.get("claim_types", [])),
                    markdown_cell(source.get("owner", "")),
                    code_list(source.get("topics", [])),
                    f"`{markdown_cell(source.get('snapshot', ''))}`",
                    link,
                    markdown_cell(source.get("note", "")),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    sources_path = Path(args.sources).resolve()
    output_path = Path(args.output).resolve()
    try:
        registry = load_structured(sources_path)
        if not isinstance(registry, dict):
            raise SkillError("sources registry root must be an object")
        rendered = render_index(registry)
        if args.check:
            if not output_path.exists():
                print(f"error: evidence index drift: missing {output_path}", file=sys.stderr)
                return 1
            current = output_path.read_text(encoding="utf-8")
            if current != rendered:
                print(
                    "error: evidence index drift: regenerate with "
                    f"{Path(__file__).name} --sources {sources_path} --output {output_path}",
                    file=sys.stderr,
                )
                return 1
            print(f"evidence index is current: {output_path}")
            return 0

        atomic_write_text(output_path, rendered)
        print(f"wrote {len(registry['sources'])} source records to {output_path}")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
