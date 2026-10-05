#!/usr/bin/env python3
"""Validate and query the offline inference-engine survey; never executes engines."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any

from _common import SkillError, atomic_write_text

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "assets" / "inference_engines.json"
REPORT = ROOT.parent / "INFERENCE_ENGINE_SURVEY.md"
SITE_REPORT = ROOT.parent / "site" / "engines.html"
MAX_BYTES = 2 * 1024 * 1024
WORKLOADS = ("interactive", "coding-agent", "multi-user", "multimodal", "audio", "embeddings", "distributed", "specialized")
CATEGORIES = {"reference-library", "serving-runtime", "embedded-runtime", "distributed-runtime", "non-mlx-comparator", "opaque-engine", "research-runtime", "specialized-benchmark", "native-runtime", "native-specialized", "specialized-runtime", "speculation-adapter", "non-mlx-specialized"}
ENGINE_FIELDS = {"id", "name", "repository", "revision", "commit_date", "category", "runtime", "license", "license_note", "workloads", "methods", "scope", "limitations", "evidence", "local_validation"}
METHOD_FIELDS = {"id", "priority", "lossiness", "bottleneck", "approach", "correctness", "measure", "rollback", "source_ids"}
EVIDENCE_FIELDS = {"path", "sha256", "url", "review_scope"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SkillError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate registry key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise SkillError(f"non-finite registry value: {value}")


def _exact(row: Any, fields: set[str], label: str) -> None:
    require(type(row) is dict and set(row) == fields, f"{label} fields do not match the closed schema")


def _strings(value: Any, label: str) -> list[str]:
    require(type(value) is list and all(type(x) is str and x for x in value), f"{label} must contain strings")
    require(len(value) == len(set(value)), f"{label} contains duplicates")
    return value


def validate_registry(data: Any) -> dict[str, Any]:
    _exact(data, {"schema_version", "reviewed", "scope", "legacy_source_policy", "engines", "methods"}, "registry")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "unsupported registry schema")
    for field in ("reviewed", "scope", "legacy_source_policy"):
        require(type(data[field]) is str and bool(data[field]), f"missing registry {field}")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["reviewed"]) is not None, "invalid review date")
    for field in ("engines", "methods"):
        require(type(data[field]) is list and 0 < len(data[field]) <= 128, f"invalid {field} inventory")
    engine_ids, method_ids, repositories = set(), set(), set()
    for row in data["methods"]:
        _exact(row, METHOD_FIELDS, "method")
        require(type(row["id"]) is str and row["id"] not in method_ids, "duplicate/invalid method id")
        method_ids.add(row["id"])
        require(all(type(row[k]) is str and row[k] for k in METHOD_FIELDS - {"source_ids"}), "invalid method scalar field")
        require(row["priority"] in {"P0", "P1", "P2", "P3"}, "invalid method priority")
        require(row["lossiness"] in {"none", "conditional", "lossy"}, "invalid lossiness")
        for key in ("bottleneck", "approach", "correctness", "measure", "rollback"):
            require(type(row[key]) is str and bool(row[key]), f"method lacks {key}")
        require(bool(_strings(row["source_ids"], "method.source_ids")), "method has no evidence")
    for row in data["engines"]:
        _exact(row, ENGINE_FIELDS, "engine")
        for key in ENGINE_FIELDS - {"workloads", "methods", "evidence"}:
            require(type(row[key]) is str and bool(row[key]), f"engine lacks {key}")
        require(row["id"] not in engine_ids and row["repository"].lower() not in repositories, "duplicate engine/repository")
        engine_ids.add(row["id"]); repositories.add(row["repository"].lower())
        require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repository"]) is not None, "invalid repository identity")
        require(re.fullmatch(r"[0-9a-f]{40}", row["revision"]) is not None, "engine revision must be immutable")
        require(row["category"] in CATEGORIES, "unrecognized engine category")
        require(row["local_validation"] == "not-run", "survey cannot confer local validation")
        require(bool(_strings(row["workloads"], "engine.workloads")) and set(row["workloads"]) <= set(WORKLOADS), "unknown workload")
        require(set(_strings(row["methods"], "engine.methods")) <= method_ids, "unknown method reference")
        require(type(row["evidence"]) is list and 0 < len(row["evidence"]) <= 32, "missing/bounded engine evidence required")
        seen_paths = set()
        for item in row["evidence"]:
            _exact(item, EVIDENCE_FIELDS, "evidence")
            require(all(type(v) is str and v for v in item.values()), "invalid evidence values")
            path = PurePosixPath(item["path"])
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in item["path"] and item["path"] not in seen_paths, "unsafe/duplicate evidence path")
            seen_paths.add(item["path"])
            require(item["url"] == f"https://github.com/{row['repository']}/blob/{row['revision']}/{item['path']}", "evidence URL must bind exact repository/revision/path")
            require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None, "invalid evidence byte digest")
            require(item["review_scope"] in {"readme-reviewed", "license-file-captured", "selected-source-sections", "architecture-doc-reviewed"}, "invalid review scope")
        require(any(e["path"].lower() == "readme.md" for e in row["evidence"]), "engine README evidence is required")
    for row in data["methods"]:
        require(set(row["source_ids"]) <= engine_ids, "method references missing engine evidence")
    return data


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(), "registry must be a regular non-symlink file")
    require(metadata.st_size <= MAX_BYTES, "registry exceeds byte bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "registry changed to a non-regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "registry exceeds byte bound")
    finally:
        os.close(descriptor)
    return validate_registry(json.loads(raw, object_pairs_hook=_pairs, parse_constant=_bad_constant))


def shortlist(data: dict[str, Any], workload: str, *, blockers: list[str] | None = None, include_comparators: bool = False) -> dict[str, Any]:
    validate_registry(data)
    require(workload in WORKLOADS, "unknown serving workload")
    held = list(blockers or [])
    require(all(type(x) is str for x in held), "blockers must be strings")
    excluded = {"non-mlx-comparator", "non-mlx-specialized", "opaque-engine", "specialized-benchmark"}
    rows = [row for row in data["engines"] if workload in row["workloads"] and (include_comparators or row["category"] not in excluded)]
    return {"schema_version": 1, "reviewed": data["reviewed"], "workload": workload, "scope": "workload-fit-research-shortlist-not-model-compatibility", "order": "alphabetical-not-performance-ranked", "execution_allowed": False, "inspection_blockers": held, "candidates": [{"id": row["id"], "name": row["name"], "category": row["category"], "revision": row["revision"], "source": next(e["url"] for e in row["evidence"] if e["path"].lower() == "readme.md"), "rationale": row["scope"], "limitations": row["limitations"], "local_validation": "not-run", "eligibility": "blocked-by-inspection" if held else "requires-model-and-runtime-qualification"} for row in sorted(rows, key=lambda row: row["name"].casefold())], "required_gates": ["Exact model, tokenizer, processor, template, adapter and weight-layout support.", "Artifact/runtime license and isolated execution review.", "Reference parity plus task-quality checks on the requested configuration.", "Comparable cold/warm/partial-cache and concurrency measurements before selection."]}


def render(data: dict[str, Any]) -> str:
    validate_registry(data)
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")
    lines = ["# Apple Silicon inference-engine survey", "", "<!-- Generated by inference_engine_advisor.py from assets/inference_engines.json. -->", "", f"Reviewed {data['reviewed']}. {len(data['engines'])} pinned projects; {len(data['methods'])} method/validation contracts.", "", data["scope"], "", "All engine measurements remain **not run locally in this survey**. Reading code is not runtime qualification. Exact pins describe the inspected snapshot, not a claim that every feature ships in the latest stable release.", "", "## Engine inventory", "", "| Project | Layer | Runtime | Declared license boundary | Workloads |", "|---|---|---|---|---|"]
    for row in data["engines"]:
        readme = next(e["url"] for e in row["evidence"] if e["path"].lower() == "readme.md")
        lines.append(f"| [{cell(row['name'])}]({readme}) | {row['category']} | {cell(row['runtime'])} | {cell(row['license'])} | {', '.join(row['workloads'])} |")
    lines.extend(["", "## What to learn, and what would invalidate it", ""])
    by_id = {e["id"]: e for e in data["engines"]}
    for row in data["methods"]:
        refs = ", ".join(f"[{by_id[s]['name']}](https://github.com/{by_id[s]['repository']}/tree/{by_id[s]['revision']})" for s in row["source_ids"])
        lines.extend([f"### {row['priority']} / {row['id']}", "", "**Bottleneck:** " + row["bottleneck"], "", row["approach"], "", "**Correctness gate:** " + row["correctness"], "", "**Measure:** " + row["measure"], "", "**Rollback:** " + row["rollback"], "", f"**Approximation classification:** {row['lossiness']}. **Primary sources:** {refs}.", ""])
    lines.extend(["## Per-project findings and provenance", ""])
    for row in data["engines"]:
        lines.extend(["### " + row["name"], "", row["scope"], "", "**Limitations:** " + row["limitations"], "", "**License review:** " + row["license_note"], "", f"Snapshot `{row['revision']}`; upstream commit date `{row['commit_date']}`. Local validation: **not run**.", ""])
        for item in row["evidence"]:
            lines.append(f"- [{item['path']}]({item['url']}) — {item['review_scope']}; SHA-256 `{item['sha256']}`.")
        lines.append("")
    lines.extend(["## Reproduction and scope", "", "The JSON registry is the canonical engine/method inventory. Evidence digests bind the bytes fetched during the review; offline validation checks their format and locator consistency, not the continued availability of upstream bytes. No third-party engine was installed or benchmarked, no upstream implementation is vendored, and no source's performance claim is promoted.", "", data["legacy_source_policy"], "", "```bash", "python3 mlx-model-porting/scripts/inference_engine_advisor.py --workload coding-agent", "python3 mlx-model-porting/scripts/inference_engine_advisor.py --validate", "python3 mlx-model-porting/scripts/inference_engine_advisor.py --generate", "python3 mlx-model-porting/scripts/inference_engine_advisor.py --check", "```", ""])
    return "\n".join(lines)



def render_html(data: dict[str, Any]) -> str:
    validate_registry(data)
    esc = html.escape
    parts = ['<!doctype html>', '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">', '<title>Inference engines — MLX Porter</title><meta name="description" content="Pinned Apple Silicon engine survey with workload-fit and validation boundaries."><link rel="stylesheet" href="./styles.css"><link rel="icon" href="./favicon.svg" type="image/svg+xml"></head><body>', '<a class="skip-link" href="#main">Skip to content</a><header class="site-header"><nav class="nav-shell" aria-label="Primary navigation"><a class="brand" href="./index.html">MLX Porter</a><a href="./docs/index.html">Documentation</a></nav></header>', '<main id="main"><section class="section"><div class="section-shell"><div class="section-heading"><span class="section-kicker">Evidence before deployment</span><h1>Apple Silicon inference engines</h1></div>', f'<p>Reviewed {esc(data["reviewed"])}. {len(data["engines"])} pinned projects and {len(data["methods"])} method contracts. This is not a performance leaderboard. No third-party engine was benchmarked in this survey.</p>', '<p>MLX-VLM remains the multimodal reference; serving layers, native specialized runtimes and non-MLX comparators solve different problems. Expand a project for its scope and limitations.</p>']
    for e in data["engines"]:
        url = next(x["url"] for x in e["evidence"] if x["path"].lower() == "readme.md")
        parts.append(f'<details><summary>{esc(e["name"])} · {esc(e["category"])}</summary><p>{esc(e["scope"])}</p><p><strong>Limit:</strong> {esc(e["limitations"])}</p><p><strong>Runtime:</strong> {esc(e["runtime"])}. <strong>Workloads:</strong> {esc(", ".join(e["workloads"]))}.</p><p><a href="{esc(url, quote=True)}">Pinned primary source</a> · Local validation: not run.</p></details>')
    parts.append('<h2>Methods and required gates</h2>')
    for m in data["methods"]:
        parts.append(f'<details><summary>{esc(m["priority"])} · {esc(m["id"])}</summary><p>{esc(m["approach"])}</p><p><strong>Correctness:</strong> {esc(m["correctness"])}</p><p><strong>Measure:</strong> {esc(m["measure"])}</p><p><strong>Rollback:</strong> {esc(m["rollback"])}</p></details>')
    parts.extend(['<h2>Agent workflow</h2><pre><code>python3 mlx-model-porting/scripts/inference_engine_advisor.py --workload coding-agent</code></pre><p>The offline registry and generated Markdown report include exact repository revisions, fetched-file hashes and evidence scope. Recognition never grants execution or promotion.</p>', '</div></section></main></body></html>', ''])
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--include-comparators", action="store_true")
    parser.add_argument("--validate", action="store_true", help="validate only the installed registry")
    parser.add_argument("--generate", action="store_true", help="regenerate the repository-root survey")
    parser.add_argument("--check", action="store_true", help="check registry and generated survey drift")
    args = parser.parse_args(argv)
    try:
        require(sum(bool(v) for v in (args.workload, args.validate, args.generate, args.check)) <= 1, "choose one query/validation/generation mode")
        data = load_registry()
        if args.generate or args.check:
            text = render(data)
            page = render_html(data)
            if args.check:
                require(REPORT.is_file() and not REPORT.is_symlink() and REPORT.read_text(encoding="utf-8") == text, "generated engine survey is missing or stale")
                require(SITE_REPORT.is_file() and not SITE_REPORT.is_symlink() and SITE_REPORT.read_text(encoding="utf-8") == page, "generated engine site page is missing or stale")
            else:
                atomic_write_text(REPORT, text)
                atomic_write_text(SITE_REPORT, page)
            result = {"ok": True, "engine_count": len(data["engines"]), "method_count": len(data["methods"])}
        elif args.workload:
            result = shortlist(data, args.workload, include_comparators=args.include_comparators)
        elif args.validate:
            result = {"ok": True, "engine_count": len(data["engines"]), "method_count": len(data["methods"])}
        else:
            result = data
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (SkillError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
