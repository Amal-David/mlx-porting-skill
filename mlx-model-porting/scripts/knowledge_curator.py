#!/usr/bin/env python3
"""Build a review-only MLX knowledge graph and daily delta report."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from _common import SkillError, dump_json, load_structured, slugify
from update_sources import normalize_arxiv_revision, parse_arxiv_identity, source_arxiv_revision

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SKILL_ROOT.parent
DEFAULT_RUN_ROOT = SKILL_ROOT / "research-runs"
STOP_TERMS = {
    "and", "approach", "are", "backlog", "before", "candidate", "for",
    "from", "into", "learning", "model", "models", "mlx", "only",
    "outcome", "paper", "source", "that", "the", "this", "with",
}
KNOWN_READ_STATES = {"already_read", "indexed_only", "source_record"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MLX research knowledge graph")
    parser.add_argument("--sources", default=str(SKILL_ROOT / "assets" / "sources.yaml"))
    parser.add_argument("--update-candidates", default=str(SKILL_ROOT / "assets" / "update-candidates.json"))
    parser.add_argument("--optimization-guidance", default=str(SKILL_ROOT / "assets" / "optimization_guidance.yaml"))
    parser.add_argument("--contributor-learnings", default=str(SKILL_ROOT / "assets" / "contributor_learnings.json"))
    parser.add_argument("--research-backlog", default=str(SKILL_ROOT / "assets" / "research_backlog.json"))
    parser.add_argument("--model-outcomes", default=str(SKILL_ROOT / "assets" / "model_outcomes.json"))
    parser.add_argument("--previous-graph", default=str(SKILL_ROOT / "assets" / "knowledge_graph.json"))
    parser.add_argument("--graph-output", default=str(SKILL_ROOT / "assets" / "knowledge_graph.json"))
    parser.add_argument("--delta-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-nightly-knowledge-curator")


def default_delta_path(run_id: str) -> Path:
    return DEFAULT_RUN_ROOT / run_id / "knowledge-delta.json"


def default_markdown_path(run_id: str) -> Path:
    return DEFAULT_RUN_ROOT / run_id / "knowledge-delta.md"


def normalize_locator(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "arxiv.org/abs/" in text or "arxiv.org/pdf/" in text:
        paper_id, _revision = parse_arxiv_identity(text)
        if paper_id:
            return f"https://arxiv.org/abs/{paper_id}"
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        path = parsed.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", "", ""))
    if "/" in text and " " not in text:
        return f"https://github.com/{text.strip('/')}"
    return text.lower()


def github_repository_identity(value: str | None) -> str:
    """Return a stable owner/repository key without weakening pinned locators."""
    locator = normalize_locator(value)
    if not locator:
        return ""
    parsed = urlparse(locator)
    if parsed.netloc.lower() != "github.com":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    owner, repository = parts[:2]
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        return ""
    return f"{owner.lower()}/{repository.lower()}"


def token_set(*values: Any) -> set[str]:
    text = " ".join(flatten_text(value) for value in values)
    return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2 and term not in STOP_TERMS}


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def source_read_state(source: dict[str, Any]) -> str:
    depth = str(source.get("review_depth") or "").lower()
    if depth in {"synthesized", "screened"}:
        return "already_read"
    if depth == "indexed":
        return "indexed_only"
    return "source_record"


def source_node(source: dict[str, Any]) -> dict[str, Any]:
    source_id = str(source.get("id") or slugify(source.get("title") or source.get("url") or "source"))
    locator = normalize_locator(source.get("url"))
    node = {
        "id": f"source:{source_id}",
        "kind": "source",
        "source_id": source_id,
        "label": source.get("title") or source_id,
        "locator": locator,
        "read_state": source_read_state(source),
        "review_depth": source.get("review_depth") or "",
        "source_kind": source.get("kind") or "",
        "owner": source.get("owner") or "",
        "topics": source.get("topics") or [],
        "snapshot": source.get("snapshot") or "",
    }
    paper_id, _url_revision = parse_arxiv_identity(source.get("url"))
    if source.get("kind") == "paper" and paper_id:
        revision = source_arxiv_revision(source)
        node.update({
            "paper_id": paper_id,
            "revision": revision or "",
            "immutable_locator": f"https://arxiv.org/abs/{paper_id}{revision}" if revision else "",
        })
    return node


def flatten_evidence_refs(refs: dict[str, Any] | None) -> list[str]:
    if not isinstance(refs, dict):
        return []
    result: list[str] = []
    for value in refs.values():
        if isinstance(value, list):
            result.extend(str(item) for item in value)
    return result


def add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    nodes[node["id"]] = node


def add_edge(edges: list[dict[str, Any]], source: str, target: str, relation: str, **extra: Any) -> None:
    record = {"source": source, "target": target, "relation": relation}
    record.update({key: value for key, value in extra.items() if value not in (None, "", [], {})})
    edges.append(record)


def build_approaches(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    guidance: dict[str, Any],
    learnings: dict[str, Any],
    backlog: dict[str, Any],
    outcomes: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    approach_terms: dict[str, set[str]] = {}
    for method in guidance.get("methods", []):
        method_id = str(method.get("id") or "")
        if not method_id:
            continue
        node_id = f"approach:{method_id}"
        add_node(nodes, {
            "id": node_id,
            "kind": "approach",
            "label": method_id,
            "status": method.get("status") or "",
            "category": method.get("category") or "",
            "decision_state": "method_registry",
            "summary": method.get("recommendation") or "",
            "validation_gate": first(method.get("validation_gates")),
            "rollback": first(method.get("rollback_conditions")),
        })
        # Retain superseded methods in the graph for provenance, but never
        # surface them as fresh approach leads. A curator match is a review
        # suggestion, and suggesting a registry tombstone would silently undo
        # the decision that superseded it.
        if method.get("status") != "rejected-or-superseded":
            approach_terms[node_id] = token_set(
                method_id,
                method.get("technique_id"),
                method.get("category"),
                method.get("objectives"),
                method.get("applies_to"),
                method.get("recommendation"),
            )
        for source_id in flatten_evidence_refs(method.get("evidence_refs")):
            if source_id in source_by_id:
                add_edge(edges, f"source:{source_id}", node_id, "evidence_for")

    for learning in learnings.get("learnings", []):
        learning_id = str(learning.get("id") or "")
        if not learning_id:
            continue
        node_id = f"learning:{learning_id}"
        add_node(nodes, {
            "id": node_id,
            "kind": "learning",
            "label": learning_id,
            "status": learning.get("status") or "",
            "decision_state": learning.get("status") or "learning",
            "summary": learning.get("porting_skill_change") or learning.get("reason_held") or "",
            "validation_gate": learning.get("validation_gate") or "",
            "rollback": learning.get("rollback_condition") or "",
        })
        approach_terms[node_id] = token_set(learning_id, learning.get("evidence"), learning.get("porting_skill_change"), learning.get("reason_held"))

    for item in backlog.get("items", []):
        backlog_id = str(item.get("id") or "")
        if not backlog_id:
            continue
        node_id = f"backlog:{backlog_id}"
        add_node(nodes, {
            "id": node_id,
            "kind": "backlog_item",
            "label": backlog_id,
            "status": item.get("status") or "",
            "priority": item.get("priority") or "",
            "decision_state": "needs-review" if item.get("status") != "validated" else "validated",
            "summary": item.get("summary") or "",
            "validation_gate": item.get("required_gate") or "",
        })
        approach_terms[node_id] = token_set(backlog_id, item.get("summary"), item.get("affected"), item.get("required_gate"))

    for outcome in outcomes.get("records", []):
        outcome_id = str(outcome.get("id") or "")
        if not outcome_id:
            continue
        node_id = f"outcome:{outcome_id}"
        add_node(nodes, {
            "id": node_id,
            "kind": "model_outcome",
            "label": outcome.get("label") or outcome_id,
            "status": outcome.get("status") or "",
            "decision_state": "outcome_registry",
            "summary": outcome.get("summary") or "",
            "validation_gate": outcome.get("next_validation") or "",
        })
        approach_terms[node_id] = token_set(outcome_id, outcome.get("summary"), outcome.get("worked"), outcome.get("did_not_work"))
        for source_id in outcome.get("source_ids", []):
            if source_id in source_by_id:
                add_edge(edges, f"source:{source_id}", node_id, "evidence_for_outcome")
    return approach_terms


def first(values: Any) -> str:
    return str(values[0]) if isinstance(values, list) and values else ""


def best_matches(candidate: dict[str, Any], approach_terms: dict[str, set[str]], limit: int = 3) -> list[dict[str, Any]]:
    terms = token_set(candidate.get("label"), candidate.get("summary"), candidate.get("query"), candidate.get("topics"))
    scored: list[tuple[int, str, list[str]]] = []
    for approach_id, terms_for_approach in approach_terms.items():
        matched = sorted(terms & terms_for_approach)
        if len(matched) >= 2:
            scored.append((len(matched), approach_id, matched[:8]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"target": target, "score": score, "matched_terms": matched} for score, target, matched in scored[:limit]]


def previous_index(path: Path) -> tuple[dict[str, Any], set[str], dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}, set(), {}
    try:
        graph = load_structured(path)
    except SkillError:
        return {}, set(), {}
    nodes = {node.get("id"): node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("id")}
    locators = {normalize_locator(node.get("locator")) for node in nodes.values() if node.get("locator")}
    return graph, locators, nodes


def compare_revisions(before: str | None, after: str | None, basis: str) -> dict[str, Any]:
    before_value = str(before or "").strip() or None
    after_value = str(after or "").strip() or None
    if before_value and after_value:
        status = "same" if before_value == after_value else "changed"
    elif before_value:
        status = "candidate_unpinned"
    elif after_value:
        status = "comparison_unpinned"
    else:
        status = "unversioned"
    return {
        "before": before_value,
        "after": after_value,
        "status": status,
        "basis": basis,
    }


def candidate_node_from_paper(
    paper: dict[str, Any],
    known: dict[str, dict[str, Any]],
    previous_locators: set[str],
    previous_nodes: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    raw_identity = paper.get("id") or paper.get("url")
    parsed_paper_id, parsed_revision = parse_arxiv_identity(raw_identity)
    paper_id = str(parsed_paper_id or paper.get("arxiv_id") or "")
    revision = parsed_revision or normalize_arxiv_revision(paper.get("revision"))
    locator = (
        f"https://arxiv.org/abs/{paper_id}"
        if paper_id
        else normalize_locator(paper.get("canonical_url") or raw_identity)
    )
    known_source = known.get(locator)
    title = paper.get("title") or locator or "paper"
    node_id = f"candidate:paper:{slugify(title)}"
    previous = previous_nodes.get(node_id)
    immutable_locator = f"https://arxiv.org/abs/{paper_id}{revision}" if paper_id and revision else ""

    revision_comparison: dict[str, Any] | None = None
    if known_source:
        revision_comparison = compare_revisions(
            str(known_source.get("revision") or "") or None,
            revision,
            "source_registry",
        )
        read_state = (
            known_source.get("read_state")
            if revision_comparison["status"] == "same"
            else "updated_candidate"
        )
    elif previous:
        revision_comparison = compare_revisions(
            str(previous.get("revision") or "") or None,
            revision,
            "previous_graph",
        )
        if previous.get("read_state") == "updated_candidate" or revision_comparison["status"] in {
            "changed",
            "candidate_unpinned",
            "comparison_unpinned",
        }:
            read_state = "updated_candidate"
        else:
            read_state = "seen_unread_candidate"
    else:
        read_state = "seen_unread_candidate" if locator in previous_locators else "unread_candidate"

    return {
        "id": node_id,
        "kind": "source_candidate",
        "candidate_kind": "paper",
        "label": title,
        "locator": locator,
        "immutable_locator": immutable_locator,
        "paper_id": paper_id,
        "revision": revision or "",
        "revision_comparison": revision_comparison,
        "read_state": read_state,
        "known_source_id": known_source.get("id") if known_source else "",
        "query": paper.get("query") or "",
        "updated": paper.get("updated") or "",
        "published": paper.get("published") or "",
        "authors": paper.get("authors") or [],
        "summary": paper.get("summary") or "",
    }, read_state


def candidate_node_from_repo(
    repo: dict[str, Any],
    known: dict[str, dict[str, Any]],
    known_repositories: dict[str, dict[str, Any]],
    previous_locators: set[str],
    previous_nodes: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    repo_name = repo.get("repo") or ""
    locator = normalize_locator(repo.get("url") or repo_name)
    repository_identity = github_repository_identity(locator)
    known_source = known.get(locator) or known_repositories.get(repository_identity)
    node_id = f"candidate:repository:{slugify(repo_name or locator)}"
    previous = previous_nodes.get(node_id)
    head_sha = repo.get("head_sha") or ""
    known_snapshot = str(known_source.get("snapshot") or "") if known_source else ""
    revision_comparison: dict[str, Any] | None = None
    if known_source:
        revision_comparison = compare_revisions(known_snapshot or None, head_sha or None, "source_registry")
        read_state = (
            known_source.get("read_state")
            if revision_comparison["status"] == "same"
            else "updated_candidate"
        )
    elif previous:
        revision_comparison = compare_revisions(
            str(previous.get("head_sha") or "") or None,
            head_sha or None,
            "previous_graph",
        )
        if previous.get("read_state") == "updated_candidate" or revision_comparison["status"] in {
            "changed",
            "candidate_unpinned",
            "comparison_unpinned",
        }:
            read_state = "updated_candidate"
        else:
            read_state = "seen_unread_candidate"
    elif locator in previous_locators:
        read_state = "seen_unread_candidate"
    else:
        read_state = "unread_candidate"
    return {
        "id": node_id,
        "kind": "source_candidate",
        "candidate_kind": "repository",
        "label": repo_name or locator,
        "locator": locator,
        "read_state": read_state,
        "known_source_id": known_source.get("id") if known_source else "",
        "head_sha": head_sha,
        "revision_comparison": revision_comparison,
        "head_date": repo.get("head_date") or "",
        "head_message": repo.get("head_message") or "",
        "topics": repo.get("topics") or [],
        "summary": repo.get("metadata_warning") or repo.get("head_message") or "",
    }, read_state


def append_candidate(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    candidate: dict[str, Any],
    approach_terms: dict[str, set[str]],
    delta: dict[str, Any],
) -> None:
    add_node(nodes, candidate)
    matches = best_matches(candidate, approach_terms)
    for match in matches:
        add_edge(edges, candidate["id"], match["target"], "candidate_relevant_to", score=match["score"], matched_terms=match["matched_terms"])
    compact = {
        "id": candidate["id"],
        "label": candidate["label"],
        "locator": candidate["locator"],
        "read_state": candidate["read_state"],
        "kind": candidate["candidate_kind"],
    }
    if candidate.get("known_source_id"):
        compact["known_source_id"] = candidate["known_source_id"]
    for field in ("immutable_locator", "revision", "revision_comparison"):
        if candidate.get(field) not in (None, "", {}):
            compact[field] = candidate[field]
    if candidate["read_state"] in KNOWN_READ_STATES:
        delta["already_read_sources"].append(compact)
    elif candidate["read_state"] == "updated_candidate":
        updated = dict(compact)
        if candidate.get("head_sha"):
            updated["head_sha"] = candidate["head_sha"]
        delta["updated_sources"].append(updated)
    elif candidate["read_state"] == "unread_candidate":
        delta["new_unread_sources"].append(compact)
    if matches and candidate["read_state"] not in KNOWN_READ_STATES | {"seen_unread_candidate"}:
        delta["new_approach_leads"].append(compact | {"matches": matches})


def add_candidate_lineage(edges: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    known_source_id = candidate.get("known_source_id")
    if not known_source_id:
        return
    add_edge(
        edges,
        candidate["id"],
        str(known_source_id),
        "candidate_version_of",
        revision_comparison=candidate.get("revision_comparison"),
    )


def build_markdown(delta: dict[str, Any], graph: dict[str, Any]) -> str:
    lines = [
        "# MLX Knowledge Curator Delta",
        "",
        f"- Run id: `{delta['run_id']}`",
        f"- Generated at: `{delta['generated_at']}`",
        f"- Graph nodes: {graph['node_count']}",
        f"- Graph edges: {graph['edge_count']}",
        "",
        "## New Unread Sources",
    ]
    lines.extend(markdown_items(delta["new_unread_sources"]))
    lines.extend(["", "## Already Read Sources"])
    lines.extend(markdown_items(delta["already_read_sources"]))
    lines.extend(["", "## Updated Sources"])
    lines.extend(markdown_items(delta["updated_sources"]))
    lines.extend(["", "## New Approach Leads"])
    if delta["new_approach_leads"]:
        for lead in delta["new_approach_leads"]:
            targets = ", ".join(match["target"] for match in lead.get("matches", []))
            lines.append(f"- `{lead['id']}` -> {targets}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Gap Hints", ""])
    lines.append(", ".join(f"`{hint}`" for hint in delta["gap_hints"]) or "None.")
    lines.extend([
        "",
        "## Policy",
        "",
        "- Review-only. Do not auto-promote a candidate source.",
        "- A candidate can update skill/app/CLI guidance only after provenance, validation gate, rollback condition, and tests are recorded.",
    ])
    return "\n".join(lines) + "\n"


def markdown_items(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None."]
    lines: list[str] = []
    for item in items:
        line = f"- `{item['id']}` - {item['label']} ({item.get('locator', '')})"
        comparison = item.get("revision_comparison")
        if isinstance(comparison, dict):
            before = comparison.get("before") or "unpinned"
            after = comparison.get("after") or "unpinned"
            line += f"; revision `{before}` -> `{after}` ({comparison.get('status', 'unknown')})"
        lines.append(line)
    return lines


def derive_gap_hints(delta: dict[str, Any]) -> list[str]:
    hints: dict[str, int] = {}

    def record(term: str, weight: int) -> None:
        if len(term) <= 2 or term.isdigit() or term in STOP_TERMS or term == "abs":
            return
        hints[term] = hints.get(term, 0) + weight

    for lead in delta["new_approach_leads"]:
        for term in token_set(lead.get("label")):
            record(term, 2)
        for match in lead.get("matches", []):
            for term in re.findall(r"[a-z0-9]+", str(match.get("target", "")).lower()):
                record(term, 3)
    for item in delta["new_unread_sources"][:8]:
        for term in token_set(item.get("label"), item.get("kind")):
            record(term, 1)
    return [term for term, _score in sorted(hints.items(), key=lambda item: (-item[1], item[0]))[:12]]


def main() -> int:
    args = parse_args()
    try:
        run_id = args.run_id or default_run_id()
        previous_graph, previous_locators, previous_nodes = previous_index(Path(args.previous_graph))
        sources = load_structured(args.sources)
        candidates = load_structured(args.update_candidates)
        guidance = load_structured(args.optimization_guidance)
        learnings = load_structured(args.contributor_learnings)
        backlog = load_structured(args.research_backlog)
        outcomes = load_structured(args.model_outcomes)

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        source_by_locator: dict[str, dict[str, Any]] = {}
        source_by_repository: dict[str, dict[str, Any]] = {}
        for source in sources.get("sources", []):
            if not isinstance(source, dict):
                continue
            node = source_node(source)
            add_node(nodes, node)
            source_by_id[node["source_id"]] = node
            if node["locator"]:
                source_by_locator[node["locator"]] = node
            repository_identity = github_repository_identity(node["locator"])
            if node["source_kind"] == "repository" and repository_identity:
                source_by_repository.setdefault(repository_identity, node)

        approach_terms = build_approaches(nodes, edges, guidance, learnings, backlog, outcomes, source_by_id)
        delta: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at": utc_now(),
            "previous_graph_generated_at": previous_graph.get("generated_at", ""),
            "already_read_sources": [],
            "new_unread_sources": [],
            "updated_sources": [],
            "new_approach_leads": [],
            "gap_hints": [],
            "policy": {
                "review_only": True,
                "auto_promote_sources": False,
                "auto_modify_recommendations": False,
            },
        }

        for paper in candidates.get("papers", []):
            node, _read_state = candidate_node_from_paper(
                paper,
                source_by_locator,
                previous_locators,
                previous_nodes,
            )
            append_candidate(nodes, edges, node, approach_terms, delta)
            add_candidate_lineage(edges, node)

        for repo in candidates.get("repositories", []):
            node, _read_state = candidate_node_from_repo(
                repo,
                source_by_locator,
                source_by_repository,
                previous_locators,
                previous_nodes,
            )
            append_candidate(nodes, edges, node, approach_terms, delta)
            add_candidate_lineage(edges, node)

        delta["gap_hints"] = derive_gap_hints(delta)
        graph = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "run_id": run_id,
            "policy": delta["policy"],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
            "edges": sorted(edges, key=lambda item: (item["source"], item["target"], item["relation"])),
            "latest_delta": delta,
        }
        graph_output = Path(args.graph_output)
        delta_output = Path(args.delta_output) if args.delta_output else default_delta_path(run_id)
        markdown_output = Path(args.markdown_output) if args.markdown_output else default_markdown_path(run_id)
        dump_json(graph, graph_output)
        dump_json(delta, delta_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(build_markdown(delta, graph), encoding="utf-8")
        print(f"wrote {graph_output}: {graph['node_count']} nodes, {graph['edge_count']} edges")
        print(f"wrote {delta_output}")
        print(f"wrote {markdown_output}")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
