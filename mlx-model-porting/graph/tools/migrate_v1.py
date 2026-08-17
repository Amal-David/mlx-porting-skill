#!/usr/bin/env python3
"""Migrate the schema-v1 knowledge assets into v2 evidence-graph shards.

    python3 mlx-model-porting/graph/tools/migrate_v1.py

Reads (never writes) the authored v1 assets under `mlx-model-porting/assets/`
and regenerates `mlx-model-porting/graph/shards/migrated/*.json`. The mapping
is intentionally mechanical so that re-running it after an asset refresh is a
diff, not a re-authoring exercise.

Mapping
-------
    approach            -> mechanism        (+ applies_to edges to trait nodes)
    approach.applies_to -> trait
    learning            -> finding
    backlog_item        -> hypothesis
    model_outcome       -> applied_result   when the record carries BOTH a
                           numeric observed range AND one unambiguous model
                           identity; otherwise finding
    source              -> external_reference, only when an edge connects it to
                           a node we kept (unconnected sources stay in
                           sources.yaml and are not graph nodes)
    source_candidate    -> external_reference (unreviewed lead, low confidence)
    contributor_*       -> dropped; contributor rosters are people, not
                           optimization knowledge, and carry personal data

Provenance is never upgraded. v1 records carry no verified measurement except
the one local BF16-cast observation, and that observation is itself recorded
as held by its own asset, so it lands as `local_measured` / low confidence
with the hold reasons preserved in the effect sample note.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _graphlib as gl  # noqa: E402

ASSETS = gl.REPO_GRAPH_ROOT.parent / "assets"
OUT_DIR = gl.SHARDS_DIR / "migrated"

ASSET_LOCATOR = "mlx-model-porting/assets/%s"

# approach.status -> (provenance, confidence). None of the v1 approach records
# carry a measurement, so none of them may claim a measured provenance.
APPROACH_PROVENANCE = {
    "native-mlx": ("code_verified", "high"),
    "official-mlx-project": ("code_verified", "high"),
    "proven-mlx-port": ("code_verified", "medium"),
    "research-candidate": ("research_inference", "low"),
    "rejected-or-superseded": ("research_inference", "low"),
}

# optimization_stacks.yaml records a per-step lossiness; it is the only
# mechanical exactness signal in the v1 corpus.
LOSSINESS_EXACTNESS = {
    "lossless": "exact_by_construction",
    "conditionally-lossy": "approximate_legal",
}

REVIEW_DEPTH_CONFIDENCE = {"synthesized": "high", "screened": "medium", "indexed": "low"}
CODE_BEARING_KINDS = {"official-doc", "repository", "source-code", "release"}

# learning.status -> confidence. `adopted` learnings changed a rule; held ones
# did not.
LEARNING_CONFIDENCE = {"adopted": "high"}

SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text, limit=120):
    value = SLUG_RE.sub("-", str(text).lower()).strip("-")
    if not value:
        value = "x"
    if len(value) > limit:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        value = value[: limit - 9].rstrip("-") + "-" + digest
    return value


def node(node_id, kind, title, status, confidence, provenance, summary, evidence, tags,
         observed_at, **extra):
    record = {
        "id": node_id,
        "kind": kind,
        "title": title[:300],
        "status": status[:100],
        "confidence": confidence,
        "provenance": provenance,
        "summary": summary[:4000],
        "evidence": [item[:1000] for item in evidence],
        "tags": sorted(set(tags)),
        "observed_at": observed_at,
    }
    record.update(extra)
    return record


def edge(source, target, relation, confidence, rationale):
    return {
        "from": source,
        "to": target,
        "relation": relation,
        "confidence": confidence,
        "rationale": rationale[:2000],
    }


def with_gate(summary, gate, rollback):
    parts = [summary.strip()]
    if gate:
        parts.append("Validation gate: %s" % gate.strip())
    if rollback:
        parts.append("Rollback: %s" % rollback.strip())
    return " ".join(parts)


def load_assets():
    """Every `.yaml` asset in this repo is JSON-compatible; parse with stdlib json."""
    names = (
        "knowledge_graph.json",
        "techniques.yaml",
        "sources.yaml",
        "model_outcomes.json",
        "effective_claims.json",
        "optimization_stacks.yaml",
        "optimization_guidance.yaml",
    )
    assets = {}
    for name in names:
        assets[name] = gl.load_strict(ASSETS / name)
    return assets


def build(assets):
    graph = assets["knowledge_graph.json"]
    observed = str(graph.get("generated_at", ""))[:10] or "2026-07-23"
    by_id = {item["id"]: item for item in graph["nodes"]}

    source_url = {item["id"]: item["url"] for item in assets["sources.yaml"]["sources"]}
    source_depth = {
        item["id"]: item.get("review_depth", "indexed")
        for item in assets["sources.yaml"]["sources"]
    }

    technique_evidence = {
        item["id"]: list(item.get("evidence", []))
        for item in assets["techniques.yaml"]["techniques"]
    }
    technique_title = {
        item["id"]: item["title"] for item in assets["techniques.yaml"]["techniques"]
    }
    exactness = {}
    for stack in assets["optimization_stacks.yaml"]["stacks"]:
        for step in stack["steps"]:
            mapped = LOSSINESS_EXACTNESS.get(step["lossiness"])
            if mapped and exactness.get(step["method"], mapped) == mapped:
                exactness[step["method"]] = mapped
            elif mapped:
                # Conflicting lossiness across stacks: fall back to the weaker claim.
                exactness[step["method"]] = "approximate_legal"

    claim_band = {
        item["method_id"]: item
        for item in assets["effective_claims.json"]["claims"]
    }

    mechanisms, mech_edges = [], []
    traits = {}
    findings, finding_edges = [], []
    hypotheses = []
    results, result_edges = [], []
    references, reference_edges = [], []

    kept = set()

    # ---- approach -> mechanism, applies_to -> trait ----------------------
    for item in sorted(
        (n for n in graph["nodes"] if n["kind"] == "approach"), key=lambda n: n["id"]
    ):
        name = item["id"].split(":", 1)[1]
        mech_id = "mechanism:" + name
        provenance, confidence = APPROACH_PROVENANCE[item["status"]]
        evidence = [ASSET_LOCATOR % "knowledge_graph.json (approach:%s)" % name]
        for source_id in sorted(technique_evidence.get(name, [])):
            if source_id in source_url:
                evidence.append(source_url[source_id])
        band = claim_band.get(name)
        summary = with_gate(item["summary"], item.get("validation_gate"), item.get("rollback"))
        if band and band.get("observed_range"):
            summary += (
                " Observed source band %s on metric %s; the repository holds this claim"
                " (promotion_state=%s) and emits no effective range."
                % (band["observed_range"], band["metric"], band["promotion_state"])
            )
        extra = {}
        if name in exactness:
            extra["exactness_class"] = exactness[name]
        mechanisms.append(
            node(
                mech_id,
                "mechanism",
                technique_title.get(name, item["label"].replace("-", " ")),
                item["status"],
                confidence,
                provenance,
                summary,
                evidence,
                ["migrated-v1", "category/" + item["category"]]
                + ["objective/" + value for value in item.get("objectives", [])],
                observed,
                **extra,
            )
        )
        kept.add(item["id"])
        for applies in sorted(set(item.get("applies_to", []))):
            if applies == "all":
                # `all` is not a discriminating trait; it would attach every
                # mechanism to one hub node and destroy trait-indexed retrieval.
                continue
            trait_id = "trait:" + slug(applies)
            traits.setdefault(
                trait_id,
                node(
                    trait_id,
                    "trait",
                    applies.replace("-", " "),
                    "declared",
                    "medium",
                    "code_verified",
                    "Applicability class used by the v1 optimization registry to scope a"
                    " method. Derived from approach.applies_to; it is a coarse family or"
                    " deployment-shape trait, not a measured structural property.",
                    [ASSET_LOCATOR % "knowledge_graph.json (approach.applies_to)"],
                    ["migrated-v1", "family-trait"],
                    observed,
                ),
            )
            mech_edges.append(
                edge(
                    mech_id,
                    trait_id,
                    "applies_to",
                    confidence,
                    "v1 approach %s declares applies_to %s." % (item["id"], applies),
                )
            )

    # ---- learning -> finding --------------------------------------------
    for item in sorted(
        (n for n in graph["nodes"] if n["kind"] == "learning"), key=lambda n: n["id"]
    ):
        name = item["id"].split(":", 1)[1]
        finding_id = "finding:" + name
        findings.append(
            node(
                finding_id,
                "finding",
                item["label"].replace("-", " "),
                item["status"],
                LEARNING_CONFIDENCE.get(item["status"], "low"),
                "research_inference",
                with_gate(item["summary"], item.get("validation_gate"), item.get("rollback")),
                [ASSET_LOCATOR % "knowledge_graph.json (learning:%s)" % name,
                 ASSET_LOCATOR % "contributor_learnings.json"],
                ["migrated-v1", "contributor-learning"],
                observed,
            )
        )
        kept.add(item["id"])

    # ---- backlog_item -> hypothesis -------------------------------------
    for item in sorted(
        (n for n in graph["nodes"] if n["kind"] == "backlog_item"), key=lambda n: n["id"]
    ):
        name = item["id"].split(":", 1)[1]
        hyp_id = "hypothesis:" + name
        hypotheses.append(
            node(
                hyp_id,
                "hypothesis",
                item["label"].replace("-", " "),
                item["status"],
                "low",
                "research_inference",
                with_gate(item["summary"], item.get("validation_gate"), None),
                [ASSET_LOCATOR % "knowledge_graph.json (backlog:%s)" % name,
                 item.get("source") or ASSET_LOCATOR % "knowledge_graph.json"],
                ["migrated-v1", "backlog", "priority/" + item["priority"]],
                observed,
            )
        )
        kept.add(item["id"])

    # ---- model_outcome -> applied_result | finding ----------------------
    outcomes = {item["id"]: item for item in assets["model_outcomes.json"]["records"]}
    for item in sorted(
        (n for n in graph["nodes"] if n["kind"] == "model_outcome"), key=lambda n: n["id"]
    ):
        name = item["id"].split(":", 1)[1]
        record = outcomes.get(name, {})
        promotion = _measured_result(name, record, observed)
        if promotion is not None:
            result_node, anchors = promotion
            results.append(result_node)
            result_edges.extend(anchors)
            kept.add(item["id"])
            continue
        band = _band_text(record)
        findings.append(
            node(
                "finding:outcome-" + name,
                "finding",
                item["label"],
                item["status"],
                "medium" if item["status"] == "source_backed_working" else "low",
                "research_inference",
                item["summary"] + band,
                [ASSET_LOCATOR % "model_outcomes.json (%s)" % name,
                 ASSET_LOCATOR % "knowledge_graph.json (outcome:%s)" % name],
                ["migrated-v1", "route-outcome"]
                + ["family/" + value for value in item.get("families", [])],
                observed,
            )
        )
        kept.add(item["id"])

    # ---- source / source_candidate -> external_reference ----------------
    target_of = {
        "approach": lambda value: "mechanism:" + value.split(":", 1)[1],
        "learning": lambda value: "finding:" + value.split(":", 1)[1],
        "backlog_item": lambda value: "hypothesis:" + value.split(":", 1)[1],
    }

    def v2_target(v1_id):
        item = by_id.get(v1_id)
        if item is None or v1_id not in kept:
            return None
        if item["kind"] == "model_outcome":
            name = v1_id.split(":", 1)[1]
            spec = MEASURED_OUTCOMES.get(name)
            return spec["result_id"] if spec else "finding:outcome-" + name
        builder = target_of.get(item["kind"])
        return builder(v1_id) if builder else None

    used_sources = {}
    for raw in sorted(graph["edges"], key=lambda e: (e["source"], e["relation"], e["target"])):
        relation = raw["relation"]
        if relation == "contributor_in_refresh":
            continue
        origin = by_id.get(raw["source"])
        if origin is None:
            continue
        if relation == "candidate_version_of":
            if raw["target"] not in used_sources:
                continue
            reference_edges.append(
                edge(
                    _reference_id(origin),
                    used_sources[raw["target"]],
                    "supersedes",
                    "low",
                    "Source registry observed revision %s of the same work."
                    % (raw.get("revision_comparison", {}).get("after", "newer")),
                )
            )
            used_sources.setdefault(raw["source"], _reference_id(origin))
            references.append(_reference_node(origin, source_depth, observed))
            continue
        target = v2_target(raw["target"])
        if target is None:
            continue
        ref_id = _reference_id(origin)
        used_sources.setdefault(raw["source"], ref_id)
        references.append(_reference_node(origin, source_depth, observed))
        if relation in ("evidence_for", "evidence_for_outcome"):
            reference_edges.append(
                edge(ref_id, target, "supports", "medium",
                     "v1 %s edge from %s." % (relation, raw["source"]))
            )
        elif relation == "candidate_relevant_to":
            reference_edges.append(
                edge(ref_id, target, "suggests", "low",
                     "Unreviewed research candidate matched %s on terms %s."
                     % (raw["target"], ", ".join(sorted(raw.get("matched_terms", []))) or "n/a"))
            )

    references = _dedupe(references)
    reference_edges = _dedupe_edges(reference_edges)

    return {
        "mechanisms.json": (mechanisms, mech_edges),
        "traits.json": ([traits[key] for key in sorted(traits)], []),
        "findings.json": (findings, finding_edges),
        "hypotheses.json": (hypotheses, []),
        "results.json": (results, result_edges),
        "references.json": (references, reference_edges),
    }, observed


def _band_text(record):
    parts = []
    for leg in ("overall", "speculative_decoding"):
        band = record.get("potential_speedup", {}).get(leg, {})
        observed_range = band.get("observed_source_range") or band.get("range")
        if observed_range and observed_range != "1.0x-1.0x":
            parts.append(
                "Observed %s band %s (%s, claim_status=%s)."
                % (leg, observed_range, band.get("provenance", "unknown"),
                   band.get("claim_status", "n/a"))
            )
    if record.get("claim_boundary"):
        parts.append("Claim boundary: %s" % record["claim_boundary"])
    return (" " + " ".join(parts)) if parts else ""


# The only v1 record that pairs a numeric measurement with one unambiguous model
# identity. Everything else in model_outcomes.json is a route-level band across
# several models, which cannot anchor a reified hyperedge.
MEASURED_OUTCOMES = {
    "qwen25-05b-instruct-local-worked-port": {
        "result_id": "result:v1-qwen25-05b-bf16-cast-walltime",
        "mechanism": "mechanism:bf16-weight-cast",
        "model": "model:qwen2.5-0.5b-instruct",
        "hardware": "hardware:m4-pro",
        "workload": "workload:six-token-greedy-decode",
        "metric": "wall-time-speedup",
        "delta_bp_ci": [3037, 3037],
        "verdict": "improved",
        "baseline_id": "local F32 port, schema-2 receipt, 2026-07-10",
    }
}


def _measured_result(name, record, observed):
    spec = MEASURED_OUTCOMES.get(name)
    if spec is None:
        return None
    band = record.get("potential_speedup", {}).get("overall", {})
    holds = " ".join(band.get("hold_reasons", []))
    result = node(
        spec["result_id"],
        "applied_result",
        "BF16 weight cast on a local Qwen2.5-0.5B-Instruct port",
        "held-observation",
        "low",
        "local_measured",
        record.get("summary", "")
        + " "
        + band.get("basis", ""),
        [
            ASSET_LOCATOR % "model_outcomes.json (%s)" % name,
            ASSET_LOCATOR % "examples/worked-port-qwen2.5-0.5b-instruct/",
            ASSET_LOCATOR % "benchmarks/attestations/qwen2.5-0.5b-port-bf16/",
        ],
        ["migrated-v1", "local-observation", "held"],
        observed,
        effect={
            "metric": spec["metric"],
            "verdict": spec["verdict"],
            "delta_bp_ci": list(spec["delta_bp_ci"]),
            "baseline_id": spec["baseline_id"],
            "sample_note": (
                "Single local wall-time observation recomputed as a 1.3037288406x inverse"
                " ratio from schema-2 receipts; the owning asset holds the claim. "
                + holds
            )[:1000],
        },
    )
    anchors = [
        edge(spec["result_id"], spec["mechanism"], "instantiates", "high",
             "The measured change was the BF16 weight cast."),
        edge(spec["result_id"], spec["model"], "applied_on", "high",
             "Measured on the repository's own Qwen2.5-0.5B-Instruct worked port."),
        edge(spec["result_id"], spec["hardware"], "measured_on", "high",
             "Receipts record one Apple M4 Pro host."),
        edge(spec["result_id"], spec["workload"], "under_workload", "high",
             "Separate-process load plus six greedy tokens."),
    ]
    return result, anchors


def _reference_id(origin):
    if origin["kind"] == "source":
        return "external:" + slug(origin["id"].split(":", 1)[1])
    paper = origin.get("paper_id")
    if paper:
        return "external:candidate-arxiv-" + slug(paper)
    return "external:candidate-" + slug(origin.get("label", origin["id"]), limit=90)


def _reference_node(origin, source_depth, observed):
    if origin["kind"] == "source":
        depth = source_depth.get(origin["id"].split(":", 1)[1], "indexed")
        kind = origin.get("source_kind", "paper")
        provenance = (
            "code_verified"
            if depth == "synthesized" and kind in CODE_BEARING_KINDS
            else "research_inference"
        )
        evidence = [origin["locator"]]
        immutable = origin.get("immutable_locator")
        if immutable:
            evidence.append(immutable)
        return node(
            _reference_id(origin),
            "external_reference",
            origin["label"],
            depth,
            REVIEW_DEPTH_CONFIDENCE.get(depth, "low"),
            provenance,
            "%s source owned by %s. Review depth %s: %s"
            % (
                kind,
                origin.get("owner", "unknown"),
                depth,
                {
                    "synthesized": "directly informed a rule, runbook, or registry decision",
                    "screened": "relevance and limits reviewed",
                    "indexed": "catalogued for future review; do not represent as fully reviewed",
                }.get(depth, "unknown"),
            ),
            evidence,
            ["migrated-v1", "review-depth/" + depth, "source-kind/" + kind]
            + ["topic/" + value for value in origin.get("topics", [])],
            observed,
        )
    evidence = [origin["locator"]]
    immutable = origin.get("immutable_locator")
    if immutable:
        evidence.append(immutable)
    return node(
        _reference_id(origin),
        "external_reference",
        origin["label"],
        "candidate-unreviewed",
        "low",
        "research_inference",
        "Automatically collected research candidate awaiting review. It has not"
        " informed any rule, runbook, or registry decision and must not be cited"
        " as if it had. Discovery query: %s" % origin.get("query", "n/a"),
        evidence,
        ["migrated-v1", "review-depth/candidate",
         "source-kind/" + origin.get("candidate_kind", "paper")]
        + ["topic/" + value for value in origin.get("topics", [])],
        str(origin.get("updated") or origin.get("published") or observed)[:10],
    )


def _dedupe(nodes):
    seen = {}
    for item in nodes:
        seen[item["id"]] = item
    return [seen[key] for key in sorted(seen)]


def _dedupe_edges(edges):
    seen = {}
    for item in edges:
        seen[(item["from"], item["relation"], item["to"])] = item
    return [seen[key] for key in sorted(seen)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    assets = load_assets()
    shards, observed = build(assets)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_nodes = total_edges = 0
    for name in sorted(shards):
        nodes, edges = shards[name]
        document = {
            "schema_version": gl.SCHEMA_VERSION,
            "graph_id": "mlx-porting-skill/migrated/" + name[: -len(".json")],
            "generated_at": observed,
            "nodes": sorted(nodes, key=lambda item: item["id"]),
            "edges": sorted(edges, key=lambda item: (item["from"], item["relation"], item["to"])),
        }
        gl.write_canonical(out_dir / name, document)
        total_nodes += len(nodes)
        total_edges += len(edges)
        print("  migrated/%-20s nodes=%-4d edges=%-4d" % (name, len(nodes), len(edges)))
    print("migrated nodes=%d edges=%d" % (total_nodes, total_edges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
