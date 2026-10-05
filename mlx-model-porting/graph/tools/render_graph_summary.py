#!/usr/bin/env python3
"""Render the compiled evidence graph as a human-readable mechanism index.

    python3 mlx-model-porting/graph/tools/render_graph_summary.py

Writes `mlx-model-porting/graph/MECHANISM_INDEX.md`. The output is generated:
never hand-edit it, edit a shard and regenerate. Every line cites the node ids
it came from so a reader can go back to the evidence, which is the same
provenance discipline the authored assets already use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _graphlib as gl  # noqa: E402

OUT_PATH = gl.REPO_GRAPH_ROOT / "MECHANISM_INDEX.md"

VERDICT_ORDER = {"improved": 0, "inconclusive": 1, "regressed": 2}

# Strongest evidence first, so the reader meets officially verified mechanisms
# before third-party claims rather than whichever number is largest.
PROVENANCE_ORDER = {
    "official_verified": 0,
    "replicated": 1,
    "local_measured": 2,
    "code_verified": 3,
    "contributor_claim": 4,
    "author_claim": 5,
    "research_inference": 6,
    "transfer_inference": 7,
}


def _pct(basis_points):
    sign = "+" if basis_points >= 0 else "-"
    value = abs(basis_points)
    return "%s%d.%02d%%" % (sign, value // 100, value % 100)


def _index(graph):
    nodes = {item["id"]: item for item in graph["nodes"]}
    out_edges = {}
    in_edges = {}
    for edge in graph["edges"]:
        out_edges.setdefault(edge["from"], []).append(edge)
        in_edges.setdefault(edge["to"], []).append(edge)
    return nodes, out_edges, in_edges


def _anchor(out_edges, result_id, relation):
    for edge in out_edges.get(result_id, []):
        if edge["relation"] == relation:
            return edge["to"]
    return None


def render(graph):
    nodes, out_edges, in_edges = _index(graph)
    lines = []
    add = lines.append

    add("# Mechanism index")
    add("")
    add(
        "Generated from `graph/compiled/evidence-graph.json` by "
        "`graph/tools/render_graph_summary.py`. Do not hand-edit; change a shard under "
        "`graph/shards/` and regenerate."
    )
    add("")
    add(
        "Every effect below is an integer basis-point delta against the specific baseline "
        "the run was measured on (100 bp = 1 percent). A verdict of `inconclusive` means the "
        "measured delta did not exceed the evaluator's own observed same-content rerun "
        "spread, so the run does not establish a sign in either direction. Node ids are "
        "given so every claim can be traced to its evidence."
    )
    add("")

    results_by_mechanism = {}
    for node in graph["nodes"]:
        if node["kind"] != "applied_result":
            continue
        for edge in out_edges.get(node["id"], []):
            if edge["relation"] == "instantiates":
                results_by_mechanism.setdefault(edge["to"], []).append(node)

    counts = {}
    for node in graph["nodes"]:
        counts[node["kind"]] = counts.get(node["kind"], 0) + 1
    add("## Corpus")
    add("")
    add("| Node kind | Count |")
    add("| --- | ---: |")
    for kind in sorted(counts):
        add("| %s | %d |" % (kind, counts[kind]))
    add("| **edges** | %d |" % len(graph["edges"]))
    add("")

    # ---- measured mechanisms -------------------------------------------
    add("## Mechanisms with measured effects")
    add("")
    measured = sorted(
        results_by_mechanism,
        key=lambda mid: (
            PROVENANCE_ORDER.get(nodes[mid]["provenance"], 9) if mid in nodes else 9,
            -max(item["effect"]["delta_bp_ci"][1] for item in results_by_mechanism[mid]),
            mid,
        ),
    )
    for mechanism_id in measured:
        mechanism = nodes.get(mechanism_id)
        if mechanism is None:
            continue
        exactness = mechanism.get("exactness_class", "unclassified")
        add("### `%s`" % mechanism_id)
        add("")
        add(
            "**%s** — status `%s`, exactness `%s`, provenance `%s`."
            % (mechanism["title"], mechanism["status"], exactness, mechanism["provenance"])
        )
        add("")
        add(mechanism["summary"])
        add("")
        add("| Effect | Verdict | Result | Model | Hardware | Workload |")
        add("| ---: | --- | --- | --- | --- | --- |")
        for result in sorted(
            results_by_mechanism[mechanism_id],
            key=lambda item: (
                VERDICT_ORDER.get(item["effect"]["verdict"], 3),
                -item["effect"]["delta_bp_ci"][1],
                item["id"],
            ),
        ):
            effect = result["effect"]
            low, high = effect["delta_bp_ci"]
            span = _pct(low) if low == high else "%s..%s" % (_pct(low), _pct(high))
            add(
                "| %s | %s | `%s` | `%s` | `%s` | `%s` |"
                % (
                    span,
                    effect["verdict"],
                    result["id"],
                    _anchor(out_edges, result["id"], "applied_on") or "-",
                    _anchor(out_edges, result["id"], "measured_on") or "-",
                    _anchor(out_edges, result["id"], "under_workload") or "-",
                )
            )
        add("")

    # ---- unmeasured mechanisms -----------------------------------------
    unmeasured = sorted(
        node["id"]
        for node in graph["nodes"]
        if node["kind"] == "mechanism" and node["id"] not in results_by_mechanism
    )
    add("## Mechanisms with no measured effect in this graph")
    add("")
    add(
        "These carry no `applied_result`. Some were rejected on structural grounds before "
        "measurement, some have never been scored because a submission gate failed first, "
        "and some are registry entries whose evidence is documentation rather than a run. "
        "Absence of a measurement here is an open question, never a negative result."
    )
    add("")
    add("| Mechanism | Status | Provenance |")
    add("| --- | --- | --- |")
    for mechanism_id in unmeasured:
        mechanism = nodes[mechanism_id]
        add(
            "| `%s` | %s | %s |"
            % (mechanism_id, mechanism["status"], mechanism["provenance"])
        )
    add("")

    # ---- traits ---------------------------------------------------------
    add("## Models and traits")
    add("")
    for node in sorted(
        (item for item in graph["nodes"] if item["kind"] == "model"),
        key=lambda item: item["id"],
    ):
        traits = sorted(
            edge["to"] for edge in out_edges.get(node["id"], []) if edge["relation"] == "exhibits"
        )
        add("- `%s` — %s" % (node["id"], node["title"]))
        if traits:
            add("  - traits: %s" % ", ".join("`%s`" % item for item in traits))
        results = sorted(
            edge["from"]
            for edge in in_edges.get(node["id"], [])
            if edge["relation"] == "applied_on"
        )
        if results:
            add("  - measured results: %d" % len(results))
    add("")

    # ---- transferable findings and constraints --------------------------
    add("## Generalizable findings and constraints")
    add("")
    add(
        "These are the parts of the corpus that are expected to survive a change of model, "
        "hardware, or workload. Each cites its own node id."
    )
    add("")
    for kind, heading in (("constraint", "Constraints"), ("finding", "Findings")):
        selected = sorted(
            (
                item
                for item in graph["nodes"]
                if item["kind"] == kind and "generalizable" in item["tags"]
            ),
            key=lambda item: item["id"],
        )
        if not selected:
            continue
        add("### %s" % heading)
        add("")
        for item in selected:
            add("- **%s** (`%s`, %s)" % (item["title"], item["id"], item["provenance"]))
            add("  %s" % item["summary"])
        add("")

    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", default=str(gl.COMPILED_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero when the checked-in output is stale",
    )
    args = parser.parse_args(argv)

    graph = gl.load_strict(args.compiled)
    rendered = render(graph)
    out = Path(args.out)
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != rendered:
            print("FAIL mechanism index is stale at %s" % out)
            return 1
        print("OK mechanism index matches the compiled graph")
        return 0
    out.write_text(rendered, encoding="utf-8", newline="\n")
    print("wrote %s (%d lines)" % (out, rendered.count("\n")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
