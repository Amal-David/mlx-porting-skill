#!/usr/bin/env python3
"""Validate the v2 evidence-graph shards and the compiled graph.

    python3 mlx-model-porting/graph/tools/validate_graph.py

Standard library only. Exits non-zero on the first failing check set and
prints every problem it found, not just the first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _graphlib as gl  # noqa: E402


def _counts(doc):
    kinds = {}
    for node in doc.get("nodes", []):
        kinds[node.get("kind")] = kinds.get(node.get("kind"), 0) + 1
    return kinds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default=str(gl.SHARDS_DIR), help="shard directory")
    parser.add_argument(
        "--compiled",
        default=str(gl.COMPILED_PATH),
        help="compiled graph to check against the shards",
    )
    parser.add_argument("--skip-compiled", action="store_true", help="validate shards only")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = parser.parse_args(argv)

    problems = []
    drift = gl.check_schema_agreement()
    if drift:
        problems.extend(drift)

    try:
        documents = gl.load_shards(args.shards)
    except gl.GraphError as exc:
        print("FAIL strict-JSON: %s" % exc)
        return 1

    if not documents:
        print("FAIL no shards found under %s" % args.shards)
        return 1

    total_nodes = 0
    total_edges = 0
    for label, doc in documents:
        problems.extend(gl.validate_document(doc, label))
        nodes = len(doc.get("nodes", []))
        edges = len(doc.get("edges", []))
        total_nodes += nodes
        total_edges += edges
        if not args.quiet:
            kinds = ", ".join("%s=%d" % item for item in sorted(_counts(doc).items()))
            print("  %-56s nodes=%-4d edges=%-4d %s" % (label, nodes, edges, kinds))

    problems.extend(gl.validate_corpus(documents))

    if not args.skip_compiled:
        compiled_path = Path(args.compiled)
        if not compiled_path.exists():
            problems.append(
                "compiled graph missing at %s; run tools/compile_graph.py" % compiled_path
            )
        else:
            try:
                compiled = gl.load_strict(compiled_path)
            except gl.GraphError as exc:
                problems.append(str(exc))
            else:
                problems.extend(gl.validate_document(compiled, "compiled/evidence-graph.json"))
                problems.extend(gl.validate_corpus([("compiled", compiled)]))
                expected = gl.merge(documents, compiled.get("graph_id", ""))
                if gl.dump_canonical(expected) != gl.dump_canonical(compiled):
                    problems.append(
                        "compiled graph is stale; re-run tools/compile_graph.py"
                    )

    print(
        "shards=%d nodes=%d edges=%d" % (len(documents), total_nodes, total_edges)
    )
    if problems:
        print("FAIL %d problem(s):" % len(problems))
        for problem in problems:
            print("  - %s" % problem)
        return 1
    print("OK evidence graph is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
