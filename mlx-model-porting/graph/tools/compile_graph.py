#!/usr/bin/env python3
"""Merge every graph shard into one deterministic compiled evidence graph.

    python3 mlx-model-porting/graph/tools/compile_graph.py

The output is byte-stable for a fixed set of inputs: sorted object keys,
node order by id, edge order by (from, relation, to), LF endings, UTF-8, and
`generated_at` taken as the newest shard `generated_at` rather than the wall
clock. Regenerating without an input change rewrites identical bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _graphlib as gl  # noqa: E402

GRAPH_ID = "mlx-porting-skill/evidence-graph"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default=str(gl.SHARDS_DIR))
    parser.add_argument("--out", default=str(gl.COMPILED_PATH))
    parser.add_argument("--graph-id", default=GRAPH_ID)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero when the checked-in output is stale",
    )
    args = parser.parse_args(argv)

    documents = gl.load_shards(args.shards)
    if not documents:
        print("FAIL no shards found under %s" % args.shards)
        return 1
    compiled = gl.merge(documents, args.graph_id)
    rendered = gl.dump_canonical(compiled)

    out = Path(args.out)
    if args.check:
        if not out.exists():
            print("FAIL compiled graph missing at %s" % out)
            return 1
        if out.read_text(encoding="utf-8") != rendered:
            print("FAIL compiled graph is stale at %s" % out)
            return 1
        print("OK compiled graph matches the shards")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8", newline="\n")
    print(
        "wrote %s from %d shards: nodes=%d edges=%d"
        % (out, len(documents), len(compiled["nodes"]), len(compiled["edges"]))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
