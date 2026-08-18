# Evidence packs

An evidence pack is how a measurement gets into the graph from outside. It is
one directory, one pull request, and one gate.

```
packs/<short-slug>/
  manifest.json      who measured it, on what host, with which runtimes, when
  graph-delta.json   a schema-valid v2 graph document: only new nodes and edges
  receipts/          optional raw measurement files, pinned by sha256
```

Validate before opening the PR:

```bash
python3 mlx-model-porting/graph/tools/validate_pack.py mlx-model-porting/graph/packs/<slug>
```

`packs/example-kvq8-m3max/` is a complete working example.

## The lifecycle

```
pack PR  ->  CI validation  ->  merge as a claim  ->  replication upgrades it
```

**Pack PR.** You add a directory. You do not edit any existing shard, and you
do not edit the compiled graph.

**CI validation.** `validate_pack.py` runs the full graph validator over your
delta plus three gates that only apply to incoming contributions (below). A
pack that fails any of them is not a judgement about your measurement; it is a
statement that the graph cannot store it in that form yet.

**Merge as a claim.** A merged pack enters the graph at `contributor_claim`.
That is a real, queryable, citable status. It is not a lesser kind of
knowledge; it is an accurate one. Somebody reading the graph can see exactly
one host measured this, once.

**Replication upgrades it.** When a second contributor measures the same
mechanism on the same or different hardware, they submit their own pack. A
maintainer may then raise the mechanism's provenance to `replicated` and link
the results. Provenance moves up through evidence, never through assertion.

## The three contribution gates

**1. Scrub.** No credentials of any kind, no absolute local paths, no bare home
directories, anywhere in the pack including receipts. Your `contributor` handle
must be a plain handle, not an email address and not a real name. If you want
to cite something local, describe it (`"paired A/B receipt, five runs"`) rather
than pasting a path.

**2. Provenance ceiling.** Any node carrying an `effect` must be
`contributor_claim` (or weaker). A pack may never assert `official_verified`,
`local_measured`, or `replicated`; those are maintainer-set and mean something
specific about who verified what. A pack node that only records what published
code says, with no measurement, may still be `research_inference`.

**3. No redefinition.** A pack adds nodes. It may not redefine a node that
already exists in the compiled graph. Your edges *may* reference existing
ids freely — that is the point — but changing an existing node is a graph edit
and gets reviewed as one.

## What makes a pack worth merging

The graph is a decision aid for someone about to spend hours on an
optimization. What helps them is knowing what happened, under exactly which
conditions, and what you did not check.

- **Name the workload.** A `workload` node with an `identity` block
  (prompt tokens, decode tokens, concurrency, sampling) is the difference
  between a reusable result and a number.
- **Name the hardware.** Chip, GPU cores, memory, OS build. If you did not
  instrument thermals, say so in the manifest; it is a real limit on the
  measurement, not an embarrassment.
- **Use the interval.** If you ran five A/B pairs, `delta_bp_ci` should span
  the observed pairs and `sample_note` should say how they were collected. A
  point interval `[x, x]` is for a single run and should say that too.
- **Declare `metric_direction`.** `higher_is_better` or `lower_is_better` on the
  effect. It makes your verdict machine-checkable: with it declared, `improved`
  or `regressed` requires the whole interval to sit strictly on the claimed side
  of zero. If your interval crosses zero, the honest verdict is `inconclusive`,
  and that is a perfectly good result to contribute.
- **Say what you did not gate.** The example pack states plainly that no
  output-quality gate was run for a quantized KV cache. That sentence is worth
  more to the next reader than the throughput number.
- **Negative results are first-class.** A `regressed` verdict on an obvious-
  looking idea saves the next person the whole experiment. Most of the highest-
  value nodes in this graph are negative.

## Effects are integer basis points

No floats, anywhere. `100 bp = 1%`. A 6.4% improvement is `640`. A 2.57%
regression is `-257`. This is not stylistic: two independent tools have to
agree on every stored number byte for byte, and floats do not survive that.

## Manifest reference

```json
{
  "pack_version": 1,
  "contributor": "your-handle",
  "submitted_at": "2026-08-14",
  "summary": "one sentence: what was compared, on what, how many runs",
  "hardware": {
    "node_id": "hardware:m3-max",
    "chip": "Apple M3 Max",
    "gpu_cores": 40,
    "memory_gb": 64,
    "os_build": "macOS 26.2",
    "thermal_instrumented": false
  },
  "runtime": {
    "os": "macOS 26.2",
    "packages": {"mlx": "0.30.4", "mlx-lm": "0.31.2", "python": "3.12.7"}
  },
  "receipts": [
    {"path": "receipts/kvq8-vs-fp16.json", "role": "paired-ab", "sha256": "<64 hex>"}
  ],
  "notes": "anything that limits the measurement"
}
```

`hardware.node_id` either names an existing `hardware:` node or names one your
delta declares. Declared receipt digests are checked against the bytes you
actually ship.
