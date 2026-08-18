# The evidence graph: the machine layer

`SKILL.md`, the family runbooks, and the registries under `assets/` are the
human layer. They tell you what to do. The evidence graph under `graph/` is the
machine layer: the same knowledge as a strict, validated graph that an agent
can query, that accumulates across models, hardware, and contributors, and that
carries the measurement provenance of every claim.

Read this before citing a graph-derived number in an engineering report.

- Contract: `graph/schema/evidence-graph.schema.json` (owned upstream by the
  auto-mlx tool repository).
- Data: `graph/shards/**.json`, merged into `graph/compiled/evidence-graph.json`.
- Human summary: `graph/MECHANISM_INDEX.md` (generated; never hand-edit).
- Storage rules and invariants: `graph/README.md`.
- Contributing a measurement: `graph/docs/evidence-packs.md`.

## What is in it

Eleven node kinds. The four that matter for retrieval:

- **`model`** — a specific model you might port or optimize.
- **`trait`** — a structural or deployment property a model has. Traits are
  the join key. `trait:gated-delta-net`, `trait:affine-q4-group-64`,
  `trait:long-context`, `trait:dense-decoder-transformer`.
- **`mechanism`** — one optimization, with an `exactness_class` saying whether
  it is exact by construction, needs a parity proof, or is legitimately
  approximate.
- **`applied_result`** — one measurement: this mechanism, on this model, on
  this hardware, under this workload, produced this effect.

An `applied_result` is a reified hyperedge. The node holds the `effect`; four
typed edges hold the binding:

```
result:qwen38-e016-gh2-grouped-sdpa
  --instantiates--> mechanism:grouped-sdpa-kv-reuse
  --applied_on----> model:qwen3.8-27b-mtp
  --measured_on---> hardware:yukon-mlxfast-runner
  --under_workload-> workload:qwen-mtp-ranked-decode
  effect: {metric: ranked-decode-speedup, verdict: regressed, delta_bp_ci: [-257, -257]}
```

That shape is what makes the knowledge compound. The same mechanism measured on
a different model or a different chip is a different `applied_result` pointing
at the same `mechanism` node, so "does this ever work, and where" becomes a
graph traversal instead of a literature review.

## How to query it

The compiled graph is a single JSON file, so stdlib is enough.

```python
import json
graph = json.load(open("mlx-model-porting/graph/compiled/evidence-graph.json"))
nodes = {n["id"]: n for n in graph["nodes"]}
edges = graph["edges"]
```

**By trait — "my model is a hybrid recurrent decoder, what has been tried?"**
Find the traits your model exhibits, then find mechanisms that `applies_to`
those traits.

```python
traits = {e["to"] for e in edges
          if e["from"] == "model:qwen3.8-27b-mtp" and e["relation"] == "exhibits"}
candidates = {e["from"] for e in edges
              if e["relation"] == "applies_to" and e["to"] in traits}
```

**By mechanism — "has anyone measured this, and what happened?"**
Collect the `applied_result` nodes that `instantiates` it, then read each
result's `effect` together with its model, hardware, and workload anchors. A
mechanism with results on one model and one host tells you almost nothing about
your host.

**By model — "what is known about this exact model?"**
Follow `applied_on` edges into the model, and `exhibits` edges out of it.

**For what is unknown.** `hypothesis` nodes and mechanisms with no
`applied_result` are the frontier. `mechanism:entropy-coded-weight-stream` and
`mechanism:two-bit-compact-draft-readout` have no measurement in either
direction because every attempt failed a submission gate before it was scored.
Absence of a result is an open question, never a negative.

## Reading provenance and effects

Provenance is the same discipline the source registry already uses, applied to
measurements. It is never upgraded by assertion:

| Provenance | Means |
| --- | --- |
| `official_verified` | scored by an authoritative external evaluator |
| `replicated` | independently reproduced |
| `local_measured` | measured on a host this repository controls |
| `code_verified` | established by reading published code or documentation |
| `contributor_claim` | one contributor measured it once |
| `author_claim` | a third party reported it |
| `research_inference` / `transfer_inference` | reasoned, not measured |

Effects are integer basis points against a named baseline: `100 bp = 1%`,
`-257` is `-2.57%`. `delta_bp_ci` is a point `[x, x]` for a single run and a
span for repeated runs; `sample_note` always says which. `metric_direction`
says which way is better, and when it is declared an `improved` or `regressed`
verdict must have an interval strictly on that side of zero.

`verdict` is conservative on purpose. In the Qwen 3.8 campaign, three official
runs of *byte-identical* candidate content scored 130 basis points apart
(`finding:official-rerun-variation`). Any result whose delta falls inside that
spread is `inconclusive` even where the owning campaign retired the candidate.
Node `status` carries the human decision; `effect.verdict` carries what the
number supports. Two mechanisms in that campaign's promoted stack were promoted
on deltas of `+3` and `+9` basis points — real promotions, unresolved effects
(`finding:promotion-is-not-effect-resolution`).

## Citing it

Graph-derived text must cite node ids, the same way a `synthesized` source must
name the rule it informed. Write:

> Grouped SDPA with shared K/V reads regressed 2.57% on the ranked path
> (`result:qwen38-e016-gh2-grouped-sdpa`, `official_verified`), and the family
> is retired (`mechanism:grouped-sdpa-kv-reuse`).

Not:

> Grouped SDPA is slower on Apple Silicon.

The second sentence is not in the graph. The graph says one grouped-SDPA
kernel, at one group width, on one model, on one undisclosed evaluation host,
under one workload, measured negative twice.

**The graph never promotes and never substitutes for a measurement.** It tells
you what has already been tried, on what, with what result, and how much that
result is worth. Deciding to spend a measurement is still your call, and
`constraint:official-evaluator-is-the-only-authority` is in there because the
campaign that produced most of this data learned it the expensive way.
