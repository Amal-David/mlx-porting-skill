# Runbook: graph message-passing models

## Applies to

GCN, GraphSAGE, GAT, GIN, and MPNN-style graph neural networks that update
node or edge features by repeatedly aggregating messages over an explicit graph.
Route point-cloud sampling, equivariant geometry, protein/chemistry force
fields, and scientific energy models to separate tracks until they have their
own fixtures and symmetry gates.

## Architecture fingerprint

Confirm:

- graph task type: node classification, graph classification, link prediction,
  edge prediction, or embedding extraction;
- node and edge feature schemas, categorical vocabularies, missing-feature
  policy, and feature normalization;
- edge index direction, self-loop policy, duplicate-edge handling, and whether
  the source expects COO, CSR, dense adjacency, or packed batches;
- aggregation type: add, mean, max, attention-weighted, edge-conditioned, or
  learned message MLP;
- normalization: degree normalization, BatchNorm/LayerNorm, residuals, dropout,
  and train/eval behavior;
- readout or pooling contract, label mapping, and whether output is per node,
  per edge, or per graph.

## Source oracle checkpoints

Capture node features after preprocessing, edge index/order, edge features,
self-loop-expanded graph, degree or normalization factors, one layer's messages
before reduction, reduced node updates, post-activation features, pooled graph
feature, logits, and task metric inputs. For packed mini-batches, capture graph
IDs, pointer arrays, and per-graph node counts.

## Weight conversion

- preserve node and edge encoder vocab/order;
- map message, update, attention, and readout linears separately;
- preserve edge-direction convention and whether messages flow source-to-target
  or target-to-source;
- preserve normalization epsilon, affine weights, and dropout/stochastic
  behavior;
- document any dense-adjacency fallback as a tiny-fixture correctness path, not
  scalable support.

## Minimal MLX path

1. Port feature preprocessing and static graph metadata.
2. Implement one scatter/segment/add aggregation against a tiny graph oracle.
3. Port one message-passing layer with fixed node/edge inputs.
4. Port all message-passing layers and readout/head.
5. Add packed-batch handling if the product advertises batched graphs.
6. Save/reload and compare logits, embeddings, and task metrics.

## Parity traps

- edge direction reversed by transposing `edge_index`;
- self-loops added twice or omitted;
- duplicate edges coalesced differently;
- degree normalization uses in-degree rather than out-degree;
- scatter reduction is nondeterministic or loses isolated nodes;
- mean aggregation divides by the wrong count after batching;
- graph-level pooling mixes nodes from different graphs;
- categorical feature vocabularies or padding IDs are reordered;
- dropout or BatchNorm training mode remains active during parity.

## Optimization ladder

1. Establish scatter/segment/reduce parity before profiling.
2. Compile stable message/update/readout regions when graph shapes are bucketed
   or fixed.
3. Use sparse or segment primitives where MLX provides matching semantics.
4. Keep graph construction, sorting, and feature decoding outside hot compiled
   regions unless they are array-pure and shape-stable.
5. Consider dense adjacency only for tiny graphs, debugging, or correctness
   fixtures.
6. Add custom kernels only after a measured scatter/segment hotspot has a
   readable MLX fallback and task-level benchmark.

## Completion gates

- fixed tiny graph passes scatter/segment/reduce parity;
- isolated nodes, duplicate edges, self-loops, and reversed-edge probes pass;
- one message-passing layer and full stack match source checkpoints;
- permutation invariance holds for reordered nodes/edges when the source
  promises it;
- batched graphs keep per-graph boundaries intact;
- task metric parity passes for the intended graph task.
