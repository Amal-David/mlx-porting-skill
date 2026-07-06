# Hypothesis-Led Learning

Use this loop when a user asks for a large model-specific optimization search:
100 hypotheses, another 2x or 5x target, a public-port comparison, a custom
kernel hunt, or a request to adapt a learn-flow from another agent system.

The output is not a direct skill rewrite. Route sources through:

1. evidence graph;
2. hypothesis backlog;
3. bounded experiments;
4. promotion ledger;
5. skill or runbook update only after validation.

## Learn-Flow Adaptation

Hermes-style `/learn` works because it turns an open-ended request into a normal
agent turn that uses existing tools, gathers sources, preserves requirements,
and writes durable skill knowledge. The MLX version should keep that simplicity,
but add an MLX-specific review gate before changing the skill:

- gather local source, benchmark, official-doc, paper, blog, and community leads;
- encode them as graph nodes and edges before drafting advice;
- keep speculative ideas in backlog until an MLX path is reproducible;
- promote only the findings with source provenance, correctness gate, benchmark
  metadata, and rollback condition.

Do not average contradictory sources. If a blog claims speedup, a paper changes
architecture, and local profiling points somewhere else, preserve each as a
separate node and let experiments decide.

## Graph Shape

Create a `hypothesis_graph.json` when the work is more than a short answer. Use
these node types:

- `model`: exact model, repo, checkpoint, dtype, and public baseline.
- `bottleneck`: measured or suspected hotspot.
- `evidence`: source path, URL, local artifact, paper, or benchmark.
- `hypothesis`: one proposed speed or quality improvement.
- `experiment`: a bounded test command or implementation slice.
- `quality_gate`: parity, perceptual, task metric, or regression threshold.
- `benchmark_result`: measured latency, memory, throughput, and environment.
- `decision`: accepted, rejected, held, or needs-validation.
- `promotion_candidate`: finding ready for skill or runbook review.
- `rejected_result`: idea that failed, with reason and rollback lesson.

Every hypothesis node needs:

- stable id;
- evidence class;
- source locator;
- affected surface;
- expected bottleneck;
- validation gate;
- rollback condition;
- status.

Use these edge types: `supports`, `contradicts`, `targets`, `depends_on`,
`validates`, `rejects`, `promotes`, and `supersedes`.

## Hypothesis Depth Bar

Each serious hypothesis should answer:

- why it is plausible;
- what evidence class supports it;
- which exact bottleneck it targets;
- how to test it without executing unreviewed remote model code;
- what quality gate would catch degradation;
- expected payoff band;
- main risk;
- rollback condition;
- current status.

Use these statuses:

- `validated_locally`: benchmarked against a named baseline with quality gate.
- `benchmark_required`: plausible, but no local measurement yet.
- `experimental_approach`: requires model change, retraining, distillation,
  pruning, gating, quantization, or custom kernel work.
- `validation_backlog`: enough source signal to revisit later.
- `rejected`: tested or ruled out, with reason.
- `promotion_candidate`: ready for human review as a skill/runbook edit.

## Evidence Policy

Treat evidence classes differently:

- local benchmark artifact: strongest evidence for speed claims;
- official docs or primary source: strongest evidence for API/runtime support;
- pinned public repository: implementation evidence, not proof of superiority;
- paper: algorithm candidate until reproduced in MLX;
- technical blog: optimization lead until validated locally;
- community discussion: backlog lead only.

If a technique changes model behavior, do not describe it as a quality-neutral
speedup. Quantization, sparse trained gates, distillation, caching approximations,
or step reduction need an explicit quality gate and a rollback path.

## 5x Stack Discipline

Separate lossless runtime work from lossy or model-changing work:

- lossless: graph cleanup, lazy-eval boundaries, compile/fusion, batching,
  memory layout, stream use, native MLX fast paths, and validated custom kernels;
- conditionally lossy: 4-bit or FP4 quantization, attention-cache approximation,
  sparse gates, layer skipping, distillation, fewer diffusion/flow steps, or
  decoder/vocoder approximation.

A 2x target may come from architecture-preserving runtime work. A 5x target
usually needs a stack of runtime, kernel, schedule, and model-changing ideas.
Say that clearly before promising real-time inference.

## Artifacts

For a large optimization search, write:

- `HYPOTHESES.md`: human-readable backlog grouped by bottleneck.
- `hypothesis_graph.json`: nodes, edges, statuses, and source locators.
- `EXPERIMENT_<id>.md` or `.json`: command, environment, metric, and gate.
- `PROMOTION_LEDGER.md`: what can move into the skill, what is held, and why.

Recommended first pass:

1. map the public baseline and local profile;
2. write 50-100 hypotheses with the depth bar above;
3. choose 3-6 priority experiments;
4. run only architecture-preserving experiments first;
5. put lossy or trained-gate ideas behind explicit user opt-in.

## Promotion Rule

Do not update `SKILL.md`, a runbook, or optimization guidance from the hypothesis
backlog alone. Promotion requires:

- source provenance;
- reproducible MLX implementation path;
- correctness or quality check;
- benchmark metadata: hardware, OS, MLX version, dtype, workload, baseline;
- rollback condition;
- affected asset or runbook;
- explicit caveat if the technique is experimental or quality-affecting.

