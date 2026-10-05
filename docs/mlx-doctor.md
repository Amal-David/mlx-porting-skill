# MLX Doctor

MLX Doctor is the single static entry point for a local model, an existing MLX
project, or the current Python environment. It uses the canonical model/project
inspectors and architecture registry. It does not add a competing router, import
ML frameworks, execute target code, download weights, convert a model, or grant
permission to skip a parity gate.

## Run from a checkout

```bash
./mlx-doctor /path/to/local-model --format markdown
./mlx-doctor /path/to/project --kind project --model /path/to/local-model
./mlx-doctor --backend mlx-lm --require-runtime
```

The portable installed-skill entry point is the Python script:

```bash
python3 mlx-model-porting/scripts/mlx_doctor.py /path/to/local-model

python3 mlx-model-porting/scripts/mlx_doctor.py /path/to/local-model \
  --output doctor.json --markdown DOCTOR.md

python3 mlx-model-porting/scripts/mlx_doctor.py /path/to/local-model \
  --backend mlx-lm --memory-budget-gib 16 --require-runtime
```

The root launcher is a checkout convenience, not an executable installed into
PATH. A copied skill includes `scripts/mlx_doctor.py`. No extra Python packages
are required for Doctor itself.

No input path means environment metadata only. Automatic input detection chooses
model mode for a configuration/checkpoint and project mode for ordinary code
projects; `--kind model` or `--kind project` removes that ambiguity. All input
paths must already exist locally. A Hub ID is not downloaded. `--model` supplies
an optional checkpoint alongside a project, and any blockers in that nested
model inspection remain blockers in the combined diagnosis.

## Read the result

The JSON report includes its schema version, status, host/package metadata,
canonical architecture families and runbooks, backend hints, inspection blockers,
runtime prerequisites, declared tensor storage, and next-step command arguments.
The Markdown format presents the same boundaries for a person.

| Finding | Meaning |
| --- | --- |
| `static-review-complete` | The canonical static inspection has no reported blocker and the selected host/package metadata prerequisites are present. No model execution or parity ran. |
| `blocked` | The canonical inspector found an integrity, provenance, license, safety, inventory, or routing hold. Preserve and resolve every blocker. |
| `runtime-prerequisites-missing` | The host or selected backend packages are absent/inappropriate, or declared tensor storage already exceeds the supplied budget. Static inspection may still be useful. |
| `host-metadata-only` | No model/project input was supplied. Only host and installed-package metadata were checked. |
| `inspection-error` | An input, argument, output, subprocess, timeout, or parsing failure prevented a trustworthy diagnosis. |

Exit status is **0** for a completed static diagnosis with no inspection blocker,
**1** for a blocked inspection, and **2** for an error. Missing runtime
prerequisites change exit status to **1 only with `--require-runtime`**. Without
that switch, a Linux machine or an environment without MLX can still inspect a
model and receive a useful report. `--require-runtime` checks metadata only; it
never changes `metal_execution: not-probed` into a runtime claim.

`family-available-profile-unverified` means that the selected family has an
existing scaffold generator. It does **not** mean that the exact configuration
will pass that generator, all weights are mapped, or the generated implementation
has source parity. Hybrid routes can require manual composition even when their
individual families have generators. Other families remain runbook-guided.

Backend names are registry hints, not a list of tested native models. The source
license report retains the inspector's evidence and review flags; Doctor does
not establish legal compatibility. Project files named parity or benchmark are
static evidence surfaces, not proof that the measurements are correct.

## Memory and next steps

`declared_tensor_storage_bytes` comes from the existing static tensor inventory.
It excludes cache, activations, preprocessing, runtime allocations, and other
processes. Quantization layouts and tied tensors retain the inspector's existing
limitations. Doctor never estimates total runtime memory or declares that a
model fits. A budget smaller than declared tensor storage is a clear prerequisite
failure; a larger budget is **not** proof of fit.

Every suggested action has an ID, an argument array, a displayed command,
required placeholders, an explicit validation gate, and `executed: false`.
Blocked inputs receive remediation and reinspection/planning steps, not an
instruction to force a scaffold or optimization. The existing planner
independently rereads the artifact bytes and produces a remediation-only plan
while blocked.

Absolute paths are omitted by default. Command templates therefore name
`<PYTHON>`, `<SKILL_SCRIPTS>`, and `<MODEL_PATH>` or `<PROJECT_PATH>`. Supply those
values before running a command. `--include-local-paths` makes the report
machine-specific and uses the inspecting interpreter and exact local paths;
consider that disclosure before sharing the report.

## Bounds and output safety

The canonical inspection runs under a bounded subprocess timeout (default 120
seconds, configurable from 1 to 300). File inventory is capped (default 2000,
configurable from 1 to 10000). A truncated inventory is not a clean report.
Doctor uses the inspector's artifact hashing and format limits, so large local
models can take longer than small metadata-only examples.

Report outputs must be **new files outside the inspected model/project**. Their
parent directory must already exist. Existing files, dangling symlinks, linked
parents, duplicate JSON/Markdown destinations, and output paths containing `..`
are refused. Each file is written privately and published create-only; the pair
of optional JSON and Markdown files is not a single transaction, so a second
publication failure can leave the first complete report. No partial report file
is intentionally published, and no existing report is overwritten.

On macOS, use a real resolved output directory rather than a symlinked prefix.
On hosts without POSIX no-follow descriptor primitives, use JSON/Markdown on
stdout instead of file output flags. Diagnostic and report strings pass through
the existing secret redactor. No network flags or automatic dependency install
commands are provided.

## Verification boundary

A useful next sequence remains: static inspection, configuration gate, explicit
weight map, source oracle, staged parity, workload quality, then benchmarking.
Doctor only makes the first step easier to discover. It does not depend on the
companion inference preview being merged or installed, and a successful Doctor
report is not permission to activate an unverified runtime profile.

## Serving research candidates

Add `--workload coding-agent` (or another supported workload) for an alphabetical
research shortlist. Add `--include-comparators` for non-MLX/opaque comparison
baselines; default output explicitly lists its exclusions. Inspection blockers
and runtime prerequisites remain distinct and neither can be cleared by a
source-declared engine feature. Installed skills can run advisor queries and
`--validate`; report regeneration/check modes require the source checkout.
