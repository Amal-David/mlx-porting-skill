# Contributor instructions

Treat this repository as an executable engineering standard, not a prompt collection.

- Preserve source provenance and review depth. Never promote an `indexed` source to `synthesized` without recording the affected rule or runbook.
- Prefer official MLX code and documentation over secondary performance claims.
- Do not introduce a technique as “supported” until a reproducible MLX path and validation gate exist.
- Keep `SKILL.md` compact. Put architecture detail in `references/` and structured facts in `assets/`.
- Scripts must be inspectable, non-destructive by default, and must not execute remote model code.
- Daily automation may collect candidates and open a review change. It must not auto-merge research-derived recommendations.
- Every optimization change needs a correctness check, benchmark metadata, and a rollback condition.

<!-- superplan-entry-instructions:start -->
# Optional Superplan Operating Contract

When Superplan is installed, load and follow `superplan-entry` from the first
available location:

- `.codex/skills/superplan-entry/SKILL.md`
- `.superplan/skills/superplan-entry/SKILL.md`
- `${HOME}/.config/superplan/skills/superplan-entry/SKILL.md`
- `${HOME}/.codex/skills/superplan-entry/SKILL.md`

If Superplan is unavailable, continue with the repository-native workflow in
`VALIDATION.md`; optional orchestration must never make a fresh checkout
unusable.

When Superplan is active:
- No implementation before loading and following `superplan-entry`.
- No broad repo exploration before loading and following `superplan-entry`.
- No planning or repo-specific clarification before loading and following `superplan-entry`.
- Keep workflow control internal: do not narrate skill names, routing, or command logs to the user.
- If `.superplan/` exists, treat the Superplan CLI as the execution control plane.
- Prefer workspace harnesses, scripts, and custom workflows when `superplan-entry` routes you there.

Canonical loop when Superplan is active:
1. Run `superplan status --json`.
2. Claim or resume work with `superplan run --json` or `superplan run <task_id> --json`.
3. Continue through the owning Superplan phase instead of improvising a parallel workflow.
4. Use lifecycle commands such as `superplan task runtime block`, `superplan task runtime request-feedback`, and `superplan task review complete`; never hand-edit `.superplan/runtime/`.

Decision guardrails:
- If Superplan readiness is missing, fall back to the repository-native
  workflow unless the user explicitly asked for Superplan itself.
- If work is already shaped, resume the owning execution or review phase instead of routing from scratch.
- If the request is large, ambiguous, or multi-workstream, route before implementing.
<!-- superplan-entry-instructions:end -->

## MLX model porting requests

- For any request about porting, converting, running, quantizing, benchmarking, or optimizing a model for MLX or Apple Silicon (e.g. "port this HF model to my Mac", "run Qwen on Apple Silicon", "make this faster on my M3", "fix NaN in my MLX port"), the domain source of truth is `mlx-model-porting/SKILL.md` - load it and its Trigger map before improvising.
- When available, Superplan governs workflow sequencing; the skill always
  governs technical content.
