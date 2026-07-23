# Cross-agent installation

`mlx-model-porting/` follows the shared Agent Skills format: one `SKILL.md` plus optional `scripts/`, `references/`, and `assets/`. Use the same folder across clients; do not fork the technical corpus per agent.

Fresh repository checkouts are pre-installed for Claude Code and `.agents`-root clients through checked-in relative symlinks at `.claude/skills/mlx-model-porting` and `.agents/skills/mlx-model-porting`. On Windows, Git may materialize symlinks as text files depending on local configuration. Native copy mode is intentionally unavailable because the attested installer requires POSIX no-follow directory APIs. Use WSL for a manifest-attested copy, or point the client directly at the checked-out `mlx-model-porting/` directory; Developer Mode symlinks are also viable when the client supports them. Do not replace the checked-in link with an unverified recursive copy.

Use the documented client preset first:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex
```

Use `--dest` when you need explicit control over a version-specific or user-scoped root:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest PATH_TO_CLIENT_SKILLS_ROOT
```

`install_skill.py --client` uses the repo-scoped defaults below and prints the resolved destination and mode before acting. `--mode` can override a preset mode when a client installation requires it.

| Client | Preset | Repository root | Recommended mode | Notes |
|---|---|---|---|---|
| Claude Code | `claude-code` | `.claude/skills` | `symlink` | Common builds use repository/user `.claude/skills` roots; verify with the product’s skill listing command for user-scoped installs. |
| OpenAI Codex | `codex` | `.agents/skills` | `symlink` | User-scoped installs use `~/.agents/skills`; pass that with `--dest`. |
| Cursor | `cursor` | `.cursor/skills` | `copy` | Version-dependent discovery; verify through Cursor’s Skills UI/installer. |
| Gemini CLI | `gemini` | `.gemini/skills` | `copy` | Common builds use `.gemini/skills`; verify discovery before relying on it. |
| Windsurf | `windsurf` | `.windsurf/skills` | `copy` | Version-dependent discovery; verify through Windsurf’s Skills UI. |
| GitHub Copilot | `copilot` | `.github/skills` | `copy` | Keep scripts subject to workspace trust and review in VS Code/Copilot. |
| Google Antigravity | `antigravity` | `.agents/skills` | `symlink` | Shares the Codex `.agents/skills` workspace root; a fresh checkout is already linked there. User-global scope lives at `~/.gemini/config/skills` — pass it with `--dest`; do not assume a legacy Gemini path. |

## Verification prompt

After installation, start a fresh agent session and ask:

> List the loaded skill named `mlx-model-porting`, then state its version and the first four non-negotiable rules without running any model code.

Expected version: `0.7.0`. The response should mention static intake, source pinning/oracle, and no optimization before parity.

## Check an installed copy for version drift

Compare an installed copy against the repository VERSION file without modifying anything:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex --check
```

It prints `current`, `stale`, or `missing`, and exits `0` only when the installed copy matches the repository VERSION file (non-zero when stale or missing). The same `--check` flag works with `--dest`.

## Adapter policy

- The shared skill is the source of truth.
- Agent-specific files may explain discovery/permissions only; they must not duplicate runbooks.
- If a client cannot execute Python scripts, the agent may read them as deterministic specifications and reproduce the steps with available tools.
- Network and shell permissions remain client-controlled. The skill never assumes pre-approval.
