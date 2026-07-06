# Cross-agent installation

`mlx-model-porting/` follows the shared Agent Skills format: one `SKILL.md` plus optional `scripts/`, `references/`, and `assets/`. Use the same folder across clients; do not fork the technical corpus per agent.

Fresh repository checkouts are pre-installed for Claude Code and `.agents`-root clients through checked-in relative symlinks at `.claude/skills/mlx-model-porting` and `.agents/skills/mlx-model-porting`. On Windows, Git may materialize symlinks as text files depending on local configuration; use the installer with an explicit `--dest` path there.

The safest installer contract is explicit:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest PATH_TO_CLIENT_SKILLS_ROOT
```

This avoids silently installing into the wrong profile when products rename or move their discovery roots.

| Client | Recommended approach |
|---|---|
| OpenAI Codex | Repository: `.agents/skills`; user: `~/.agents/skills`. These roots are implemented by Codex’s skill loader. |
| Claude Code | Install the canonical folder into the Skills root shown by the installed Claude Code version or plugin manager. Common builds use repository/user `.claude/skills` roots; verify with the product’s skill listing command. |
| Gemini CLI | Use the Agent Skills root exposed by the installed version. Common builds use `.gemini/skills`; verify discovery before relying on it. |
| Cursor | Install through Cursor’s Skills UI/installer or its current project skills root. Keep the canonical folder unchanged. |
| Google Antigravity | Install through Antigravity’s current Skills/plugin manager. Pass its resolved root to `install_skill.py`; do not assume a legacy Gemini path. |
| Windsurf | Install through Windsurf’s Skills UI or current workspace/user skills root. Common builds use `.windsurf/skills`; verify discovery. |
| GitHub Copilot | Install through the Copilot/VS Code Agent Skills interface or current repository skills root. Keep scripts subject to workspace trust and review. |

## Verification prompt

After installation, start a fresh agent session and ask:

> List the loaded skill named `mlx-model-porting`, then state its version and the first three non-negotiable rules without running any model code.

Expected version: `0.1.0`. The response should mention static intake, source pinning/oracle, and no optimization before parity.

## Adapter policy

- The shared skill is the source of truth.
- Agent-specific files may explain discovery/permissions only; they must not duplicate runbooks.
- If a client cannot execute Python scripts, the agent may read them as deterministic specifications and reproduce the steps with available tools.
- Network and shell permissions remain client-controlled. The skill never assumes pre-approval.
