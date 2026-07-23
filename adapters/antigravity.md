# antigravity adapter

Use the unchanged `mlx-model-porting/` directory. Antigravity resolves workspace skills from `.agents/skills` (the same root as Codex), so a fresh checkout is already linked there. Install or refresh the workspace link with the preset:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client antigravity
```

For Antigravity's user-global scope (`~/.gemini/config/skills`, not a legacy Gemini path), point `--dest` at it explicitly:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest PATH_TO_SKILLS_ROOT
```

Restart or reload the client, list loaded skills, and verify `mlx-model-porting` version `0.7.0`. Do not copy technical instructions into a client-specific prompt; that would create update drift.
