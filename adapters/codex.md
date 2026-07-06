# OpenAI Codex adapter

Codex discovers repository-scoped skills under `.agents/skills` and user-scoped skills under `~/.agents/skills`.

Repository installation:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client codex
```

Explicit repository root:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest .agents/skills --mode symlink
```

User installation:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest ~/.agents/skills
```

Start a new Codex session or reload skills, then verify `mlx-model-porting` version `0.2.0`.
