# github-copilot adapter

Use the unchanged `mlx-model-porting/` directory. For the documented repository root, run:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client copilot
```

For explicit control over a version-specific or user-scoped root, run:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest PATH_TO_SKILLS_ROOT
```

Restart or reload the client, list loaded skills, and verify `mlx-model-porting` version `0.4.0`. Do not copy technical instructions into a client-specific prompt; that would create update drift.
