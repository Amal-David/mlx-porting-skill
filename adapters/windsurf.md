# windsurf adapter

Use the unchanged `mlx-model-porting/` directory. For the documented repository root, run:

```bash
python3 mlx-model-porting/scripts/install_skill.py --client windsurf
```

For explicit control over a version-specific or user-scoped root, run:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest PATH_TO_SKILLS_ROOT
```

Restart or reload the client, list loaded skills, and verify `mlx-model-porting` version `0.1.0`. Do not copy technical instructions into a client-specific prompt; that would create update drift.
