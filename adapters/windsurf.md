# windsurf adapter

Use the unchanged `mlx-model-porting/` directory. Resolve this client's current Agent Skills root, then run:

```bash
python3 mlx-model-porting/scripts/install_skill.py --dest PATH_TO_SKILLS_ROOT
```

Restart or reload the client, list loaded skills, and verify `mlx-model-porting` version `0.1.0`. Do not copy technical instructions into a client-specific prompt; that would create update drift.
