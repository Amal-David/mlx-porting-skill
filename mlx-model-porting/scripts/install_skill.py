#!/usr/bin/env python3
"""Copy or symlink this skill into an explicit Agent Skills root."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from _common import SkillError

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install mlx-model-porting into an explicit Agent Skills directory")
    parser.add_argument("--dest", required=True, help="Agent Skills root, e.g. .agents/skills")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--name", default=SOURCE.name)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.name or "/" in args.name or "\\" in args.name or args.name in {".", ".."}:
            raise SkillError("--name must be a single safe directory name")
        dest_root = Path(args.dest).expanduser().resolve()
        target = (dest_root / args.name).resolve(strict=False)
        if target == SOURCE.resolve():
            raise SkillError("Target is the source skill directory")
        try:
            target.relative_to(dest_root)
        except ValueError as exc:
            raise SkillError("Target escapes destination root") from exc
        print(f"source: {SOURCE}")
        print(f"target: {target}")
        print(f"mode: {args.mode}")
        if args.dry_run:
            return 0
        dest_root.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not args.force:
                raise SkillError(f"Target already exists: {target}; use --force to replace")
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if args.mode == "copy":
            shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            relative = os.path.relpath(SOURCE, dest_root)
            target.symlink_to(relative, target_is_directory=True)
        print("installed")
        return 0
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
