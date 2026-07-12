#!/usr/bin/env python3
"""Digest-pinned local runner for the Qwen2.5 worked-port receipts."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


MODEL_ID = "local/Qwen2.5-0.5B-Instruct-standalone-mlx"
VARIANTS = {
    "f32": {
        "revision": "7df68182b704fb0933625cdb44cc4e89e8f4cfdb5cb75b3e2251f608807f65f0",
        "weights": "converted-f32/model.safetensors",
    },
    "bf16": {
        "revision": "7d325703c071cbba116c2cbd2242f20510c797fb9d9384573e5f6ca370582c7f",
        "weights": "converted-bf16/model.safetensors",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--mode", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variant = VARIANTS[args.mode]
    if args.model != MODEL_ID or args.revision != variant["revision"]:
        raise ValueError("model identity does not match the selected local variant")

    fixture = json.loads(Path(args.input).read_text(encoding="utf-8"))
    token_ids = fixture.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(type(value) is int and value >= 0 for value in token_ids)
        or args.steps != fixture.get("max_new_tokens")
    ):
        raise ValueError("input fixture is invalid or disagrees with --steps")

    work = Path.home() / ".cache" / "mlx-porting-work" / "qwen2.5-0.5b-instruct" / "run"
    package = work / "mlx_port"
    weights = work / str(variant["weights"])
    if not package.is_dir() or not weights.is_file():
        raise FileNotFoundError("local worked-port package or converted weights are missing")

    sys.path.insert(0, str(package))
    try:
        for name in ("config", "model"):
            sys.modules.pop(name, None)
        generated_model = importlib.import_module("model")
        import mlx.core as mx

        model = generated_model.load_model(package / "config.json", weights)
        generated = generated_model.greedy_generate(
            model,
            mx.array([token_ids], dtype=mx.int32),
            args.steps,
        )
        payload = {"generated_token_ids": generated.tolist()[0]}
    finally:
        sys.path.pop(0)
        for name in ("config", "model"):
            sys.modules.pop(name, None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
