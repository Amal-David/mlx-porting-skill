#!/usr/bin/env python3
"""Deterministically (re)generate the tiny synthetic test fixtures.

Every fixture under ``tests/fixtures`` is a small, non-executable stand-in. This
script is the single source of truth for them so they are reproducible and
auditable: no committed binary blob exists without a spec here that explains it.

Usage:
    python3 tests/fixtures/generate_fixtures.py            # rewrite fixtures in place
    python3 tests/fixtures/generate_fixtures.py --dest DIR # write a copy elsewhere

Determinism: tensor values come from a SHA-256-seeded numpy RNG keyed by
``model/key`` so repeated runs are byte-identical for the raw ``.safetensors``
files (``.npz`` archives match by array content; their zip container embeds
timestamps and is therefore compared by value, not bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES_ROOT = Path(__file__).resolve().parent

DTYPE_TO_NUMPY = {"F16": np.float16, "F32": np.float32}

# --- Declarative model specs -------------------------------------------------
# Shapes are deliberately tiny. The codec codebook is a size-reduced stand-in
# (the real EnCodec codebook would be [num_codebooks, codebook_size, dim]); a
# 2 MB blob is not needed to exercise name-based architecture routing.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "decoder": {
        "config": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "torch_dtype": "float16",
            "license": "apache-2.0",
        },
        "tensors": {
            "model.embed_tokens.weight": ("F16", [32, 8]),
            "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
            "model.layers.0.mlp.gate_proj.weight": ("F16", [16, 8]),
            "lm_head.weight": ("F16", [32, 8]),
        },
    },
    "moe": {
        "config": {
            "model_type": "mixtral",
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "vocab_size": 32,
            "license": "apache-2.0",
        },
        "tensors": {
            "model.layers.0.block_sparse_moe.gate.weight": ("F16", [4, 8]),
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": ("F16", [16, 8]),
            "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
        },
    },
    "codec": {
        "config": {
            "model_type": "encodec",
            "architectures": ["EncodecModel"],
            "sample_rate": 24000,
            "codebook_size": 1024,
            "num_codebooks": 8,
            "hop_length": 320,
            "audio_channels": 1,
            "license": "mit",
        },
        "tensors": {
            "encoder.layers.0.conv.weight": ("F16", [32, 1, 7]),
            # Size-reduced stand-in for the residual-VQ codebook (was [8, 1024, 128]).
            "quantizer.codebook.weight": ("F16", [8, 32, 16]),
            "decoder.layers.0.conv.weight": ("F16", [1, 32, 7]),
        },
    },
}

# Unsafe fixture: a non-executable lure that must trip the remote-code/pickle gates.
UNSAFE_FILES: dict[str, bytes] = {
    "config.json": json.dumps(
        {
            "model_type": "mystery",
            "architectures": ["MysteryModel"],
            "auto_map": {"AutoModel": "modeling_mystery.MysteryModel"},
        },
        indent=2,
    ).encode()
    + b"\n",
    "modeling_mystery.py": b"class MysteryModel: pass\n",
    "setup.py": b"from setuptools import setup\n",
    "pytorch_model.bin": b"not-a-real-pickle",
}

# Tensor-oracle .npz fixtures used by compare_tensors tests.
NPZ_SHAPES = {"hidden": (3, 4), "logits": (2,)}


def _seeded(model: str, key: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    digest = hashlib.sha256(f"{model}/{key}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed).standard_normal(tuple(shape))


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]], model: str) -> None:
    header: dict[str, Any] = {}
    buffer = bytearray()
    for key, (dtype, shape) in tensors.items():
        array = _seeded(model, key, shape).astype(DTYPE_TO_NUMPY[dtype])
        raw = array.tobytes()
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(buffer), len(buffer) + len(raw)]}
        buffer.extend(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(bytes(buffer))


def write_npz_fixtures(root: Path) -> None:
    source = {name: _seeded("tensors-source", name, shape).astype(np.float32) for name, shape in NPZ_SHAPES.items()}
    close = {name: array.copy() for name, array in source.items()}
    close["hidden"] = (close["hidden"] + np.float32(1e-7)).astype(np.float32)  # within atol/rtol
    bad = {name: array.copy() for name, array in source.items()}
    bad["hidden"].flat[0] += np.float32(11.0)  # large divergence
    bad["logits"][0] += np.float32(0.7)
    out = root / "tensors"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in {"source": source, "close": close, "bad": bad}.items():
        np.savez(out / f"{name}.npz", **data)


def generate(root: Path) -> list[Path]:
    written: list[Path] = []
    for model, spec in MODEL_SPECS.items():
        model_dir = root / "models" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "config.json"
        config_path.write_text(json.dumps(spec["config"], indent=2) + "\n", encoding="utf-8")
        st_path = model_dir / "model.safetensors"
        write_safetensors(st_path, spec["tensors"], model)
        written += [config_path, st_path]
    unsafe_dir = root / "models" / "unsafe"
    unsafe_dir.mkdir(parents=True, exist_ok=True)
    for name, data in UNSAFE_FILES.items():
        (unsafe_dir / name).write_bytes(data)
        written.append(unsafe_dir / name)
    write_npz_fixtures(root)
    written += [root / "tensors" / f"{n}.npz" for n in ("source", "close", "bad")]
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate synthetic test fixtures")
    parser.add_argument("--dest", default=str(FIXTURES_ROOT), help="Destination fixtures root")
    args = parser.parse_args()
    written = generate(Path(args.dest))
    print(f"wrote {len(written)} fixture files under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
