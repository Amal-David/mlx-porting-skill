#!/usr/bin/env python3
"""Repository-owned attested runner for the Qwen2.5 worked-port benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ADAPTER = {"id": "attested-mlx-port-wall-time", "version": 1}
MODEL_ID = "local/Qwen2.5-0.5B-Instruct-standalone-mlx"
WORKLOAD_ID = "qwen2.5-0.5b-load-plus-six-token-greedy-decode"
VARIANTS = {
    "f32": {
        "revision": "7df68182b704fb0933625cdb44cc4e89e8f4cfdb5cb75b3e2251f608807f65f0",
        "weights": "converted-f32/model.safetensors",
        "size_bytes": 1_976_163_149,
    },
    "bf16": {
        "revision": "7d325703c071cbba116c2cbd2242f20510c797fb9d9384573e5f6ca370582c7f",
        "weights": "converted-bf16/model.safetensors",
        "size_bytes": 988_097_714,
    },
}
MAX_DEPENDENCY_BYTES = 16 * 1024 * 1024
MAX_DEPENDENCY_SET_BYTES = 32 * 1024 * 1024
HEX_DIGEST_CHARS = frozenset("0123456789abcdef")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_regular_file(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe benchmark path: {value}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked benchmark path: {value}")
    resolved = current.resolve()
    resolved.relative_to(root.resolve())
    if not resolved.is_file():
        raise FileNotFoundError(f"benchmark file is missing: {value}")
    return resolved


def relative_output(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe benchmark output path: {value}")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ValueError(f"unsafe benchmark output parent: {value}")
        current.mkdir(exist_ok=True)
    output = root / relative
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError(f"unsafe benchmark output: {value}")
    return output


def file_descriptor(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    return {
        **({"path": logical_path} if logical_path is not None else {}),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def bytes_descriptor(raw: bytes, *, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--mode", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attestation-challenge", required=True)
    parser.add_argument("--attestation-output", required=True)
    return parser.parse_args()


def dependency_sources(package: Path) -> list[tuple[str, list[str], str, Path]]:
    grouped: dict[Path, dict[str, Any]] = {}
    for name, module in sorted(sys.modules.items()):
        if not (
            name in {"model", "config", "mlx", "_mlx"}
            or name.startswith("mlx.")
            or name.startswith("_mlx.")
        ):
            continue
        value = getattr(module, "__file__", None)
        if not isinstance(value, str):
            continue
        path = Path(value).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"loaded dependency is not a regular file: {name}")
        try:
            logical = f"port/{path.relative_to(package.resolve())}"
            scope = "port-package"
        except ValueError:
            if name not in {"mlx", "_mlx"} and not name.startswith(("mlx.", "_mlx.")):
                raise ValueError(f"loaded port module escaped the package root: {name}")
            logical = f"runtime/{name.replace('.', '/')}/{path.name}"
            scope = "mlx-runtime"
        row = grouped.setdefault(path, {"scope": scope, "modules": [], "logical": logical})
        row["modules"].append(name)
    result = [
        (row["scope"], sorted(row["modules"]), row["logical"], path)
        for path, row in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    config_path = package / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError("loaded model config is missing or unsafe")
    result.append(("port-config", [], "port/config.json", config_path.resolve()))
    return result


def snapshot_dependencies(root: Path, package: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    total = 0
    for scope, module_names, logical_path, source in dependency_sources(package):
        size = source.stat().st_size
        total += size
        if size > MAX_DEPENDENCY_BYTES or total > MAX_DEPENDENCY_SET_BYTES:
            raise ValueError("loaded dependency evidence exceeds the bounded attestation limit")
        digest = file_sha256(source)
        relative = f"attestations/dependencies/{digest}.bin"
        destination = relative_output(root, relative)
        if destination.exists():
            if destination.stat().st_size != size or file_sha256(destination) != digest:
                raise ValueError("content-addressed dependency evidence collision")
        else:
            shutil.copyfile(source, destination)
        result.append({
            "scope": scope,
            "module_names": module_names,
            "loaded_path": logical_path,
            "artifact": {"path": relative, "sha256": digest, "size_bytes": size},
        })
    scopes = {row["scope"] for row in result}
    if scopes != {"mlx-runtime", "port-package", "port-config"}:
        raise ValueError("attestation did not capture MLX, port-package, and model-config bytes")
    return result


def main() -> int:
    args = parse_args()
    variant = VARIANTS[args.mode]
    if args.model != MODEL_ID or args.revision != variant["revision"]:
        raise ValueError("model identity does not match the selected local variant")

    root = Path.cwd().resolve()
    challenge_path = relative_regular_file(root, args.attestation_challenge)
    challenge_raw = challenge_path.read_bytes()
    challenge = json.loads(challenge_raw.decode("utf-8"))
    challenge_parts = Path(args.attestation_challenge).parts
    evidence_parts = Path(args.attestation_output).parts
    challenge_label = challenge_parts[1] if len(challenge_parts) == 3 else None
    quality_contract = challenge.get("quality_contract") if isinstance(challenge, dict) else None
    if (
        not isinstance(challenge, dict)
        or set(challenge) != {
            "schema_version",
            "nonce",
            "receipt_label",
            "phase",
            "run_index",
            "command",
            "command_sha256",
            "runner_argv_sha256",
            "quality_contract",
        }
        or challenge.get("schema_version") != 1
        or challenge_parts != ("attestations", challenge_label, "current-challenge.json")
        or evidence_parts != ("attestations", challenge_label, "current-evidence.json")
        or challenge.get("receipt_label") != challenge_label
        or challenge.get("phase") not in {"warmup", "measure"}
        or type(challenge.get("run_index")) is not int
        or challenge["run_index"] < 1
        or not isinstance(challenge.get("nonce"), str)
        or len(challenge["nonce"]) != 64
        or any(char not in HEX_DIGEST_CHARS for char in challenge["nonce"])
        or challenge.get("command") != [sys.executable.rsplit("/", 1)[-1], *sys.argv]
        or challenge.get("command_sha256") != canonical_hash(challenge.get("command"))
        or challenge.get("runner_argv_sha256") != canonical_hash(sys.argv)
        or not isinstance(quality_contract, dict)
        or set(quality_contract) != {"path", "sha256", "size_bytes"}
    ):
        raise ValueError("parent attestation challenge is invalid for this argv")

    quality_contract_path = relative_regular_file(root, quality_contract["path"])
    if (
        quality_contract_path.stat().st_size != quality_contract["size_bytes"]
        or file_sha256(quality_contract_path) != quality_contract["sha256"]
    ):
        raise ValueError("parent quality-contract snapshot does not match the challenge")

    input_path = relative_regular_file(root, args.input)
    input_raw = input_path.read_bytes()
    fixture = json.loads(input_raw.decode("utf-8"))
    token_ids = fixture.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(type(value) is int and value >= 0 for value in token_ids)
        or args.steps != fixture.get("max_new_tokens")
        or args.steps != 6
    ):
        raise ValueError("input fixture is invalid or disagrees with the fixed workload")

    work = Path.home() / ".cache" / "mlx-porting-work" / "qwen2.5-0.5b-instruct" / "run"
    package = work / "mlx_port"
    weights = work / str(variant["weights"])
    if not package.is_dir() or package.is_symlink() or not weights.is_file() or weights.is_symlink():
        raise FileNotFoundError("local worked-port package or converted weights are missing or unsafe")
    model_artifact = file_descriptor(weights, logical_path=str(variant["weights"]))
    if model_artifact["sha256"] != args.revision or model_artifact["size_bytes"] != variant["size_bytes"]:
        raise ValueError("on-disk model artifact does not match the declared revision and size")

    sys.path.insert(0, str(package))
    try:
        for name in ("config", "model"):
            sys.modules.pop(name, None)
        generated_model = importlib.import_module("model")
        import mlx.core as mx

        model = generated_model.load_model(package / "config.json", weights)
        if file_descriptor(weights, logical_path=str(variant["weights"])) != model_artifact:
            raise ValueError("model artifact changed while it was loaded")
        generated = generated_model.greedy_generate(
            model,
            mx.array([token_ids], dtype=mx.int32),
            args.steps,
        )
        payload = {"generated_token_ids": generated.tolist()[0]}
        output = relative_output(root, args.output)
        output.write_text(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_artifact = file_descriptor(output, logical_path=args.output)
        dependencies = snapshot_dependencies(root, package)
    finally:
        sys.path.pop(0)

    input_artifact = bytes_descriptor(input_raw, logical_path=args.input)
    if file_descriptor(input_path, logical_path=args.input) != input_artifact:
        raise ValueError("workload input changed while it was consumed")
    workload = {
        "id": WORKLOAD_ID,
        "model": {"id": args.model, "revision": args.revision},
        "input": input_artifact,
        "parameters": {
            "batch": 1,
            "cache": "growing-kv",
            "compile": False,
            "generate_steps": args.steps,
            "prompt_tokens": len(token_ids),
            "timing_scope": "separate-process-load-plus-greedy-decode",
        },
        "variant": {"mode": args.mode, "quality_output_path": args.output},
    }
    runner_path = Path(__file__).resolve()
    evidence_payload = {
        "schema_version": 1,
        "adapter": {
            **ADAPTER,
            "implementation": file_descriptor(
                runner_path,
                logical_path="runners/attested_mlx_port.py",
            ),
        },
        "challenge": {
            "sha256": hashlib.sha256(challenge_raw).hexdigest(),
            "size_bytes": len(challenge_raw),
        },
        "runner_argv_sha256": canonical_hash(sys.argv),
        "workload": {"descriptor": workload, "sha256": canonical_hash(workload)},
        "model_artifact": model_artifact,
        "dependencies": dependencies,
        "dependency_set_sha256": canonical_hash(dependencies),
        "output_artifact": output_artifact,
    }
    evidence = {**evidence_payload, "evidence_sha256": canonical_hash(evidence_payload)}
    attestation_output = relative_output(root, args.attestation_output)
    attestation_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
