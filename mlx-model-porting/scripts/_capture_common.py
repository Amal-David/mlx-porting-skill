#!/usr/bin/env python3
"""Dependency-free helpers shared by source and MLX tensor capture tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from _common import SkillError, sha256_file


SCHEMA_VERSION = 1
DEFAULT_GENERATE_STEPS = 4
DEFAULT_MAX_OUTPUT_MB = 1024.0
MAX_MANIFEST_JSON_BYTES = 16 * 1024 * 1024
MAX_PROMPTS_FILE_BYTES = 1024 * 1024
MAX_PROMPTS = 64
MAX_PROMPT_CHARACTERS = 64 * 1024
MAX_TOKEN_IDS = 16 * 1024
MAX_LISTED_WEIGHT_FILES = 256


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def strict_json_bytes(value: Any, *, label: str = "value") -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                sort_keys=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SkillError(f"{label} is not strict JSON: {exc}") from exc


def read_bounded_json(path: Path, limit: int, *, label: str) -> Any:
    """Read bounded strict UTF-8 JSON from a regular non-symlink file."""
    if path.is_symlink():
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise SkillError(f"could not read {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    if metadata.st_size > limit:
        raise SkillError(f"{label} exceeds size limit: {metadata.st_size} > {limit} bytes")
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise SkillError(f"could not read {label} {path}: {exc}") from exc
    if len(raw) > limit:
        raise SkillError(f"{label} exceeds size limit while reading: {path.name}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError(f"{label} must be valid UTF-8: {path.name}") from exc
    try:
        return json.loads(text, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise SkillError(f"{label} is not valid JSON: {path.name}: {exc.msg}") from exc


def read_bounded_text(path: Path, limit: int, *, label: str) -> str:
    if path.is_symlink():
        raise SkillError(f"{label} must be a regular non-symlink file: {path}")
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise SkillError(f"{label} must be a regular non-symlink file: {path}")
        if metadata.st_size > limit:
            raise SkillError(f"{label} exceeds size limit: {metadata.st_size} > {limit} bytes")
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise SkillError(f"could not read {label} {path}: {exc}") from exc
    if len(raw) > limit:
        raise SkillError(f"{label} exceeds size limit while reading: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError(f"{label} must be valid UTF-8: {path}") from exc


def parse_token_ids(values: list[str]) -> list[int]:
    pieces: list[str] = []
    for value in values:
        pieces.extend(value.split(","))
    if not pieces or any(not piece.strip() for piece in pieces):
        raise SkillError("--token-ids must contain one or more comma- or space-separated integers")
    token_ids: list[int] = []
    for piece in pieces:
        try:
            value = int(piece.strip(), 10)
        except ValueError as exc:
            raise SkillError(f"--token-ids contains a non-integer value: {piece!r}") from exc
        if value < 0:
            raise SkillError("--token-ids values must be non-negative integers")
        token_ids.append(value)
    if len(token_ids) > MAX_TOKEN_IDS:
        raise SkillError(f"--token-ids exceeds the {MAX_TOKEN_IDS}-token limit")
    return token_ids


def collect_prompts(inline: list[str] | None, files: list[str] | None) -> list[str]:
    prompts = list(inline or [])
    for raw_path in files or []:
        text = read_bounded_text(
            Path(raw_path),
            MAX_PROMPTS_FILE_BYTES,
            label="prompts file",
        )
        prompts.extend(line.rstrip("\r") for line in text.splitlines() if line.strip())
    if len(prompts) > MAX_PROMPTS:
        raise SkillError(f"prompt batch exceeds the {MAX_PROMPTS}-prompt limit")
    if any(len(prompt) > MAX_PROMPT_CHARACTERS for prompt in prompts):
        raise SkillError(f"a prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit")
    return prompts


def validate_input_mode(
    token_id_values: list[str] | None,
    inline_prompts: list[str] | None,
    prompt_files: list[str] | None,
) -> tuple[list[str], list[int] | None]:
    token_ids = parse_token_ids(token_id_values) if token_id_values else None
    prompts = collect_prompts(inline_prompts, prompt_files)
    if token_ids is not None and prompts:
        raise SkillError("--token-ids cannot be combined with --prompt or --prompts-file")
    if token_ids is None and not prompts:
        raise SkillError("provide --prompt, --prompts-file, or --token-ids")
    return prompts, token_ids


def validate_capture_limits(generate_steps: int, max_output_mb: float) -> int:
    if generate_steps < 0:
        raise SkillError("--generate-steps must be a non-negative integer")
    if not math.isfinite(max_output_mb) or max_output_mb < 0:
        raise SkillError("--max-output-mb must be a finite non-negative number")
    return int(max_output_mb * 1024 * 1024)


def absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def validate_output_path(path: Path, *, label: str) -> Path:
    """Reject an output whose existing lexical path contains any symlink."""
    absolute = absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SkillError(f"could not inspect {label} path component {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SkillError(f"{label} path contains symlink component: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SkillError(f"{label} parent component is not a directory: {current}")
    try:
        leaf = os.lstat(absolute)
    except FileNotFoundError:
        return absolute
    except OSError as exc:
        raise SkillError(f"could not inspect {label} path {absolute}: {exc}") from exc
    if stat.S_ISLNK(leaf.st_mode):
        raise SkillError(f"{label} must not be a symlink: {absolute}")
    if not stat.S_ISREG(leaf.st_mode):
        raise SkillError(f"{label} must be a regular file path: {absolute}")
    return absolute


def prepare_output_parent(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise SkillError(f"could not create {label} parent {current}: {exc}") from exc
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise SkillError(f"{label} path contains symlink component: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SkillError(f"{label} parent component is not a directory: {current}")
    validate_output_path(path, label=label)


def portable_basename(path: Path) -> str:
    name = PurePosixPath(path.name).as_posix()
    if not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise SkillError(f"artifact name is not portable: {path.name!r}")
    return name


def regular_file_record(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise SkillError(f"could not inspect {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    return {
        "name": portable_basename(path),
        "size_bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def build_model_record(
    directory: Path,
    config_path: Path,
    weight_paths: list[Path],
) -> dict[str, Any]:
    config_record = regular_file_record(config_path, label="config.json")
    weight_records = [
        regular_file_record(path, label="model weight") for path in sorted(weight_paths)
    ]
    identity = strict_json_bytes(
        {"schema_version": 1, "config": config_record, "weights": weight_records},
        label="model identity",
    )
    listed = weight_records[:MAX_LISTED_WEIGHT_FILES]
    return {
        "directory": portable_basename(directory) or ".",
        "fingerprint": "sha256:" + hashlib.sha256(identity).hexdigest(),
        "config": config_record,
        "weights": {
            "algorithm": "sha256",
            "file_count": len(weight_records),
            "total_bytes": sum(record["size_bytes"] for record in weight_records),
            "max_listed_files": MAX_LISTED_WEIGHT_FILES,
            "truncated": len(weight_records) > len(listed),
            "omitted_file_count": len(weight_records) - len(listed),
            "files": listed,
        },
    }


def _require_exact_fields(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SkillError(f"{label} must be a JSON object")
    if set(value) != fields:
        raise SkillError(f"{label} has an invalid field set")
    return value


def _validate_sha256(value: Any, *, label: str, prefixed: bool = False) -> None:
    pattern = r"sha256:[0-9a-f]{64}\Z" if prefixed else r"[0-9a-f]{64}\Z"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise SkillError(f"{label} must be a lowercase SHA-256 digest")


def _validate_file_record(value: Any, *, label: str) -> None:
    record = _require_exact_fields(value, {"name", "size_bytes", "sha256"}, label=label)
    name = record["name"]
    if (
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).is_absolute()
        or len(PurePosixPath(name).parts) != 1
        or ".." in PurePosixPath(name).parts
        or "\\" in name
    ):
        raise SkillError(f"{label}.name must be a portable basename")
    if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
        raise SkillError(f"{label}.size_bytes must be a non-negative integer")
    _validate_sha256(record["sha256"], label=f"{label}.sha256")


def validate_capture_manifest(payload: Any) -> dict[str, Any]:
    """Validate the source-oracle-compatible capture manifest schema."""
    manifest = _require_exact_fields(
        payload,
        {"schema_version", "model", "capture", "tensors", "libraries"},
        label="oracle manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise SkillError(f"oracle manifest schema_version must be integer {SCHEMA_VERSION}")

    model = _require_exact_fields(
        manifest["model"],
        {"directory", "fingerprint", "config", "weights"},
        label="oracle manifest model",
    )
    directory = model["directory"]
    if (
        not isinstance(directory, str)
        or not directory
        or PurePosixPath(directory).is_absolute()
        or len(PurePosixPath(directory).parts) != 1
        or ".." in PurePosixPath(directory).parts
        or "\\" in directory
    ):
        raise SkillError("oracle manifest model.directory must be a portable basename")
    _validate_sha256(model["fingerprint"], label="oracle manifest model.fingerprint", prefixed=True)
    _validate_file_record(model["config"], label="oracle manifest model.config")
    weights = _require_exact_fields(
        model["weights"],
        {
            "algorithm",
            "file_count",
            "total_bytes",
            "max_listed_files",
            "truncated",
            "omitted_file_count",
            "files",
        },
        label="oracle manifest model.weights",
    )
    if weights["algorithm"] != "sha256":
        raise SkillError("oracle manifest model.weights.algorithm must be 'sha256'")
    for key in ("file_count", "total_bytes", "max_listed_files", "omitted_file_count"):
        if type(weights[key]) is not int or weights[key] < 0:
            raise SkillError(f"oracle manifest model.weights.{key} must be a non-negative integer")
    if not isinstance(weights["truncated"], bool) or not isinstance(weights["files"], list):
        raise SkillError("oracle manifest model.weights truncation fields are invalid")
    if weights["file_count"] != len(weights["files"]) + weights["omitted_file_count"]:
        raise SkillError("oracle manifest model.weights counts are inconsistent")
    if weights["truncated"] != (weights["omitted_file_count"] > 0):
        raise SkillError("oracle manifest model.weights truncated flag is inconsistent")
    if len(weights["files"]) > weights["max_listed_files"]:
        raise SkillError("oracle manifest model.weights exceeds max_listed_files")
    weight_names: list[str] = []
    for index, record in enumerate(weights["files"]):
        _validate_file_record(record, label=f"oracle manifest model.weights.files[{index}]")
        weight_names.append(record["name"])
    if weight_names != sorted(weight_names) or len(weight_names) != len(set(weight_names)):
        raise SkillError("oracle manifest model.weights.files must be sorted and unique")
    if not weights["truncated"] and weights["total_bytes"] != sum(
        record["size_bytes"] for record in weights["files"]
    ):
        raise SkillError("oracle manifest model.weights.total_bytes is inconsistent")

    capture = _require_exact_fields(
        manifest["capture"],
        {"prompts", "token_ids", "generate_steps", "seed", "dtype_policy"},
        label="oracle manifest capture",
    )
    prompts = capture["prompts"]
    token_ids = capture["token_ids"]
    if prompts is not None and (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(item, str) for item in prompts)
    ):
        raise SkillError("oracle manifest capture.prompts must be null or an array of strings")
    if token_ids is not None and (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(type(item) is int and item >= 0 for item in token_ids)
    ):
        raise SkillError("oracle manifest capture.token_ids must be null or non-negative integers")
    if (prompts is None) == (token_ids is None):
        raise SkillError("oracle manifest capture must record exactly one input mode")
    if type(capture["generate_steps"]) is not int or capture["generate_steps"] < 0:
        raise SkillError("oracle manifest capture.generate_steps must be a non-negative integer")
    if type(capture["seed"]) is not int:
        raise SkillError("oracle manifest capture.seed must be an integer")
    if capture["dtype_policy"] not in {"float32", "keep"}:
        raise SkillError("oracle manifest capture.dtype_policy must be 'float32' or 'keep'")

    tensors = manifest["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise SkillError("oracle manifest tensors must be a non-empty array")
    tensor_names: list[str] = []
    for index, value in enumerate(tensors):
        record = _require_exact_fields(
            value,
            {"name", "shape", "dtype", "sha256"},
            label=f"oracle manifest tensors[{index}]",
        )
        if not isinstance(record["name"], str) or not record["name"]:
            raise SkillError(f"oracle manifest tensors[{index}].name must be non-empty")
        if (
            not isinstance(record["shape"], list)
            or not all(type(item) is int and item >= 0 for item in record["shape"])
        ):
            raise SkillError(f"oracle manifest tensors[{index}].shape is invalid")
        if not isinstance(record["dtype"], str) or not record["dtype"]:
            raise SkillError(f"oracle manifest tensors[{index}].dtype must be non-empty")
        _validate_sha256(record["sha256"], label=f"oracle manifest tensors[{index}].sha256")
        tensor_names.append(record["name"])
    if tensor_names != sorted(tensor_names) or len(tensor_names) != len(set(tensor_names)):
        raise SkillError("oracle manifest tensor names must be sorted and unique")

    libraries = _require_exact_fields(
        manifest["libraries"],
        {"python", "numpy", "torch", "transformers"},
        label="oracle manifest libraries",
    )
    if not all(isinstance(value, str) and value for value in libraries.values()):
        raise SkillError("oracle manifest library versions must be non-empty strings")
    return manifest


def tensor_inventory(arrays: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
        for name, array in sorted(arrays.items())
    ]


def raw_npz_bound(arrays: dict[str, Any]) -> int:
    return sum(int(array.nbytes) + len(name.encode("utf-8")) + 512 for name, array in arrays.items())


def _write_staged_npz(path: Path, arrays: dict[str, Any], cap_bytes: int, np: Any) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        size = temporary.stat().st_size
        if size > cap_bytes:
            raise SkillError(f"NPZ output exceeds --max-output-mb: {size} > {cap_bytes} bytes")
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_staged_json(path: Path, payload: dict[str, Any], *, label: str) -> Path:
    raw = strict_json_bytes(payload, label=label)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_strict_json(path: Path, payload: dict[str, Any], *, label: str) -> None:
    """Atomically replace one strict-JSON output without following symlink components."""
    output = validate_output_path(path, label=label)
    prepare_output_parent(output, label=label)
    temporary: Path | None = None
    try:
        temporary = _write_staged_json(output, payload, label=label)
        validate_output_path(output, label=label)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def write_capture_outputs(
    output: Path,
    manifest: Path,
    arrays: dict[str, Any],
    manifest_payload: dict[str, Any],
    cap_bytes: int,
    np: Any,
) -> None:
    bounded_size = raw_npz_bound(arrays)
    if bounded_size > cap_bytes:
        raise SkillError(
            f"captured tensor payload exceeds --max-output-mb: {bounded_size} > {cap_bytes} bytes"
        )
    validate_capture_manifest(manifest_payload)
    prepare_output_parent(output, label="NPZ output")
    prepare_output_parent(manifest, label="manifest output")
    staged_npz: Path | None = None
    staged_manifest: Path | None = None
    try:
        staged_npz = _write_staged_npz(output, arrays, cap_bytes, np)
        staged_manifest = _write_staged_json(
            manifest,
            manifest_payload,
            label="oracle manifest",
        )
        validate_output_path(output, label="NPZ output")
        validate_output_path(manifest, label="manifest output")
        os.replace(staged_npz, output)
        staged_npz = None
        os.replace(staged_manifest, manifest)
        staged_manifest = None
    finally:
        for temporary in (staged_npz, staged_manifest):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
