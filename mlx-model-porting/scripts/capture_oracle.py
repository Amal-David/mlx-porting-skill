#!/usr/bin/env python3
"""Capture a deterministic PyTorch/Hugging Face causal-decoder source oracle.

The stable NPZ tensor-key scheme is:

``input_ids``
    Token IDs supplied directly or produced by the local tokenizer.
``attention_mask``
    The corresponding integer attention mask.
``embed``
    Decoder embedding-stage hidden state before the first decoder block.
``layer.{i}.hidden``
    Post-block hidden state for zero-based decoder layer ``i``.
``layer.{i}.attention``
    Optional attention-module branch output when a standard submodule is found.
``layer.{i}.mlp``
    Optional MLP branch output before the block residual when hookable.
``final_norm``
    Output of the decoder's final normalization module.
``logits``
    Full prompt logits.
``generated_token_ids``
    Exactly ``--generate-steps`` greedy continuation IDs (not the prompt IDs).

PyTorch, Transformers, and NumPy are execution-only dependencies. Importing
this module, using ``--help``, and validating a manifest do not import them.
The execution path is local-only and never enables Hugging Face remote code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from _common import SkillError, sha256_file


SCHEMA_VERSION = 1
DEFAULT_GENERATE_STEPS = 4
DEFAULT_MAX_OUTPUT_MB = 1024.0
MAX_CONFIG_JSON_BYTES = 16 * 1024 * 1024
MAX_TOKENIZER_JSON_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_JSON_BYTES = 16 * 1024 * 1024
MAX_PROMPTS_FILE_BYTES = 1024 * 1024
MAX_PROMPTS = 64
MAX_PROMPT_CHARACTERS = 64 * 1024
MAX_TOKEN_IDS = 16 * 1024
MAX_MODEL_ROOT_ENTRIES = 10_000
MAX_WEIGHT_FILES = 4_096
MAX_LISTED_WEIGHT_FILES = 256

TOKENIZER_JSON_NAMES = frozenset({
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture deterministic intermediate tensors and greedy tokens from a local "
            "PyTorch/Hugging Face causal language model"
        ),
    )
    parser.add_argument("model", nargs="?", help="Local model directory; hub IDs are not accepted")
    parser.add_argument("--output", help="Destination .npz archive")
    parser.add_argument(
        "--manifest",
        help="Destination JSON manifest (default: <output stem>.manifest.json)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to tokenize locally; repeat for a prompt batch",
    )
    parser.add_argument(
        "--prompts-file",
        action="append",
        help="Bounded UTF-8 file containing one prompt per non-empty line; repeatable",
    )
    parser.add_argument(
        "--token-ids",
        nargs="+",
        help="One tokenizer-free token sequence as space- or comma-separated integers",
    )
    parser.add_argument(
        "--generate-steps",
        type=int,
        default=DEFAULT_GENERATE_STEPS,
        help=f"Exact greedy continuation length (default: {DEFAULT_GENERATE_STEPS})",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic execution seed")
    parser.add_argument(
        "--keep-dtype",
        action="store_true",
        help="Keep representable source floating dtypes instead of casting floats to float32",
    )
    parser.add_argument(
        "--max-output-mb",
        type=float,
        default=DEFAULT_MAX_OUTPUT_MB,
        help=f"Maximum raw and serialized NPZ size in MiB (default: {DEFAULT_MAX_OUTPUT_MB:g})",
    )
    parser.add_argument(
        "--validate-manifest",
        help="Validate one oracle manifest without importing execution dependencies",
    )
    return parser.parse_args(argv)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def read_bounded_json(path: Path, limit: int, *, label: str) -> Any:
    """Read a bounded regular UTF-8 JSON file without following the leaf."""
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


def _portable_name(path: Path) -> str:
    name = PurePosixPath(path.name).as_posix()
    if not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise SkillError(f"model artifact name is not portable: {path.name!r}")
    return name


def _regular_file_record(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise SkillError(f"could not inspect {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    return {
        "name": _portable_name(path),
        "size_bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def _model_root_entries(model_dir: Path) -> list[Path]:
    try:
        iterator = os.scandir(model_dir)
    except OSError as exc:
        raise SkillError(f"could not enumerate local model directory {model_dir}: {exc}") from exc
    entries: list[Path] = []
    with iterator:
        for entry in iterator:
            entries.append(Path(entry.path))
            if len(entries) > MAX_MODEL_ROOT_ENTRIES:
                raise SkillError(
                    f"model directory exceeds root-entry limit {MAX_MODEL_ROOT_ENTRIES}"
                )
    return sorted(entries, key=lambda path: path.name)


def _is_weight_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".safetensors") or (
        name.endswith(".bin")
        and (name.startswith("pytorch_model") or name.startswith("model"))
    )


def inspect_local_model(model_value: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    model_dir = Path(os.path.abspath(os.path.expanduser(model_value)))
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise SkillError(f"model must be a local, non-symlink directory: {model_value}")
    config_path = model_dir / "config.json"
    config = read_bounded_json(config_path, MAX_CONFIG_JSON_BYTES, label="config.json")
    if not isinstance(config, dict):
        raise SkillError("config.json must contain a JSON object")
    if config.get("auto_map"):
        raise SkillError(
            "config.json declares auto_map remote model code; capture_oracle.py refuses remote code"
        )
    architectures = config.get("architectures")
    if architectures is not None and (
        not isinstance(architectures, list)
        or not all(isinstance(item, str) and item for item in architectures)
    ):
        raise SkillError("config.json architectures must be an array of non-empty strings")

    entries = _model_root_entries(model_dir)
    weight_paths = [path for path in entries if _is_weight_file(path)]
    if not weight_paths:
        raise SkillError("local model directory has no .safetensors or PyTorch .bin weight files")
    if len(weight_paths) > MAX_WEIGHT_FILES:
        raise SkillError(f"model has more than {MAX_WEIGHT_FILES} weight files")

    config_record = _regular_file_record(config_path, label="config.json")
    weight_records = [
        _regular_file_record(path, label="model weight")
        for path in weight_paths
    ]
    identity_bytes = json.dumps(
        {
            "schema_version": 1,
            "config": config_record,
            "weights": weight_records,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    listed = weight_records[:MAX_LISTED_WEIGHT_FILES]
    model_record = {
        "directory": model_dir.name or ".",
        "fingerprint": "sha256:" + hashlib.sha256(identity_bytes).hexdigest(),
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
    return model_dir, config, model_record


def validate_tokenizer_json_files(model_dir: Path) -> None:
    for path in _model_root_entries(model_dir):
        name = path.name.lower()
        if name not in TOKENIZER_JSON_NAMES and not (
            name.startswith("tokenizer") and name.endswith(".json")
        ):
            continue
        payload = read_bounded_json(
            path,
            MAX_TOKENIZER_JSON_BYTES,
            label=f"tokenizer JSON {path.name}",
        )
        if name == "tokenizer_config.json" and isinstance(payload, dict) and payload.get("auto_map"):
            raise SkillError(
                "tokenizer_config.json declares auto_map remote tokenizer code; "
                "capture_oracle.py refuses remote code"
            )


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
        raise SkillError(
            f"a prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit"
        )
    return prompts


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def validate_output_path(path: Path, *, label: str) -> Path:
    """Reject an output whose existing lexical path contains any symlink."""
    absolute = _absolute_lexical(path)
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
    """Create missing parents one component at a time and recheck for links."""
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


def validate_manifest(payload: Any) -> dict[str, Any]:
    """Validate the dependency-free schema for a capture_oracle manifest."""
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


def _find_decoder_layers(model: Any, expected_layers: int | None, torch: Any) -> tuple[str, Any]:
    candidates: list[tuple[int, str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not module:
            continue
        if expected_layers is not None and len(module) != expected_layers:
            continue
        lowered = name.lower()
        leaf = lowered.rsplit(".", 1)[-1]
        score = 0
        if leaf == "layers":
            score += 100
        elif leaf in {"h", "blocks", "block"}:
            score += 80
        if "decoder" in lowered or lowered.startswith("model"):
            score += 20
        if "encoder" in lowered:
            score -= 100
        score += min(len(module), 20)
        candidates.append((score, name, module))
    if not candidates:
        expected = f" with {expected_layers} entries" if expected_layers is not None else ""
        raise SkillError(f"could not locate a standard decoder layer ModuleList{expected}")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0]:
        raise SkillError(
            "decoder layer discovery is ambiguous: "
            + ", ".join(item[1] or "<root>" for item in candidates[:2])
        )
    return best[1], best[2]


def _expected_layer_count(config: dict[str, Any]) -> int | None:
    for key in ("num_hidden_layers", "n_layer", "num_layers", "decoder_layers"):
        value = config.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def _find_final_norm(model: Any, layers_name: str) -> Any:
    parent_name, _, layers_leaf = layers_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    candidates: list[tuple[int, str, Any]] = []
    for name, module in parent.named_children():
        if name == layers_leaf:
            continue
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if "norm" not in lowered and "norm" not in class_name and lowered != "ln_f":
            continue
        score = 100 if lowered in {"norm", "final_layer_norm", "ln_f"} else 20
        candidates.append((score, name, module))
    if not candidates:
        raise SkillError("could not locate the decoder final normalization module")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        raise SkillError("decoder final normalization discovery is ambiguous")
    return candidates[0][2]


def _find_branch_module(layer: Any, branch: str) -> Any | None:
    if branch == "attention":
        preferred = {"self_attn": 100, "attn": 90, "attention": 80, "self_attention": 80}
        marker = ("attn", "attention")
    else:
        preferred = {"mlp": 100, "feed_forward": 90, "ffn": 80, "feedforward": 80}
        marker = ("mlp", "feed_forward", "ffn", "feedforward")
    candidates: list[tuple[int, str, Any]] = []
    for name, module in layer.named_children():
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if lowered in preferred:
            candidates.append((preferred[lowered], name, module))
        elif any(value in lowered or value in class_name for value in marker):
            candidates.append((20, name, module))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        return None
    return candidates[0][2]


def _first_tensor(value: Any, torch: Any) -> Any | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item, torch)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item, torch)
            if tensor is not None:
                return tensor
    return None


def _capture_hook(captures: dict[str, Any], key: str, torch: Any) -> Callable[..., None]:
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if key in captures:
            return
        tensor = _first_tensor(output, torch)
        if tensor is not None:
            captures[key] = tensor.detach()

    return hook


def configure_determinism(torch: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A hosting process may already have initialized its one inter-op pool.
        pass
    torch.use_deterministic_algorithms(True)


def build_inputs(
    model_dir: Path,
    prompts: list[str],
    token_ids: list[int] | None,
    config: dict[str, Any],
    torch: Any,
    AutoTokenizer: Any,
) -> tuple[Any, Any]:
    if token_ids is not None:
        vocab_size = config.get("vocab_size")
        if type(vocab_size) is int and any(value >= vocab_size for value in token_ids):
            raise SkillError(f"--token-ids values must be below config vocab_size={vocab_size}")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return input_ids, torch.ones_like(input_ids)

    validate_tokenizer_json_files(model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise SkillError(
            "could not load a built-in local tokenizer; provide complete tokenizer files "
            f"or use --token-ids: {exc}"
        ) from exc
    if len(prompts) > 1:
        if tokenizer.pad_token_id is None:
            raise SkillError("tokenizer has no pad token; use one prompt or --token-ids")
        tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=len(prompts) > 1,
            add_special_tokens=True,
        )
    except Exception as exc:
        raise SkillError(f"could not tokenize local prompt fixture: {exc}") from exc
    input_ids = encoded.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise SkillError("local tokenizer did not produce rank-2 input_ids")
    if input_ids.shape[1] > MAX_TOKEN_IDS:
        raise SkillError(f"tokenized prompt exceeds the {MAX_TOKEN_IDS}-token limit")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def capture_tensors(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    config: dict[str, Any],
    generate_steps: int,
    torch: Any,
) -> dict[str, Any]:
    layers_name, layers = _find_decoder_layers(model, _expected_layer_count(config), torch)
    final_norm = _find_final_norm(model, layers_name)
    captures: dict[str, Any] = {}
    handles: list[Any] = []
    try:
        try:
            embedding = model.get_input_embeddings()
        except (AttributeError, NotImplementedError) as exc:
            raise SkillError("model does not expose a standard input embedding module") from exc
        if embedding is None:
            raise SkillError("model does not expose a standard input embedding module")
        handles.append(embedding.register_forward_hook(_capture_hook(captures, "embed", torch)))
        handles.append(final_norm.register_forward_hook(_capture_hook(captures, "final_norm", torch)))
        for index, layer in enumerate(layers):
            handles.append(
                layer.register_forward_hook(
                    _capture_hook(captures, f"layer.{index}.hidden", torch)
                )
            )
            attention = _find_branch_module(layer, "attention")
            if attention is not None:
                handles.append(
                    attention.register_forward_hook(
                        _capture_hook(captures, f"layer.{index}.attention", torch)
                    )
                )
            mlp = _find_branch_module(layer, "mlp")
            if mlp is not None:
                handles.append(
                    mlp.register_forward_hook(
                        _capture_hook(captures, f"layer.{index}.mlp", torch)
                    )
                )

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states and torch.is_tensor(hidden_states[0]):
            captures["embed"] = hidden_states[0].detach()
        logits = getattr(outputs, "logits", None)
        if logits is None or not torch.is_tensor(logits):
            raise SkillError("model forward did not return logits")
        captures["logits"] = logits.detach()
    finally:
        for handle in reversed(handles):
            handle.remove()

    required = {"embed", "final_norm", "logits"}
    required.update(f"layer.{index}.hidden" for index in range(len(layers)))
    missing = sorted(required - set(captures))
    if missing:
        raise SkillError("model hooks did not capture required tensors: " + ", ".join(missing))

    generated: list[Any] = []
    sequence = input_ids
    mask = attention_mask
    with torch.inference_mode():
        for _ in range(generate_steps):
            outputs = model(
                input_ids=sequence,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(outputs, "logits", None)
            if logits is None or not torch.is_tensor(logits):
                raise SkillError("model generation forward did not return logits")
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_token)
            sequence = torch.cat((sequence, next_token[:, None]), dim=1)
            mask = torch.cat(
                (
                    mask,
                    torch.ones(
                        (mask.shape[0], 1),
                        dtype=mask.dtype,
                        device=mask.device,
                    ),
                ),
                dim=1,
            )
    if generated:
        generated_ids = torch.stack(generated, dim=1)
    else:
        generated_ids = torch.empty(
            (input_ids.shape[0], 0),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    return {
        "input_ids": input_ids.detach(),
        "attention_mask": attention_mask.detach(),
        **captures,
        "generated_token_ids": generated_ids.detach(),
    }


def to_numpy_tensors(tensors: dict[str, Any], *, keep_dtype: bool, np: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        if tensor.is_floating_point() and not keep_dtype:
            tensor = tensor.float()
        try:
            array = tensor.numpy()
        except (TypeError, RuntimeError) as exc:
            policy = " with --keep-dtype" if keep_dtype else ""
            raise SkillError(
                f"tensor {key} dtype {tensor.dtype} cannot be represented in NumPy{policy}"
            ) from exc
        if array.dtype.hasobject:
            raise SkillError(f"tensor {key} produced an unsafe object dtype")
        arrays[key] = np.ascontiguousarray(array)
    return arrays


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


def _raw_npz_bound(arrays: dict[str, Any]) -> int:
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
            raise SkillError(
                f"NPZ output exceeds --max-output-mb: {size} > {cap_bytes} bytes"
            )
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_staged_json(path: Path, payload: dict[str, Any]) -> Path:
    try:
        raw = (
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SkillError(f"oracle manifest is not strict JSON: {exc}") from exc
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


def _validate_capture_args(args: argparse.Namespace) -> tuple[list[str], list[int] | None]:
    if args.generate_steps < 0:
        raise SkillError("--generate-steps must be a non-negative integer")
    if not math.isfinite(args.max_output_mb) or args.max_output_mb < 0:
        raise SkillError("--max-output-mb must be a finite non-negative number")
    if args.model is None:
        raise SkillError("model directory is required unless --validate-manifest is used")
    if args.output is None:
        raise SkillError("--output is required for oracle capture")
    token_ids = parse_token_ids(args.token_ids) if args.token_ids else None
    prompts = collect_prompts(args.prompt, args.prompts_file)
    if token_ids is not None and prompts:
        raise SkillError("--token-ids cannot be combined with --prompt or --prompts-file")
    if token_ids is None and not prompts:
        raise SkillError("provide --prompt, --prompts-file, or --token-ids")
    return prompts, token_ids


def _run_manifest_validation(path_value: str) -> int:
    payload = read_bounded_json(
        Path(path_value),
        MAX_MANIFEST_JSON_BYTES,
        label="oracle manifest",
    )
    manifest = validate_manifest(payload)
    print(json.dumps({"ok": True, "schema_version": manifest["schema_version"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_manifest:
            conflicting = any((
                args.model is not None,
                args.output is not None,
                args.manifest is not None,
                bool(args.prompt),
                bool(args.prompts_file),
                bool(args.token_ids),
                args.keep_dtype,
                args.generate_steps != DEFAULT_GENERATE_STEPS,
                args.seed != 0,
                args.max_output_mb != DEFAULT_MAX_OUTPUT_MB,
            ))
            if conflicting:
                raise SkillError("--validate-manifest cannot be combined with capture arguments")
            return _run_manifest_validation(args.validate_manifest)

        prompts, token_ids = _validate_capture_args(args)
        model_dir, config, model_record = inspect_local_model(args.model)

        output = validate_output_path(Path(args.output), label="NPZ output")
        if output.suffix.lower() != ".npz":
            raise SkillError("--output must end with .npz")
        manifest = validate_output_path(
            Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json"),
            label="manifest output",
        )
        if output == manifest:
            raise SkillError("NPZ and manifest outputs must be different paths")
        protected = {_absolute_lexical(model_dir / "config.json")}
        protected.update(
            _absolute_lexical(path)
            for path in _model_root_entries(model_dir)
            if _is_weight_file(path)
        )
        if output in protected or manifest in protected:
            raise SkillError("oracle outputs must not overwrite config.json or model weights")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            import torch
        except ImportError as exc:
            raise SkillError(
                "capture_oracle.py requires optional package 'torch'; "
                "install it with python3 -m pip install torch"
            ) from exc
        try:
            import numpy as np
        except ImportError as exc:
            raise SkillError(
                "capture_oracle.py requires optional package 'numpy'; "
                "install it with python3 -m pip install numpy"
            ) from exc
        try:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SkillError(
                "capture_oracle.py requires optional package 'transformers'; "
                "install it with python3 -m pip install transformers"
            ) from exc
        transformers.logging.disable_progress_bar()

        configure_determinism(torch, np, args.seed)
        try:
            hf_config = AutoConfig.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise SkillError(
                "model architecture is not built into Transformers; "
                "capture_oracle.py refuses remote code"
            ) from exc
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                config=hf_config,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            message = str(exc)
            if "remote code" in message.lower() or "trust_remote_code" in message.lower():
                raise SkillError(
                    "model architecture requires remote code; capture_oracle.py supports only "
                    "built-in Transformers architectures"
                ) from exc
            raise SkillError(f"could not load built-in local causal language model: {exc}") from exc
        model.train(False)

        input_ids, attention_mask = build_inputs(
            model_dir,
            prompts,
            token_ids,
            config,
            torch,
            AutoTokenizer,
        )
        captured = capture_tensors(
            model,
            input_ids,
            attention_mask,
            config,
            args.generate_steps,
            torch,
        )
        arrays = to_numpy_tensors(captured, keep_dtype=args.keep_dtype, np=np)
        cap_bytes = int(args.max_output_mb * 1024 * 1024)
        bounded_size = _raw_npz_bound(arrays)
        if bounded_size > cap_bytes:
            raise SkillError(
                f"captured tensor payload exceeds --max-output-mb: "
                f"{bounded_size} > {cap_bytes} bytes"
            )

        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "model": model_record,
            "capture": {
                "prompts": prompts if token_ids is None else None,
                "token_ids": token_ids,
                "generate_steps": args.generate_steps,
                "seed": args.seed,
                "dtype_policy": "keep" if args.keep_dtype else "float32",
            },
            "tensors": tensor_inventory(arrays),
            "libraries": {
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        }
        validate_manifest(manifest_payload)

        prepare_output_parent(output, label="NPZ output")
        prepare_output_parent(manifest, label="manifest output")
        staged_npz: Path | None = None
        staged_manifest: Path | None = None
        try:
            staged_npz = _write_staged_npz(output, arrays, cap_bytes, np)
            staged_manifest = _write_staged_json(manifest, manifest_payload)
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

        print(json.dumps({
            "manifest": str(manifest),
            "npz": str(output),
            "ok": True,
            "tensor_count": len(arrays),
        }, sort_keys=True))
        return 0
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
