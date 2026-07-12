#!/usr/bin/env python3
"""Statically inspect a local or opt-in Hugging Face model without importing model code."""
from __future__ import annotations

import argparse
import atexit
import errno
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from _common import SkillError, atomic_write_text, bounded_files, dump_json, load_structured

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = SKILL_ROOT / "assets" / "architectures.yaml"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}

ONNX_TENSOR_TYPES = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
}
ONNX_TENSOR_TYPE_BYTES = {
    1: 4,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 4,
    7: 8,
    9: 1,
    10: 2,
    11: 8,
    12: 4,
    13: 8,
    14: 8,
    15: 16,
    16: 2,
}

GGUF_VALUE_TYPES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

GGUF_TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}

MAX_FORMAT_FILE_BYTES = 256 * 1024 * 1024
MAX_SOURCE_FORMAT_ARTIFACTS = 256
MAX_SOURCE_FORMAT_TOTAL_BYTES = 512 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
MAX_SAFETENSORS_INDEX_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_TENSORS = 1_000_000
MAX_TENSOR_DIMENSIONS = 64
MAX_TENSOR_DIM = (1 << 63) - 1
MAX_TENSOR_ELEMENTS = (1 << 63) - 1
MAX_ONNX_FIELDS_PER_MESSAGE = 100_000
MAX_ONNX_STRING_BYTES = 1024 * 1024
MAX_ONNX_NODES = 100_000
MAX_ONNX_INITIALIZERS = 100_000
MAX_ONNX_VALUE_INFOS = 100_000
MAX_ONNX_OPSETS = 1_024
MAX_ONNX_METADATA_PROPERTIES = 16_384
MAX_ONNX_EXTERNAL_DATA_ENTRIES = 64
MAX_GGUF_METADATA_ENTRIES = 65_536
MAX_GGUF_TENSORS = 1_000_000
MAX_GGUF_STRING_BYTES = 1024 * 1024
MAX_GGUF_ARRAY_VALUES = 128
MAX_GGUF_ALIGNMENT = 1024 * 1024
MAX_HF_TREE_ENTRIES = 20_000
MAX_HF_SELECTED_FILES = 5_000
MAX_HF_FILE_BYTES = 512 * 1024 * 1024
MAX_HF_TOTAL_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
GGUF_DEFAULT_ALIGNMENT = 32
STATIC_TEXT_LIMIT = 2 * 1024 * 1024
MAX_TRAVERSAL_FILES = 20_000
MAX_ARCHIVE_MEMBERS = 1_024
MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 1_000
MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
ZIP_MAX_COMMENT_BYTES = 65_535
ZIP_EOCD = struct.Struct("<4s4H2IH")
ZIP_CENTRAL_FILE_HEADER = struct.Struct("<4s6H3I5H2I")
ONNX_STATIC_COVERED_OPS = {
    "Abs", "Add", "Cast", "Concat", "Constant", "Cos", "Div", "Equal", "Erf",
    "Exp", "Flatten", "Gather", "Gemm", "LayerNormalization", "Log",
    "MatMul", "Mul", "Neg", "Pow", "ReduceMean", "Relu", "Reshape",
    "Shape", "Sigmoid", "Sin", "Slice", "Softmax", "Split", "Sqrt",
    "Sub", "Tanh", "Transpose", "Unsqueeze",
}
TENSORFLOW_STATIC_COVERED_OPS = {
    "Add", "BatchMatMulV2", "BiasAdd", "Cast", "ConcatV2", "Const",
    "Einsum", "Erf", "Exp", "GatherV2", "Identity", "MatMul", "Mul",
    "NoOp", "Pack", "Placeholder", "RealDiv", "Relu", "Reshape",
    "Rsqrt", "Shape", "Softmax", "Square", "StridedSlice", "Sub",
    "Tanh", "Transpose",
}
KERAS_STATIC_COVERED_LAYERS = {
    "Activation", "Add", "Concatenate", "Conv1D", "Conv2D", "Dense",
    "Dropout", "Embedding", "Flatten", "InputLayer", "LayerNormalization",
    "MultiHeadAttention", "Reshape", "Softmax",
}

CONFIG_NAMES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "feature_extractor_config.json",
    "model_index.json",
    "adapter_config.json",
]

ARTIFACT_IDENTITY_SCHEMA_VERSION = 1
UNACCEPTABLE_LICENSE_VALUES = {
    "",
    "all rights reserved",
    "custom",
    "n/a",
    "no license",
    "none",
    "not specified",
    "other",
    "proprietary",
    "see license",
    "unknown",
    "unlicensed",
    "unspecified",
}
LICENSE_TEXT_MARKERS = (
    "apache license",
    "gnu general public license",
    "mit license",
    "mozilla public license",
    "permission is hereby granted",
    "redistribution and use in source and binary forms",
    "terms and conditions for use, reproduction, and distribution",
    "creative commons",
    "open rail license",
    "openrail license",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statically inspect model metadata, safetensors headers, architecture signals, and risk flags."
    )
    parser.add_argument("model", help="Local model directory/file or Hugging Face repo id")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Architecture registry")
    parser.add_argument("--output", help="Write JSON report to this path; stdout if omitted")
    parser.add_argument("--markdown", help="Also write a compact Markdown report")
    parser.add_argument("--allow-network", action="store_true", help="Allow metadata-only Hugging Face download")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="With --allow-network, include safetensors files. This can download very large artifacts.",
    )
    parser.add_argument("--revision", help="Pinned Hugging Face revision")
    parser.add_argument("--max-files", type=int, default=5000, help="Maximum files to inventory")
    parser.add_argument("--hash-small-files", action="store_true", help="SHA-256 files up to 10 MiB")
    parser.add_argument(
        "--include-local-paths",
        action="store_true",
        help="Include absolute local/cache paths in reports (off by default for portable output)",
    )
    return parser.parse_args()


def _same_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_mode,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_mode,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _open_regular_nofollow(path: Path, label: str) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise SkillError("this platform does not provide O_NOFOLLOW; refusing static artifact reads")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SkillError(f"{label} must not be a symlink") from exc
        raise SkillError(f"could not open {label} without following symlinks: {exc.strerror or exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SkillError(f"{label} must be a regular file")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _verify_stable_descriptor(
    path: Path,
    descriptor: int,
    initial: os.stat_result,
    label: str,
) -> None:
    final = os.fstat(descriptor)
    try:
        current = os.lstat(path)
    except FileNotFoundError as exc:
        raise SkillError(f"{label} changed while it was being read") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or not _same_snapshot(initial, final)
        or not _same_snapshot(initial, current)
    ):
        raise SkillError(f"{label} changed while it was being read")


def _read_exact(descriptor: int, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
        if not chunk:
            raise SkillError(f"{label} changed or ended while it was being read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_regular_bytes(path: Path, limit: int, *, label: str | None = None) -> bytes:
    """Read one stable regular-file snapshot without following the leaf."""
    display = label or path.name
    descriptor, metadata = _open_regular_nofollow(path, display)
    try:
        if metadata.st_size > limit:
            raise SkillError(f"{display} exceeds size limit: {metadata.st_size} > {limit} bytes")
        data = _read_exact(descriptor, metadata.st_size, display)
        if os.read(descriptor, 1):
            raise SkillError(f"{display} changed while it was being read")
        _verify_stable_descriptor(path, descriptor, metadata, display)
        return data
    finally:
        os.close(descriptor)


def regular_file_size(path: Path, *, label: str | None = None) -> int:
    """Return size only after pinning and verifying a no-follow regular descriptor."""
    display = label or path.name
    descriptor, metadata = _open_regular_nofollow(path, display)
    try:
        _verify_stable_descriptor(path, descriptor, metadata, display)
        return metadata.st_size
    finally:
        os.close(descriptor)


def hash_regular_file(path: Path, *, label: str | None = None) -> tuple[int, str]:
    """Hash one stable regular-file snapshot without materializing its contents."""
    display = label or path.name
    descriptor, metadata = _open_regular_nofollow(path, display)
    digest = hashlib.sha256()
    try:
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise SkillError(f"{display} changed or ended while it was being hashed")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SkillError(f"{display} changed while it was being hashed")
        _verify_stable_descriptor(path, descriptor, metadata, display)
        return metadata.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def read_regular_prefix(path: Path, limit: int, *, label: str | None = None) -> bytes:
    """Read at most ``limit`` bytes from one stable no-follow regular descriptor."""
    display = label or path.name
    descriptor, metadata = _open_regular_nofollow(path, display)
    try:
        data = _read_exact(descriptor, min(metadata.st_size, limit), display)
        _verify_stable_descriptor(path, descriptor, metadata, display)
        return data
    finally:
        os.close(descriptor)


def load_json_unique(data: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SkillError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    except SkillError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"invalid JSON in {label}: {exc}") from exc


def huggingface_allow_patterns(download_weights: bool) -> list[str]:
    patterns = [
        "*.json", "*.md", "*.txt", "LICENSE*", "NOTICE*", "*.model", "*.tiktoken",
        "*.jinja", "*.py", "*.yaml", "*.yml", "*.safetensors.index.json",
    ]
    if download_weights:
        patterns.append("*.safetensors")
    return patterns


def preflight_huggingface_snapshot(
    api: Any,
    repo_id: str,
    revision: str | None,
    allow_patterns: list[str],
) -> tuple[str, dict[str, int]]:
    """Resolve one commit and bound the exact Hub tree before snapshot download."""
    info = api.model_info(repo_id=repo_id, revision=revision)
    resolved_revision = str(getattr(info, "sha", "") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", resolved_revision):
        raise SkillError("Hugging Face metadata did not resolve to a pinned 40-character commit")
    selected_files = 0
    selected_bytes = 0
    tree_entries = 0
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        revision=resolved_revision,
        recursive=True,
        expand=True,
    ):
        tree_entries += 1
        if tree_entries > MAX_HF_TREE_ENTRIES:
            raise SkillError(
                f"Hugging Face tree entry limit exceeded: more than {MAX_HF_TREE_ENTRIES}"
            )
        path = str(getattr(entry, "path", "") or "")
        entry_type = str(getattr(entry, "type", "") or "")
        if entry_type not in {"file", ""} or not any(fnmatch.fnmatch(path, pattern) for pattern in allow_patterns):
            continue
        size = getattr(entry, "size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SkillError(f"Hugging Face tree metadata is missing a valid size for {path!r}")
        if size > MAX_HF_FILE_BYTES:
            raise SkillError(
                f"Hugging Face file exceeds download quota: {path!r} is {size} bytes"
            )
        selected_files += 1
        selected_bytes += size
        if selected_files > MAX_HF_SELECTED_FILES:
            raise SkillError(
                f"Hugging Face selected-file limit exceeded: more than {MAX_HF_SELECTED_FILES}"
            )
        if selected_bytes > MAX_HF_TOTAL_DOWNLOAD_BYTES:
            raise SkillError(
                f"Hugging Face aggregate download quota exceeded: {selected_bytes} bytes"
            )
    return resolved_revision.lower(), {
        "tree_entries": tree_entries,
        "selected_files": selected_files,
        "selected_bytes": selected_bytes,
    }


def resolve_model(
    value: str,
    allow_network: bool,
    revision: str | None,
    download_weights: bool,
    include_local_paths: bool = False,
) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    if path.exists():
        resolved = path.resolve()
        display = str(resolved) if include_local_paths else (resolved.name or ".")
        return resolved, {"kind": "local", "input": display, "revision": revision}
    if not allow_network:
        raise SkillError(
            f"{value!r} does not exist locally. Re-run with --allow-network to fetch only inspectable metadata."
        )
    try:
        from huggingface_hub import HfApi, snapshot_download  # type: ignore
    except ImportError as exc:
        raise SkillError("huggingface_hub is required for --allow-network") from exc

    allow_patterns = huggingface_allow_patterns(download_weights)
    resolved_revision, preflight = preflight_huggingface_snapshot(
        HfApi(),
        value,
        revision,
        allow_patterns,
    )
    materialized_dir = Path(tempfile.mkdtemp(prefix="mlx-model-intake-"))
    try:
        local = snapshot_download(
            repo_id=value,
            revision=resolved_revision,
            allow_patterns=allow_patterns,
            local_dir=str(materialized_dir),
        )
    except BaseException:
        shutil.rmtree(materialized_dir, ignore_errors=True)
        raise
    atexit.register(shutil.rmtree, materialized_dir, ignore_errors=True)
    resolved_local = Path(local).resolve()
    source: dict[str, Any] = {
        "kind": "huggingface",
        "input": value,
        "requested_revision": revision,
        "revision": resolved_revision,
        "downloaded_weights": download_weights,
        "preflight": preflight,
    }
    if include_local_paths:
        source["local_cache_path"] = str(resolved_local)
    return resolved_local, source


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = load_json_unique(
            read_regular_bytes(path, STATIC_TEXT_LIMIT, label=path.name),
            path.name,
        )
    except (OSError, SkillError):
        return None
    return value if isinstance(value, dict) else None


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    label = path.name
    descriptor, file_stat = _open_regular_nofollow(path, label)
    try:
        raw_len = _read_exact(descriptor, 8, label) if file_stat.st_size >= 8 else b""
        if len(raw_len) != 8:
            raise SkillError(f"invalid safetensors header in {label}: missing length")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise SkillError(f"suspicious safetensors header length {header_len} in {label}")
        if 8 + header_len > file_stat.st_size:
            raise SkillError(f"truncated safetensors header in {label}")
        raw = _read_exact(descriptor, header_len, label)
        _verify_stable_descriptor(path, descriptor, file_stat, label)
    finally:
        os.close(descriptor)

    header = load_json_unique(raw, f"safetensors header {label}")
    if not isinstance(header, dict):
        raise SkillError(f"safetensors header in {label} is not a mapping")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise SkillError(f"safetensors metadata in {label} must map strings to strings")
    if len(header) > MAX_SAFETENSORS_TENSORS:
        raise SkillError(
            f"safetensors tensor-count limit exceeded in {label}: "
            f"{len(header)} > {MAX_SAFETENSORS_TENSORS}"
        )

    payload_size = file_stat.st_size - 8 - header_len
    intervals: list[tuple[int, int, str]] = []
    for key, spec in header.items():
        if not isinstance(key, str) or not key:
            raise SkillError(f"safetensors tensor key in {label} must be a non-empty string")
        if not isinstance(spec, dict):
            raise SkillError(f"safetensors tensor record {key!r} in {label} is not a mapping")
        if set(spec) != {"dtype", "shape", "data_offsets"}:
            raise SkillError(
                f"safetensors tensor record {key!r} in {label} must contain only "
                "dtype, shape, and data_offsets"
            )
        dtype = spec.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise SkillError(f"safetensors tensor {key!r} in {label} has invalid dtype {dtype!r}")
        shape = spec.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) > MAX_TENSOR_DIMENSIONS
            or not all(
                isinstance(dim, int)
                and not isinstance(dim, bool)
                and 0 <= dim <= MAX_TENSOR_DIM
                for dim in shape
            )
        ):
            raise SkillError(f"safetensors tensor {key!r} in {label} has invalid shape")
        count = product(shape)
        if count > MAX_TENSOR_ELEMENTS:
            raise SkillError(f"safetensors tensor {key!r} in {label} exceeds element-count limit")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) and not isinstance(offset, bool) for offset in offsets)
        ):
            raise SkillError(f"safetensors tensor {key!r} in {label} has invalid data_offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise SkillError(f"safetensors tensor {key!r} in {label} has out-of-range data_offsets")
        expected_bytes = count * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise SkillError(
                f"safetensors tensor {key!r} in {label} byte span does not match dtype and shape"
            )
        intervals.append((start, end, key))

    cursor = 0
    for start, end, key in sorted(intervals, key=lambda item: (item[0], item[1], item[2])):
        if start != cursor:
            raise SkillError(
                f"safetensors tensor {key!r} in {label} creates overlapping or non-contiguous data offsets"
            )
        cursor = end
    if cursor != payload_size:
        raise SkillError(f"safetensors payload size in {label} does not match declared tensor offsets")
    return header, metadata


def product(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def read_safetensors_index(path: Path) -> dict[str, Any]:
    value = load_json_unique(
        read_regular_bytes(path, MAX_SAFETENSORS_INDEX_BYTES, label=path.name),
        path.name,
    )
    if not isinstance(value, dict):
        raise SkillError(f"safetensors shard index {path.name} is not a mapping")
    if set(value) - {"metadata", "weight_map"}:
        raise SkillError(f"safetensors shard index {path.name} contains unsupported top-level keys")
    metadata = value.get("metadata", {})
    weight_map = value.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict) or not weight_map:
        raise SkillError(f"safetensors shard index {path.name} has invalid metadata or weight_map")
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(shard, str) or not shard:
            raise SkillError(f"safetensors shard index {path.name} has an invalid weight_map entry")
        shard_path = PurePosixPath(shard)
        if shard_path.is_absolute() or ".." in shard_path.parts or len(shard_path.parts) != 1:
            raise SkillError(
                f"safetensors shard index {path.name} has unsafe shard path {shard!r}"
            )
        if not shard.endswith(".safetensors"):
            raise SkillError(
                f"safetensors shard index {path.name} maps {key!r} to a non-safetensors file"
            )
    total_size = metadata.get("total_size")
    if total_size is not None and (
        not isinstance(total_size, int) or isinstance(total_size, bool) or total_size < 0
    ):
        raise SkillError(f"safetensors shard index {path.name} has invalid metadata.total_size")
    return value


def inventory(root: Path, max_files: int, hash_small: bool) -> tuple[list[dict[str, Any]], bool]:
    paths, truncated = bounded_files(root, max_files)
    records: list[dict[str, Any]] = []
    for path in paths[:max_files]:
        display = str(path.relative_to(root) if root.is_dir() else path.name)
        size = regular_file_size(path, label=display)
        rec: dict[str, Any] = {
            "path": display,
            "size_bytes": size,
            "suffix": path.suffix.lower(),
        }
        if hash_small and size <= 10 * 1024 * 1024:
            data = read_regular_bytes(path, 10 * 1024 * 1024, label=display)
            rec["size_bytes"] = len(data)
            rec["sha256"] = hashlib.sha256(data).hexdigest()
        records.append(rec)
    return records, truncated


def _protected_model_artifact(relative: str) -> bool:
    path = PurePosixPath(relative)
    name = path.name.lower()
    config_names = {value.lower() for value in CONFIG_NAMES}
    protected_suffixes = {
        ".bin", ".ckpt", ".gguf", ".h5", ".keras", ".mlmodel", ".model",
        ".msgpack", ".onnx", ".pickle", ".pkl", ".pt", ".pth",
        ".safetensors", ".tiktoken",
    }
    return (
        name in config_names
        or name in {"pyproject.toml", "requirements.txt", "setup.py"}
        or name.endswith(".safetensors.index.json")
        or name.startswith((
            "copying", "license", "model_card", "notice", "readme", "tokenizer",
            "processor", "preprocessor", "special_tokens", "vocab", "merges",
        ))
        or path.suffix.lower() in protected_suffixes
    )


def _is_prior_json_report(path: Path) -> bool:
    try:
        prefix = read_regular_prefix(path, 256 * 1024, label=path.name).decode("utf-8")
    except (OSError, SkillError, UnicodeDecodeError):
        return False
    return (
        re.match(
            r'\A\{\s*"schema_version":\s*1,\s*"generated_at":\s*"[^"]+",\s*'
            r'"inspection_mode":\s*"static-no-model-import",',
            prefix,
        )
        is not None
        and '"artifact_identity": {' in prefix
        and '"algorithm": "sha256-tree-v1"' in prefix
    )


def _is_prior_markdown_report(path: Path) -> bool:
    try:
        prefix = read_regular_prefix(path, 256 * 1024, label=path.name).decode("utf-8")
    except (OSError, SkillError, UnicodeDecodeError):
        return False
    return (
        prefix.startswith("# Static model inspection\n\n")
        and "- Artifact identity: `" in prefix
        and "- License evidence: `" in prefix
        and "## Architecture routing decision" in prefix
    )


def exclude_generated_reports(
    root: Path,
    file_records: list[dict[str, Any]],
    output_path: str | None,
    markdown_path: str | None,
    model_input: str | None = None,
) -> list[dict[str, Any]]:
    """Validate in-tree report ownership before excluding prior generated reports."""
    base_path = root if root.is_dir() else root.parent
    canonical_lexical_base = Path(os.path.abspath(base_path.expanduser()))
    base = canonical_lexical_base.resolve()
    lexical_bases = [canonical_lexical_base]
    if model_input:
        supplied_model = Path(os.path.abspath(Path(model_input).expanduser()))
        try:
            supplied_resolved = supplied_model.resolve()
        except (OSError, RuntimeError):
            supplied_resolved = None
        if supplied_resolved == root:
            supplied_base = supplied_model if root.is_dir() else supplied_model.parent
            if supplied_base not in lexical_bases:
                lexical_bases.append(supplied_base)
    destinations: list[tuple[str, Path]] = []
    for kind, raw_path in (("output", output_path), ("markdown", markdown_path)):
        if raw_path:
            destinations.append((kind, Path(os.path.abspath(Path(raw_path).expanduser()))))

    resolved_destinations: list[tuple[str, Path]] = []
    for kind, path in destinations:
        try:
            resolved_destinations.append((kind, path.resolve()))
        except (OSError, RuntimeError) as exc:
            raise SkillError(f"--{kind} path contains an invalid link or cycle: {path}") from exc
    for index, (kind, path) in enumerate(resolved_destinations):
        for other_kind, other in resolved_destinations[index + 1:]:
            if path == other or path in other.parents or other in path.parents:
                raise SkillError(
                    f"--{kind} and --{other_kind} paths must be distinct non-nested files"
                )

    inventoried_paths = {
        str(record.get("path"))
        for record in file_records
        if isinstance(record.get("path"), str)
    }
    excluded: set[str] = set()
    for (kind, lexical), (_, resolved) in zip(destinations, resolved_destinations):
        lexical_base: Path | None = None
        lexical_relative: Path | None = None
        for candidate_base in lexical_bases:
            try:
                lexical_relative = lexical.relative_to(candidate_base)
                lexical_base = candidate_base
                break
            except ValueError:
                continue
        lexical_inside = lexical_relative is not None
        try:
            resolved_relative = resolved.relative_to(base)
            resolved_inside = True
        except ValueError:
            resolved_relative = None
            resolved_inside = False

        if lexical_inside != resolved_inside:
            raise SkillError(f"--{kind} path must not cross the model boundary through a link")
        if not resolved_inside:
            continue
        assert lexical_base is not None and lexical_relative is not None
        assert resolved_relative is not None
        relative = PurePosixPath(resolved_relative.as_posix()).as_posix()

        cursor = lexical_base
        for part in lexical_relative.parts[:-1]:
            cursor /= part
            if cursor.is_symlink():
                raise SkillError(f"--{kind} path must not traverse a link inside the model tree")
            if cursor.exists() and not cursor.is_dir():
                raise SkillError(f"--{kind} parent is not a directory inside the model tree")

        if not (lexical.exists() or lexical.is_symlink()):
            excluded.add(relative)
            continue
        node = os.lstat(lexical)
        if not stat.S_ISREG(node.st_mode):
            raise SkillError(f"--{kind} must not target a directory, link, or special node")
        if _protected_model_artifact(relative):
            raise SkillError(f"--{kind} must not overwrite model artifact {relative}")
        if relative not in inventoried_paths:
            raise SkillError(f"--{kind} must not overwrite an unverified in-tree file {relative}")
        is_prior_report = (
            _is_prior_json_report(lexical)
            if kind == "output"
            else _is_prior_markdown_report(lexical)
        )
        if not is_prior_report:
            raise SkillError(
                f"--{kind} may overwrite only a canonical prior inspector report inside the model tree"
            )
        excluded.add(relative)

    return [record for record in file_records if record.get("path") not in excluded]


def build_artifact_identity(
    root: Path,
    file_records: list[dict[str, Any]],
    truncated: bool,
    *,
    extra_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Build a portable, content-addressed manifest for every intake input file."""
    base = root if root.is_dir() else root.parent
    paths_by_name: dict[str, Path] = {}
    errors: list[str] = []

    for record in file_records:
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append("inventory contains a file without a normalized relative path")
            continue
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw_path:
            errors.append(f"inventory path is not portable: {raw_path!r}")
            continue
        paths_by_name[relative.as_posix()] = base.joinpath(*relative.parts)

    for path in extra_paths or []:
        try:
            relative_path = path.relative_to(base)
        except ValueError:
            errors.append(f"identity input escapes artifact root: {path.name}")
            continue
        relative = PurePosixPath(relative_path.as_posix())
        raw_path = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw_path:
            errors.append(f"identity input path is not portable: {raw_path!r}")
            continue
        paths_by_name[raw_path] = path

    manifest: list[dict[str, Any]] = []
    for relative, path in sorted(paths_by_name.items()):
        try:
            size_bytes, digest = hash_regular_file(path, label=relative)
        except (OSError, SkillError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        manifest.append({
            "path": relative,
            "size_bytes": size_bytes,
            "sha256": digest,
        })

    if truncated:
        errors.append("file inventory is truncated; the artifact manifest is incomplete")
    complete = not errors and len(manifest) == len(paths_by_name)
    fingerprint = None
    if complete:
        canonical = json.dumps(
            {
                "schema_version": ARTIFACT_IDENTITY_SCHEMA_VERSION,
                "files": manifest,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = "sha256:" + hashlib.sha256(canonical).hexdigest()

    return {
        "schema_version": ARTIFACT_IDENTITY_SCHEMA_VERSION,
        "algorithm": "sha256-tree-v1",
        "status": "verified" if complete else "incomplete",
        "immutable": complete,
        "fingerprint": fingerprint,
        "file_count": len(manifest),
        "total_bytes": sum(int(record["size_bytes"]) for record in manifest),
        "manifest": manifest,
        "errors": errors,
    }


def load_configs(root: Path) -> dict[str, dict[str, Any]]:
    base = root if root.is_dir() else root.parent
    configs: dict[str, dict[str, Any]] = {}
    for name in CONFIG_NAMES:
        path = base / name
        if path.exists():
            value = read_json(path)
            if value is not None:
                configs[name] = value
    return configs


def inspect_tensors(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if root.is_file() and root.suffix == ".safetensors":
        files = [root]
        index_files: list[Path] = []
    elif root.is_dir():
        walked, truncated = bounded_files(root, MAX_TRAVERSAL_FILES)
        if truncated:
            raise SkillError(f"model traversal exceeds {MAX_TRAVERSAL_FILES} files")
        files = [path for path in walked if path.parent == root and path.suffix == ".safetensors"]
        index_files = [
            path
            for path in walked
            if path.parent == root and path.name.endswith(".safetensors.index.json")
        ]
    else:
        files = []
        index_files = []
    tensors: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    errors: list[str] = []
    header_keys: dict[str, set[str]] = {}
    tensor_owner: dict[str, str] = {}
    for file in files:
        try:
            header, file_meta = read_safetensors_header(file)
        except SkillError as exc:
            errors.append(str(exc))
            continue
        header_keys[file.name] = set(header)
        if file_meta:
            metadata[file.name] = file_meta
        for key, spec in header.items():
            shape = spec.get("shape")
            dtype = spec.get("dtype")
            assert isinstance(shape, list) and isinstance(dtype, str)
            if key in tensor_owner:
                errors.append(
                    f"duplicate safetensors tensor key {key!r} in {tensor_owner[key]} and {file.name}"
                )
                continue
            tensor_owner[key] = file.name
            count = product(shape)
            tensors.append({
                "key": key,
                "shape": shape,
                "dtype": dtype,
                "parameters": count,
                "estimated_bytes": count * DTYPE_BYTES.get(str(dtype), 0),
                "file": file.name,
            })

    if len(index_files) > 1:
        errors.append(
            "multiple safetensors shard indexes were found: "
            + ", ".join(path.name for path in sorted(index_files))
        )
    elif index_files:
        index_path = index_files[0]
        try:
            index = read_safetensors_index(index_path)
            weight_map = index["weight_map"]
            assert isinstance(weight_map, dict)
            discovered_files = {file.name for file in files}
            mapped_files = {str(value) for value in weight_map.values()}
            missing_files = sorted(mapped_files - discovered_files)
            unexpected_files = sorted(discovered_files - mapped_files)
            if missing_files:
                errors.append(
                    f"safetensors shard index {index_path.name} references missing shards: "
                    + ", ".join(missing_files)
                )
            if unexpected_files:
                errors.append(
                    f"safetensors shard index {index_path.name} does not account for shards: "
                    + ", ".join(unexpected_files)
                )
            for key, shard_value in weight_map.items():
                shard = str(shard_value)
                if shard not in header_keys:
                    errors.append(
                        f"safetensors shard index {index_path.name} maps {key!r} to unreadable shard {shard}"
                    )
                elif key not in header_keys[shard]:
                    errors.append(
                        f"safetensors shard index {index_path.name} maps {key!r} to {shard}, "
                        "but that shard header does not contain the tensor"
                    )
                elif tensor_owner.get(key) != shard:
                    errors.append(
                        f"safetensors shard index {index_path.name} maps {key!r} to the wrong shard"
                    )
            unindexed_keys = sorted(set(tensor_owner) - set(weight_map))
            if unindexed_keys:
                errors.append(
                    f"safetensors shard index {index_path.name} omits tensors: "
                    + ", ".join(unindexed_keys[:20])
                )
            total_size = index.get("metadata", {}).get("total_size")
            if total_size is not None:
                indexed_bytes = sum(
                    tensor["estimated_bytes"]
                    for tensor in tensors
                    if tensor["key"] in weight_map
                )
                if total_size != indexed_bytes:
                    errors.append(
                        f"safetensors shard index {index_path.name} metadata.total_size "
                        f"{total_size} does not match indexed tensor bytes {indexed_bytes}"
                    )
        except SkillError as exc:
            errors.append(str(exc))
    elif len(files) > 1:
        errors.append("multiple safetensors shards were found without a shard index")
    tensors.sort(key=lambda x: x["key"])
    return tensors, metadata, errors


def read_varint(data: bytes, pos: int, limit: int | None = None) -> tuple[int, int]:
    limit = len(data) if limit is None else limit
    shift = 0
    value = 0
    while pos < limit:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise SkillError("protobuf varint is too long")
    raise SkillError("truncated protobuf varint")


def iter_proto_fields(data: bytes) -> Any:
    pos = 0
    limit = len(data)
    field_count = 0
    while pos < limit:
        field_count += 1
        if field_count > MAX_ONNX_FIELDS_PER_MESSAGE:
            raise SkillError(
                f"protobuf message field-count limit exceeded: more than {MAX_ONNX_FIELDS_PER_MESSAGE}"
            )
        key, pos = read_varint(data, pos, limit)
        field_no = key >> 3
        wire_type = key & 0x7
        if field_no <= 0:
            raise SkillError("protobuf field number must be positive")
        if wire_type == 0:
            value, pos = read_varint(data, pos, limit)
            yield field_no, wire_type, value
        elif wire_type == 1:
            if pos + 8 > limit:
                raise SkillError("truncated protobuf fixed64")
            yield field_no, wire_type, data[pos:pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = read_varint(data, pos, limit)
            end = pos + length
            if end > limit:
                raise SkillError("truncated protobuf length-delimited field")
            yield field_no, wire_type, data[pos:end]
            pos = end
        elif wire_type == 5:
            if pos + 4 > limit:
                raise SkillError("truncated protobuf fixed32")
            yield field_no, wire_type, data[pos:pos + 4]
            pos += 4
        else:
            raise SkillError(f"unsupported protobuf wire type {wire_type}")


def decode_proto_string(value: bytes) -> str:
    if len(value) > MAX_ONNX_STRING_BYTES:
        raise SkillError(
            f"ONNX string exceeds {MAX_ONNX_STRING_BYTES}-byte static parser limit"
        )
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError("ONNX string is not valid UTF-8") from exc


def parse_onnx_value_info(data: bytes) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            info["name"] = decode_proto_string(value)
        elif field_no == 2 and wire_type == 2:
            info.update(parse_onnx_type(value))
    return info


def parse_onnx_type(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            for tensor_field, tensor_wire, tensor_value in iter_proto_fields(value):
                if tensor_field == 1 and tensor_wire == 0:
                    if tensor_value not in ONNX_TENSOR_TYPES:
                        raise SkillError(f"ONNX value info uses unknown data type {tensor_value}")
                    result["dtype"] = ONNX_TENSOR_TYPES.get(tensor_value, f"UNKNOWN_{tensor_value}")
                elif tensor_field == 2 and tensor_wire == 2:
                    result["shape"] = parse_onnx_shape(tensor_value)
    return result


def parse_onnx_shape(data: bytes) -> list[Any]:
    shape: list[Any] = []
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            if len(shape) >= MAX_TENSOR_DIMENSIONS:
                raise SkillError(
                    f"ONNX tensor rank exceeds {MAX_TENSOR_DIMENSIONS} dimensions"
                )
            dim_value: int | None = None
            dim_param: str | None = None
            for dim_field, dim_wire, dim_raw in iter_proto_fields(value):
                if dim_field == 1 and dim_wire == 0:
                    dim_value = int(dim_raw)
                    if dim_value < 0 or dim_value > MAX_TENSOR_DIM:
                        raise SkillError("ONNX tensor dimension is outside the supported range")
                elif dim_field == 2 and dim_wire == 2:
                    dim_param = decode_proto_string(dim_raw)
            if dim_value is not None and dim_param is not None:
                raise SkillError("ONNX tensor dimension declares both dim_value and dim_param")
            shape.append(dim_value if dim_value is not None else dim_param)
    return shape


def parse_onnx_tensor(data: bytes) -> dict[str, Any]:
    tensor: dict[str, Any] = {"shape": [], "external_data": []}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 0:
            if len(tensor["shape"]) >= MAX_TENSOR_DIMENSIONS:
                raise SkillError(
                    f"ONNX tensor rank exceeds {MAX_TENSOR_DIMENSIONS} dimensions"
                )
            if int(value) > MAX_TENSOR_DIM:
                raise SkillError("ONNX tensor dimension is outside the supported range")
            tensor["shape"].append(int(value))
        elif field_no == 1 and wire_type == 2:
            pos = 0
            while pos < len(value):
                if len(tensor["shape"]) >= MAX_TENSOR_DIMENSIONS:
                    raise SkillError(
                        f"ONNX tensor rank exceeds {MAX_TENSOR_DIMENSIONS} dimensions"
                    )
                dim, pos = read_varint(value, pos, len(value))
                if dim > MAX_TENSOR_DIM:
                    raise SkillError("ONNX tensor dimension is outside the supported range")
                tensor["shape"].append(int(dim))
        elif field_no == 2 and wire_type == 0:
            if value not in ONNX_TENSOR_TYPES:
                raise SkillError(f"ONNX tensor uses unknown data type {value}")
            tensor["dtype"] = ONNX_TENSOR_TYPES.get(value, f"UNKNOWN_{value}")
            tensor["data_type"] = int(value)
        elif field_no == 8 and wire_type == 2:
            tensor["name"] = decode_proto_string(value)
        elif field_no == 9 and wire_type == 2:
            tensor["raw_data_bytes"] = len(value)
        elif field_no == 13 and wire_type == 2:
            if len(tensor["external_data"]) >= MAX_ONNX_EXTERNAL_DATA_ENTRIES:
                raise SkillError(
                    "ONNX external-data entry-count limit exceeded for one tensor"
                )
            entry: dict[str, Any] = {}
            for kv_field, kv_wire, kv_value in iter_proto_fields(value):
                if kv_field == 1 and kv_wire == 2:
                    entry["key"] = decode_proto_string(kv_value)
                elif kv_field == 2 and kv_wire == 2:
                    entry["value"] = decode_proto_string(kv_value)
            if entry:
                key = entry.get("key")
                if key and any(item.get("key") == key for item in tensor["external_data"]):
                    raise SkillError(f"ONNX initializer has duplicate external-data key {key!r}")
                tensor["external_data"].append(entry)
        elif field_no == 14 and wire_type == 0:
            tensor["data_location"] = int(value)
    if "data_type" not in tensor:
        raise SkillError("ONNX initializer is missing data_type")
    if not isinstance(tensor.get("name"), str) or not tensor.get("name"):
        raise SkillError("ONNX initializer is missing a non-empty name")
    if tensor.get("external_data") and tensor.get("raw_data_bytes"):
        raise SkillError("ONNX initializer declares both raw and external data")
    return tensor


def parse_onnx_node(data: bytes) -> dict[str, Any]:
    node: dict[str, Any] = {"inputs": [], "outputs": []}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            if len(node["inputs"]) >= MAX_ONNX_VALUE_INFOS:
                raise SkillError("ONNX node input-count limit exceeded")
            node["inputs"].append(decode_proto_string(value))
        elif field_no == 2 and wire_type == 2:
            if len(node["outputs"]) >= MAX_ONNX_VALUE_INFOS:
                raise SkillError("ONNX node output-count limit exceeded")
            node["outputs"].append(decode_proto_string(value))
        elif field_no == 3 and wire_type == 2:
            node["name"] = decode_proto_string(value)
        elif field_no == 4 and wire_type == 2:
            node["op_type"] = decode_proto_string(value)
        elif field_no == 7 and wire_type == 2:
            node["domain"] = decode_proto_string(value)
    return node


def parse_onnx_graph(data: bytes) -> dict[str, Any]:
    graph: dict[str, Any] = {"nodes": [], "inputs": [], "outputs": [], "initializers": []}
    op_counts: dict[str, int] = {}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            if len(graph["nodes"]) >= MAX_ONNX_NODES:
                raise SkillError(f"ONNX node-count limit exceeded: more than {MAX_ONNX_NODES}")
            node = parse_onnx_node(value)
            graph["nodes"].append(node)
            op_type = node.get("op_type")
            if op_type:
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
        elif field_no == 2 and wire_type == 2:
            graph["name"] = decode_proto_string(value)
        elif field_no == 5 and wire_type == 2:
            if len(graph["initializers"]) >= MAX_ONNX_INITIALIZERS:
                raise SkillError(
                    f"ONNX initializer-count limit exceeded: more than {MAX_ONNX_INITIALIZERS}"
                )
            tensor = parse_onnx_tensor(value)
            shape = tensor.get("shape", [])
            if all(isinstance(dim, int) for dim in shape):
                parameters = product(shape)
                if parameters > MAX_TENSOR_ELEMENTS:
                    raise SkillError("ONNX initializer exceeds element-count limit")
                tensor["parameters"] = parameters
                tensor["estimated_bytes"] = parameters * ONNX_TENSOR_TYPE_BYTES.get(
                    int(tensor.get("data_type", 0)),
                    0,
                )
                if "raw_data_bytes" in tensor and tensor.get("data_location") != 1:
                    expected_bytes = tensor["estimated_bytes"]
                    if expected_bytes and tensor["raw_data_bytes"] != expected_bytes:
                        raise SkillError(
                            f"ONNX initializer {tensor.get('name', '<unknown>')} raw_data size "
                            "does not match dtype and shape"
                        )
            graph["initializers"].append(tensor)
        elif field_no == 11 and wire_type == 2:
            if len(graph["inputs"]) >= MAX_ONNX_VALUE_INFOS:
                raise SkillError("ONNX graph input-count limit exceeded")
            graph["inputs"].append(parse_onnx_value_info(value))
        elif field_no == 12 and wire_type == 2:
            if len(graph["outputs"]) >= MAX_ONNX_VALUE_INFOS:
                raise SkillError("ONNX graph output-count limit exceeded")
            graph["outputs"].append(parse_onnx_value_info(value))
    graph["node_count"] = len(graph["nodes"])
    graph["op_types"] = dict(sorted(op_counts.items()))
    graph["initializer_count"] = len(graph["initializers"])
    graph["external_initializer_count"] = sum(1 for item in graph["initializers"] if item.get("external_data") or item.get("data_location") == 1)
    return graph


def onnx_external_data_holds(graph: dict[str, Any]) -> list[str]:
    holds = []
    for tensor in graph.get("initializers", []):
        for entry in tensor.get("external_data", []):
            if entry.get("key") != "location":
                continue
            value = str(entry.get("value") or "")
            if not value:
                holds.append(f"Initializer {tensor.get('name', '<unknown>')} has empty external data location.")
            elif value.startswith(("/", "http://", "https://")) or ".." in Path(value).parts:
                holds.append(f"Initializer {tensor.get('name', '<unknown>')} has unsafe external data location {value!r}.")
            else:
                holds.append(f"Initializer {tensor.get('name', '<unknown>')} uses external data location {value!r}; do not open it during intake.")
    return holds


def static_coverage_report(
    observed_counts: dict[str, int],
    covered_names: set[str],
    *,
    subject: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    if unavailable_reason:
        return {
            "scope": "static-triage",
            "subject": subject,
            "coverage_state": "unavailable",
            "reason": unavailable_reason,
            "total_count": 0,
            "covered_count": 0,
            "unsupported_or_unclassified_count": 0,
            "covered": {},
            "unsupported_or_unclassified": {},
        }
    covered = {name: count for name, count in sorted(observed_counts.items()) if name in covered_names}
    unknown = {name: count for name, count in sorted(observed_counts.items()) if name not in covered_names}
    total = sum(observed_counts.values())
    return {
        "scope": "static-triage",
        "subject": subject,
        "coverage_state": "partial" if observed_counts else "empty",
        "total_count": total,
        "covered_count": sum(covered.values()),
        "unsupported_or_unclassified_count": sum(unknown.values()),
        "covered": covered,
        "unsupported_or_unclassified": unknown,
        "note": "Static coverage means the name maps to ordinary MLX primitives; it is not conversion support.",
    }


def inspect_onnx_file(path: Path, root: Path) -> dict[str, Any]:
    display = rel_display(path, root)
    try:
        data = read_regular_bytes(path, MAX_FORMAT_FILE_BYTES, label=display)
    except SkillError as exc:
        return {
            "format": "onnx",
            "path": display,
            "errors": [str(exc)],
            "limitations": ["Large ONNX files need a path-based parser before metadata can be reported safely."],
        }
    manifest: dict[str, Any] = {
        "format": "onnx",
        "path": display,
        "size_bytes": len(data),
        "opsets": [],
        "metadata_props": {},
        "limitations": [
            "ONNX metadata supports intake triage only; conversion support requires unsupported-op reports and MLX parity fixtures."
        ],
    }
    try:
        for field_no, wire_type, value in iter_proto_fields(data):
            if field_no == 1 and wire_type == 0:
                manifest["ir_version"] = int(value)
            elif field_no == 2 and wire_type == 2:
                manifest["producer_name"] = decode_proto_string(value)
            elif field_no == 3 and wire_type == 2:
                manifest["producer_version"] = decode_proto_string(value)
            elif field_no == 4 and wire_type == 2:
                manifest["domain"] = decode_proto_string(value)
            elif field_no == 5 and wire_type == 0:
                manifest["model_version"] = int(value)
            elif field_no == 7 and wire_type == 2:
                manifest["graph"] = parse_onnx_graph(value)
            elif field_no == 8 and wire_type == 2:
                if len(manifest["opsets"]) >= MAX_ONNX_OPSETS:
                    raise SkillError(f"ONNX opset-count limit exceeded: more than {MAX_ONNX_OPSETS}")
                opset: dict[str, Any] = {}
                for op_field, op_wire, op_value in iter_proto_fields(value):
                    if op_field == 1 and op_wire == 2:
                        opset["domain"] = decode_proto_string(op_value)
                    elif op_field == 2 and op_wire == 0:
                        opset["version"] = int(op_value)
                manifest["opsets"].append(opset)
            elif field_no == 14 and wire_type == 2:
                if len(manifest["metadata_props"]) >= MAX_ONNX_METADATA_PROPERTIES:
                    raise SkillError(
                        "ONNX metadata-property count limit exceeded: "
                        f"more than {MAX_ONNX_METADATA_PROPERTIES}"
                    )
                key = None
                item_value = None
                for prop_field, prop_wire, prop_value in iter_proto_fields(value):
                    if prop_field == 1 and prop_wire == 2:
                        key = decode_proto_string(prop_value)
                    elif prop_field == 2 and prop_wire == 2:
                        item_value = decode_proto_string(prop_value)
                if key:
                    if key in manifest["metadata_props"]:
                        raise SkillError(f"ONNX metadata contains duplicate key {key!r}")
                    manifest["metadata_props"][key] = item_value
    except SkillError as exc:
        manifest.setdefault("errors", []).append(str(exc))
    graph = manifest.get("graph", {})
    if "ir_version" not in manifest:
        manifest.setdefault("errors", []).append("ONNX ModelProto is missing ir_version")
    if not isinstance(graph, dict) or not graph:
        manifest.setdefault("errors", []).append("ONNX ModelProto is missing a graph")
    manifest["hold_conditions"] = [
        "Record unsupported opset/domain/operator coverage before porting.",
        "Require source-framework oracle and MLX parity fixtures before recommending conversion.",
    ]
    if graph and not graph.get("initializers"):
        manifest["hold_conditions"].append("No initializer metadata was found; weight coverage cannot be inferred.")
    if graph:
        coverage = static_coverage_report(graph.get("op_types", {}), ONNX_STATIC_COVERED_OPS, subject="onnx-operators")
        manifest["operator_coverage"] = coverage
        if coverage["unsupported_or_unclassified_count"]:
            names = ", ".join(coverage["unsupported_or_unclassified"])
            manifest["hold_conditions"].append(f"Unsupported or unclassified ONNX operators require review: {names}.")
        manifest["hold_conditions"].extend(onnx_external_data_holds(graph))
        for tensor in graph.get("initializers", []):
            if tensor.get("data_location") == 1 and not any(
                entry.get("key") == "location" and entry.get("value")
                for entry in tensor.get("external_data", [])
            ):
                manifest.setdefault("errors", []).append(
                    f"ONNX initializer {tensor.get('name', '<unknown>')} declares external data "
                    "without a location"
                )
    return manifest


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(data):
        raise SkillError("truncated GGUF u32")
    return struct.unpack_from("<I", data, pos)[0], pos + 4


def read_u64(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 8 > len(data):
        raise SkillError("truncated GGUF u64")
    return struct.unpack_from("<Q", data, pos)[0], pos + 8


def read_i64(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 8 > len(data):
        raise SkillError("truncated GGUF i64")
    return struct.unpack_from("<q", data, pos)[0], pos + 8


def read_gguf_string(data: bytes, pos: int) -> tuple[str, int]:
    length, pos = read_u64(data, pos)
    if length > MAX_GGUF_STRING_BYTES:
        raise SkillError(
            f"GGUF string exceeds {MAX_GGUF_STRING_BYTES}-byte static parser limit"
        )
    end = pos + length
    if end > len(data):
        raise SkillError("truncated GGUF string")
    try:
        return data[pos:end].decode("utf-8"), end
    except UnicodeDecodeError as exc:
        raise SkillError("GGUF string is not valid UTF-8") from exc


def read_gguf_value(data: bytes, pos: int, value_type: int) -> tuple[Any, int]:
    if value_type == 0:
        if pos >= len(data):
            raise SkillError("truncated GGUF uint8")
        return data[pos], pos + 1
    if value_type == 1:
        if pos >= len(data):
            raise SkillError("truncated GGUF int8")
        return struct.unpack_from("<b", data, pos)[0], pos + 1
    if value_type in {2, 3}:
        if pos + 2 > len(data):
            raise SkillError("truncated GGUF 16-bit value")
        fmt = "<H" if value_type == 2 else "<h"
        return struct.unpack_from(fmt, data, pos)[0], pos + 2
    if value_type in {4, 5, 6}:
        if pos + 4 > len(data):
            raise SkillError("truncated GGUF 32-bit value")
        fmt = {4: "<I", 5: "<i", 6: "<f"}[value_type]
        return struct.unpack_from(fmt, data, pos)[0], pos + 4
    if value_type == 7:
        if pos >= len(data):
            raise SkillError("truncated GGUF bool")
        if data[pos] not in {0, 1}:
            raise SkillError("GGUF bool value must be 0 or 1")
        return bool(data[pos]), pos + 1
    if value_type == 8:
        return read_gguf_string(data, pos)
    if value_type == 9:
        array_type, pos = read_u32(data, pos)
        if array_type not in GGUF_VALUE_TYPES or array_type == 9:
            raise SkillError(f"unsupported GGUF array element type {array_type}")
        length, pos = read_u64(data, pos)
        if length > MAX_GGUF_ARRAY_VALUES:
            raise SkillError(
                f"GGUF arrays longer than {MAX_GGUF_ARRAY_VALUES} values are not expanded by the static parser"
            )
        values = []
        for _ in range(length):
            item, pos = read_gguf_value(data, pos, array_type)
            values.append(item)
        return {"array_type": GGUF_VALUE_TYPES.get(array_type, str(array_type)), "values": values}, pos
    if value_type == 10:
        return read_u64(data, pos)
    if value_type == 11:
        return read_i64(data, pos)
    if value_type == 12:
        if pos + 8 > len(data):
            raise SkillError("truncated GGUF float64")
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    raise SkillError(f"unsupported GGUF metadata value type {value_type}")


def align_offset(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise SkillError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def quantized_tensor_type(value: str) -> bool:
    return value not in {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}


def inspect_gguf_file(path: Path, root: Path) -> dict[str, Any]:
    display = rel_display(path, root)
    try:
        data = read_regular_bytes(path, MAX_FORMAT_FILE_BYTES, label=display)
    except SkillError as exc:
        return {
            "format": "gguf",
            "path": display,
            "errors": [str(exc)],
            "limitations": ["Large GGUF files need streaming metadata parsing before all metadata can be reported safely."],
        }
    manifest: dict[str, Any] = {
        "format": "gguf",
        "path": display,
        "size_bytes": len(data),
        "metadata": {},
        "metadata_types": {},
        "tensors": [],
        "limitations": [
            "GGUF is commonly a converted inference artifact; require source provenance and quality gates before treating it as source-preserving."
        ],
    }
    try:
        if data[:4] != b"GGUF":
            raise SkillError("missing GGUF magic")
        pos = 4
        version, pos = read_u32(data, pos)
        if version not in {2, 3}:
            raise SkillError(f"unsupported GGUF version {version}")
        tensor_count, pos = read_u64(data, pos)
        metadata_count, pos = read_u64(data, pos)
        if tensor_count > MAX_GGUF_TENSORS:
            raise SkillError(
                f"GGUF tensor-count limit exceeded: {tensor_count} > {MAX_GGUF_TENSORS}"
            )
        if metadata_count > MAX_GGUF_METADATA_ENTRIES:
            raise SkillError(
                "GGUF metadata-entry count limit exceeded: "
                f"{metadata_count} > {MAX_GGUF_METADATA_ENTRIES}"
            )
        manifest.update({
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_count,
            "header": {
                "magic": "GGUF",
                "version": version,
                "endian": "little",
                "tensor_count": tensor_count,
                "metadata_kv_count": metadata_count,
            },
        })
        for _ in range(metadata_count):
            key, pos = read_gguf_string(data, pos)
            if not key:
                raise SkillError("GGUF metadata key must not be empty")
            if key in manifest["metadata"]:
                raise SkillError(f"duplicate GGUF metadata key {key!r}")
            value_type, pos = read_u32(data, pos)
            if value_type not in GGUF_VALUE_TYPES:
                raise SkillError(f"unsupported GGUF metadata value type {value_type}")
            value, pos = read_gguf_value(data, pos, value_type)
            manifest["metadata"][key] = value
            manifest["metadata_types"][key] = GGUF_VALUE_TYPES.get(value_type, str(value_type))
        manifest["metadata_keys"] = sorted(manifest["metadata"])
        names_seen: set[str] = set()
        for _ in range(tensor_count):
            name, pos = read_gguf_string(data, pos)
            if name in names_seen:
                manifest.setdefault("errors", []).append(f"duplicate GGUF tensor name {name}")
            names_seen.add(name)
            n_dims, pos = read_u32(data, pos)
            if n_dims == 0 or n_dims > MAX_TENSOR_DIMENSIONS:
                raise SkillError(
                    f"GGUF tensor {name!r} has invalid dimension count {n_dims}"
                )
            shape = []
            for _ in range(n_dims):
                dim, pos = read_u64(data, pos)
                if dim <= 0 or dim > MAX_TENSOR_DIM:
                    raise SkillError(f"GGUF tensor {name!r} has invalid dimension {dim}")
                shape.append(dim)
            tensor_type, pos = read_u32(data, pos)
            if tensor_type not in GGUF_TENSOR_TYPES:
                raise SkillError(f"GGUF tensor {name!r} has unknown type {tensor_type}")
            offset, pos = read_u64(data, pos)
            type_name = GGUF_TENSOR_TYPES.get(tensor_type, f"UNKNOWN_{tensor_type}")
            element_count = product([int(dim) for dim in shape])
            if element_count > MAX_TENSOR_ELEMENTS:
                raise SkillError(f"GGUF tensor {name!r} exceeds element-count limit")
            manifest["tensors"].append({
                "name": name,
                "shape": shape,
                "type_id": tensor_type,
                "type": type_name,
                "element_count": element_count,
                "relative_data_offset": offset,
            })
        raw_alignment = manifest["metadata"].get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(raw_alignment, int) or isinstance(raw_alignment, bool):
            raise SkillError("GGUF general.alignment must be an integer")
        alignment = raw_alignment
        if (
            alignment <= 0
            or alignment > MAX_GGUF_ALIGNMENT
            or alignment & (alignment - 1)
        ):
            raise SkillError(
                f"GGUF alignment must be a power of two no larger than {MAX_GGUF_ALIGNMENT}"
            )
        data_start = align_offset(pos, alignment)
        manifest["alignment"] = alignment
        manifest["header"]["alignment"] = alignment
        manifest["tensor_data_start_offset"] = data_start
        if data_start > len(data):
            manifest.setdefault("errors", []).append("GGUF tensor data start is outside the file")
        for tensor in manifest["tensors"]:
            if int(tensor["relative_data_offset"]) % alignment:
                manifest.setdefault("errors", []).append(
                    f"tensor {tensor['name']} relative data offset is not aligned to {alignment} bytes"
                )
            tensor["absolute_data_offset"] = data_start + int(tensor["relative_data_offset"])
            if tensor["absolute_data_offset"] > len(data):
                manifest.setdefault("errors", []).append(f"tensor {tensor['name']} data offset is outside the file")
            byte_width = DTYPE_BYTES.get(str(tensor["type"]))
            if byte_width:
                tensor_end = tensor["absolute_data_offset"] + tensor["element_count"] * byte_width
                if tensor_end > len(data):
                    manifest.setdefault("errors", []).append(
                        f"tensor {tensor['name']} data extent is outside the file"
                    )
    except SkillError as exc:
        manifest.setdefault("errors", []).append(str(exc))
    metadata = manifest.get("metadata", {})
    manifest["architecture"] = metadata.get("general.architecture")
    manifest["name"] = metadata.get("general.name")
    manifest["tokenizer_model"] = metadata.get("tokenizer.ggml.model")
    manifest["file_type"] = metadata.get("general.file_type")
    manifest["quantization_version"] = metadata.get("general.quantization_version")
    source_keys = sorted(
        key for key in metadata
        if key.startswith("general.source")
        or key.startswith("general.base_model")
        or key in {"general.url", "general.repo_url", "general.license", "general.quantized_by"}
    )
    manifest["source_provenance_keys"] = source_keys
    tensor_type_counts = dict(sorted(_counts(tensor["type"] for tensor in manifest.get("tensors", [])).items()))
    manifest["quantization_summary"] = {
        "file_type": manifest.get("file_type"),
        "quantization_version": manifest.get("quantization_version"),
        "tensor_type_counts": tensor_type_counts,
        "quantized_tensor_count": sum(count for dtype, count in tensor_type_counts.items() if quantized_tensor_type(dtype)),
    }
    manifest["tokenizer_summary"] = {
        "model": manifest.get("tokenizer_model"),
        "has_chat_template": "tokenizer.chat_template" in metadata,
    }
    manifest["hold_conditions"] = [
        "Verify source/base-model provenance, tokenizer metadata, and quantization deltas before mapping weights.",
        "Build a source oracle from the pre-GGUF source when source-preserving conversion is required.",
    ]
    if not manifest.get("architecture"):
        manifest["hold_conditions"].append("Missing general.architecture metadata; architecture routing must remain manual.")
    if not source_keys:
        manifest["hold_conditions"].append("Missing source/base-model provenance metadata.")
    if any(quantized_tensor_type(tensor.get("type", "")) for tensor in manifest.get("tensors", [])) and manifest.get("quantization_version") is None:
        manifest["hold_conditions"].append("Quantized tensors are present but general.quantization_version is missing.")
    return manifest


def rel_display(path: Path, root: Path) -> str:
    return str(path.relative_to(root) if root.is_dir() else path.name)


def portable_diagnostic(value: str, root: Path) -> str:
    """Remove tool-owned absolute root prefixes from portable report diagnostics."""
    root_text = str(root)
    if not root_text:
        return value
    return value.replace(root_text + os.sep, "").replace(root_text, root.name or ".")


def read_text_prefix(path: Path, limit: int = STATIC_TEXT_LIMIT) -> str:
    data = read_regular_prefix(path, limit, label=path.name)
    return data.decode("utf-8", errors="replace")


def safe_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = load_json_unique(
        read_regular_bytes(path, STATIC_TEXT_LIMIT, label=path.name),
        path.name,
    )
    return value if isinstance(value, dict) else None


def preflight_zip_archive(path: Path, data: bytes | None = None) -> bytes:
    """Bound and validate a ZIP central directory before ZipFile allocates ZipInfo objects."""
    payload = data if data is not None else read_regular_bytes(
        path,
        MAX_FORMAT_FILE_BYTES,
        label=path.name,
    )
    archive_size = len(payload)
    if archive_size < ZIP_EOCD.size:
        raise SkillError("archive is missing an end-of-central-directory record")
    tail_size = min(archive_size, ZIP_EOCD.size + ZIP_MAX_COMMENT_BYTES)
    tail = payload[archive_size - tail_size:]
    relative_offset = tail.rfind(b"PK\x05\x06")
    if relative_offset < 0 or relative_offset + ZIP_EOCD.size > len(tail):
        raise SkillError("archive is missing an end-of-central-directory record")
    (
        signature,
        disk_number,
        central_directory_disk,
        entries_on_disk,
        entry_count,
        central_directory_size,
        central_directory_offset,
        comment_size,
    ) = ZIP_EOCD.unpack_from(tail, relative_offset)
    if signature != b"PK\x05\x06":
        raise SkillError("archive end-of-central-directory signature is invalid")
    eocd_offset = archive_size - tail_size + relative_offset
    if eocd_offset + ZIP_EOCD.size + comment_size != archive_size:
        raise SkillError("archive end-of-central-directory comment length is inconsistent")
    if disk_number != 0 or central_directory_disk != 0 or entries_on_disk != entry_count:
        raise SkillError("multi-disk ZIP archives are not supported by static intake")
    if entry_count > MAX_ARCHIVE_MEMBERS:
        raise SkillError(
            f"archive member limit exceeded: {entry_count} > {MAX_ARCHIVE_MEMBERS}"
        )
    if central_directory_size > MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES:
        raise SkillError(
            "archive central directory size limit exceeded: "
            f"{central_directory_size} > {MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES}"
        )
    if central_directory_offset + central_directory_size > eocd_offset:
        raise SkillError("archive central directory extends beyond its declared bounds")

    observed_entries = 0
    consumed = 0
    cursor = central_directory_offset
    while consumed < central_directory_size:
        end = cursor + ZIP_CENTRAL_FILE_HEADER.size
        header = payload[cursor:end]
        if len(header) != ZIP_CENTRAL_FILE_HEADER.size:
            raise SkillError("archive central directory header is truncated")
        fields = ZIP_CENTRAL_FILE_HEADER.unpack(header)
        if fields[0] != b"PK\x01\x02":
            raise SkillError("archive central directory contains an invalid member header")
        filename_size, extra_size, member_comment_size = fields[10:13]
        variable_size = filename_size + extra_size + member_comment_size
        record_size = ZIP_CENTRAL_FILE_HEADER.size + variable_size
        if consumed + record_size > central_directory_size:
            raise SkillError("archive central directory member exceeds declared bounds")
        cursor += record_size
        consumed += record_size
        observed_entries += 1
        if observed_entries > MAX_ARCHIVE_MEMBERS:
            raise SkillError(
                f"archive member limit exceeded: more than {MAX_ARCHIVE_MEMBERS} central records"
            )
    if observed_entries != entry_count:
        raise SkillError(
            "archive central directory member count does not match end-of-directory metadata"
        )
    return payload


def validated_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise SkillError(f"archive member limit exceeded: {len(members)} > {MAX_ARCHIVE_MEMBERS}")
    total_size = 0
    for member in members:
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise SkillError(f"archive member path is unsafe: {member.filename}")
        unix_mode = member.external_attr >> 16
        if unix_mode & 0o170000 == 0o120000:
            raise SkillError(f"archive member is a symlink: {member.filename}")
        if member.file_size > MAX_ARCHIVE_MEMBER_UNCOMPRESSED_BYTES:
            raise SkillError(
                f"archive member expansion limit exceeded for {member.filename}: "
                f"{member.file_size} > {MAX_ARCHIVE_MEMBER_UNCOMPRESSED_BYTES}"
            )
        total_size += member.file_size
        if total_size > MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES:
            raise SkillError(
                f"archive total expansion limit exceeded: {total_size} > {MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES}"
            )
        if member.file_size and (
            member.compress_size == 0
            or member.file_size / member.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO
        ):
            raise SkillError(f"archive compression ratio is suspicious for {member.filename}")
    return members


def inspect_flax_orbax_dir(path: Path, root: Path) -> dict[str, Any]:
    files, truncated = bounded_files(path, MAX_TRAVERSAL_FILES)
    if truncated:
        raise SkillError(f"Flax/Orbax traversal exceeds {MAX_TRAVERSAL_FILES} files")
    msgpack_files = [rel_display(p, path) for p in files if p.suffix == ".msgpack"]
    metadata_files = [p for p in files if p.name in {"_CHECKPOINT_METADATA", "tree_metadata.json", "checkpoint"}]
    metadata: dict[str, Any] = {}
    for file in metadata_files:
        parsed = safe_json_file(file)
        metadata[rel_display(file, path)] = parsed if parsed is not None else {"parse_error": "not JSON metadata"}
    manifest = {
        "format": "flax-orbax",
        "path": rel_display(path, root),
        "file_count": len(files),
        "msgpack_files": msgpack_files,
        "metadata_files": [rel_display(p, path) for p in metadata_files],
        "metadata": metadata,
        "tree_paths": sorted(
            key
            for item in metadata.values()
            if isinstance(item, dict)
            for key in item.get("tree_paths", [])
            if isinstance(key, str)
        ),
        "limitations": [
            "Flax/Orbax metadata supports intake triage only; restoring checkpoints requires source runtime review."
        ],
        "hold_conditions": [
            "Record the source apply function, params tree, preprocessing, and RNG/dropout mode before mapping weights.",
            "Do not restore Orbax or Flax checkpoints during intake; build a source oracle and deterministic tree manifest first.",
        ],
    }
    if not msgpack_files and not metadata_files:
        manifest["hold_conditions"].append("No Flax/Orbax checkpoint metadata files were recognized.")
    return manifest


def inspect_saved_model_dir(path: Path, root: Path) -> dict[str, Any]:
    pbtxt = path / "saved_model.pbtxt"
    pb = path / "saved_model.pb"
    text = read_text_prefix(pbtxt) if pbtxt.exists() else ""
    signature_keys = sorted(set(re.findall(r'(?m)^\s*key:\s*"([^"]+)"', text)))
    method_names = sorted(set(re.findall(r'(?m)^\s*method_name:\s*"([^"]+)"', text)))
    tensor_names = sorted(set(re.findall(r'(?m)^\s*name:\s*"([^"]+)"', text)))
    op_counts = dict(sorted(_counts(re.findall(r'\bop:\s*"([^"]+)"', text)).items()))
    variable_dir = path / "variables"
    variable_paths, variables_truncated = bounded_files(variable_dir, 1_000) if variable_dir.exists() else ([], False)
    if variables_truncated:
        raise SkillError("SavedModel variables traversal exceeds 1000 files")
    variable_files = sorted(rel_display(p, path) for p in variable_paths)
    asset_paths, assets_truncated = bounded_files(path / "assets", 1_000) if (path / "assets").exists() else ([], False)
    if assets_truncated:
        raise SkillError("SavedModel assets traversal exceeds 1000 files")
    coverage = static_coverage_report(op_counts, TENSORFLOW_STATIC_COVERED_OPS, subject="tensorflow-operators")
    manifest = {
        "format": "tensorflow-savedmodel",
        "path": rel_display(path, root),
        "saved_model_files": [name for name, exists in {"saved_model.pb": pb.exists(), "saved_model.pbtxt": pbtxt.exists()}.items() if exists],
        "signature_keys": signature_keys,
        "method_names": method_names,
        "tensor_names": tensor_names[:50],
        "operator_counts": op_counts,
        "operator_coverage": coverage,
        "variables": {
            "present": bool(variable_files),
            "files": variable_files,
        },
        "assets_count": len(asset_paths),
        "limitations": [
            "SavedModel metadata supports intake triage only; protobuf graph and TensorFlow op support are not fully decoded."
        ],
        "hold_conditions": [
            "Record concrete signatures, TensorFlow ops, preprocessing, and source-framework outputs before conversion.",
            "Do not load SavedModel objects during intake; parse signatures statically or in a sandboxed source-oracle step.",
        ],
    }
    if coverage["unsupported_or_unclassified_count"]:
        names = ", ".join(coverage["unsupported_or_unclassified"])
        manifest["hold_conditions"].append(f"Unsupported or unclassified TensorFlow operators require review: {names}.")
    return manifest


def inspect_keras_archive(path: Path, root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": "keras-archive",
        "path": rel_display(path, root),
        "entries": [],
        "metadata": {},
        "config": {},
        "limitations": [
            "Keras archive metadata supports intake triage only; custom objects and TensorFlow ops require separate review."
        ],
        "hold_conditions": [
            "Review config, custom objects, preprocessing, and TensorFlow/Keras version before mapping weights.",
            "Do not load the Keras archive during intake; inspect archive members statically.",
        ],
    }
    try:
        data = read_regular_bytes(path, MAX_FORMAT_FILE_BYTES, label=rel_display(path, root))
        manifest["size_bytes"] = len(data)
        preflight_zip_archive(path, data)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = validated_zip_members(archive)
            names = sorted(member.filename for member in members)
            member_by_name = {member.filename: member for member in members}
            manifest["entries"] = names
            for name, key in (("metadata.json", "metadata"), ("config.json", "config")):
                if name not in names:
                    continue
                if member_by_name[name].file_size > STATIC_TEXT_LIMIT:
                    manifest.setdefault("errors", []).append(f"{name} exceeds static JSON read limit")
                    continue
                raw = archive.read(name)
                parsed = load_json_unique(raw, f"{path.name}:{name}")
                if isinstance(parsed, dict):
                    manifest[key] = parsed
    except (SkillError, zipfile.BadZipFile, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        manifest.setdefault("errors", []).append(str(exc))
    config = manifest.get("config", {})
    model_config = config.get("config", {}) if isinstance(config, dict) else {}
    layers = model_config.get("layers", []) if isinstance(model_config, dict) else []
    manifest["class_name"] = config.get("class_name") if isinstance(config, dict) else None
    manifest["layer_count"] = len(layers) if isinstance(layers, list) else 0
    manifest["layer_class_names"] = [
        str(layer.get("class_name"))
        for layer in layers
        if isinstance(layer, dict) and layer.get("class_name")
    ][:50]
    layer_counts = dict(sorted(_counts(manifest["layer_class_names"]).items()))
    coverage = static_coverage_report(layer_counts, KERAS_STATIC_COVERED_LAYERS, subject="keras-layers")
    manifest["layer_counts"] = layer_counts
    manifest["layer_coverage"] = coverage
    if coverage["unsupported_or_unclassified_count"]:
        names = ", ".join(coverage["unsupported_or_unclassified"])
        manifest["hold_conditions"].append(f"Unsupported or unclassified Keras layers require review: {names}.")
    manifest["weight_files"] = [
        name for name in manifest.get("entries", [])
        if name.endswith((".weights.h5", ".h5"))
    ]
    return manifest


def inspect_coreml_artifact(path: Path, root: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "format": "coreml-model",
            "path": rel_display(path, root),
            "size_bytes": regular_file_size(path, label=rel_display(path, root)),
            "operator_coverage": static_coverage_report(
                {},
                set(),
                subject="coreml-operators",
                unavailable_reason="Core ML protobuf/spec decoding is not implemented in static intake.",
            ),
            "limitations": [
                "Core ML protobuf metadata is not decoded by static intake yet."
            ],
            "hold_conditions": [
                "Require source model provenance, operator coverage, preprocessing, and source-oracle parity before porting from Core ML.",
            ],
        }
    manifest_json = safe_json_file(path / "Manifest.json") or {}
    files, truncated = bounded_files(path, MAX_TRAVERSAL_FILES)
    if truncated:
        raise SkillError(f"Core ML package traversal exceeds {MAX_TRAVERSAL_FILES} files")
    model_files = [rel_display(p, path) for p in files if p.suffix in {".mlmodel", ".mlmodelc"} or "model.mlmodel" in p.name]
    weight_files = [rel_display(p, path) for p in files if p.suffix in {".bin", ".weights"}]
    item_info = manifest_json.get("itemInfoEntries", {})
    return {
        "format": "coreml-package",
        "path": rel_display(path, root),
        "file_count": len(files),
        "manifest": {
            "fileFormatVersion": manifest_json.get("fileFormatVersion"),
            "rootModelIdentifier": manifest_json.get("rootModelIdentifier"),
            "item_count": len(item_info) if isinstance(item_info, dict) else 0,
        },
        "model_files": model_files,
        "weight_files": weight_files,
        "operator_coverage": static_coverage_report(
            {},
            set(),
            subject="coreml-operators",
            unavailable_reason="Core ML protobuf/spec decoding is not implemented in static intake.",
        ),
        "limitations": [
            "Core ML package metadata supports intake triage only; compiled/runtime semantics are not source-preserving proof."
        ],
        "hold_conditions": [
            "Require source-model provenance, Core ML op coverage, preprocessing metadata, and source-oracle parity before conversion.",
            "Do not treat Core ML packages as authoritative source checkpoints unless the pre-CoreML source is unavailable and quality gates are explicit.",
        ],
    }


def discover_source_format_artifacts(root: Path) -> list[tuple[str, Path]]:
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix == ".onnx":
            return [("onnx", root)]
        if suffix == ".gguf":
            return [("gguf", root)]
        if suffix == ".keras":
            return [("keras", root)]
        if suffix == ".mlmodel":
            return [("coreml", root)]
        if suffix == ".msgpack":
            return [("flax-orbax", root.parent)]
        return []

    files, truncated = bounded_files(root, MAX_TRAVERSAL_FILES)
    if truncated:
        raise SkillError(f"source-format traversal exceeds {MAX_TRAVERSAL_FILES} files")

    artifacts: list[tuple[str, Path]] = []
    mlpackages: set[Path] = set()
    saved_model_dirs: set[Path] = set()
    for file in files:
        package_parents = [
            parent
            for parent in file.parents
            if parent.suffix == ".mlpackage" and (parent == root or root in parent.parents)
        ]
        mlpackages.update(package_parents)
        inside_mlpackage = bool(package_parents)
        suffix = file.suffix.lower()
        if suffix == ".onnx":
            artifacts.append(("onnx", file))
        elif suffix == ".gguf":
            artifacts.append(("gguf", file))
        elif suffix == ".keras":
            artifacts.append(("keras", file))
        elif suffix == ".mlmodel" and not inside_mlpackage:
            artifacts.append(("coreml", file))
        if file.name in {"saved_model.pb", "saved_model.pbtxt"}:
            saved_model_dirs.add(file.parent)

    dir_artifacts: list[tuple[str, Path]] = [
        *(('coreml', directory) for directory in sorted(mlpackages)),
        *(('tensorflow-savedmodel', directory) for directory in sorted(saved_model_dirs)),
    ]

    flax_dirs: list[Path] = []
    for file in files:
        if file.name in {"flax_model.msgpack", "model.msgpack", "_CHECKPOINT_METADATA", "tree_metadata.json"}:
            flax_dirs.append(file.parent)
    selected_flax_dirs: list[Path] = []
    for directory in sorted(set(flax_dirs), key=lambda p: (len(p.parts), str(p))):
        if not any(parent in directory.parents for parent in selected_flax_dirs):
            selected_flax_dirs.append(directory)
    dir_artifacts.extend(("flax-orbax", directory) for directory in selected_flax_dirs)

    all_artifacts = artifacts + dir_artifacts
    unique: dict[tuple[str, Path], tuple[str, Path]] = {}
    for kind, path in all_artifacts:
        unique[(kind, path)] = (kind, path)
    return [unique[key] for key in sorted(unique, key=lambda item: (str(item[1]), item[0]))]


def source_format_budget_files(kind: str, artifact: Path) -> list[Path]:
    """Return only files that static intake opens or sizes for one artifact."""
    node = os.lstat(artifact)
    if stat.S_ISREG(node.st_mode) or stat.S_ISLNK(node.st_mode):
        return [artifact]
    if not stat.S_ISDIR(node.st_mode):
        raise SkillError(f"{artifact.name} must be a regular file or directory")
    files, truncated = bounded_files(artifact, MAX_TRAVERSAL_FILES)
    if truncated:
        raise SkillError(
            f"source-format artifact traversal exceeds {MAX_TRAVERSAL_FILES} files"
        )
    if kind == "flax-orbax":
        return [
            path for path in files
            if path.name in {"_CHECKPOINT_METADATA", "tree_metadata.json", "checkpoint"}
        ]
    if kind == "tensorflow-savedmodel":
        return [path for path in files if path.parent == artifact and path.name == "saved_model.pbtxt"]
    if kind == "coreml":
        return [path for path in files if path.parent == artifact and path.name == "Manifest.json"]
    return []


def inspect_source_formats(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifests: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        artifacts = discover_source_format_artifacts(root)
    except (OSError, SkillError) as exc:
        return [], [str(exc)]
    if len(artifacts) > MAX_SOURCE_FORMAT_ARTIFACTS:
        return [], [
            "source-format artifact-count limit exceeded: "
            f"{len(artifacts)} > {MAX_SOURCE_FORMAT_ARTIFACTS}"
        ]
    aggregate_bytes = 0
    budget_files: dict[Path, str] = {}
    for kind, artifact in artifacts:
        try:
            for budget_file in source_format_budget_files(kind, artifact):
                budget_files.setdefault(budget_file, rel_display(budget_file, root))
        except (OSError, SkillError) as exc:
            errors.append(f"{rel_display(artifact, root)}: {exc}")
    for budget_file, display in sorted(budget_files.items(), key=lambda item: str(item[0])):
        try:
            aggregate_bytes += regular_file_size(budget_file, label=display)
        except (OSError, SkillError) as exc:
            errors.append(f"{display}: {exc}")
            continue
        if aggregate_bytes > MAX_SOURCE_FORMAT_TOTAL_BYTES:
            return [], [
                "source-format aggregate byte budget exceeded: "
                f"{aggregate_bytes} > {MAX_SOURCE_FORMAT_TOTAL_BYTES}"
            ]
    inspected_bytes = 0
    for kind, artifact in artifacts:
        try:
            manifest: dict[str, Any] | None = None
            if kind == "onnx":
                manifest = inspect_onnx_file(artifact, root)
            elif kind == "gguf":
                manifest = inspect_gguf_file(artifact, root)
            elif kind == "flax-orbax":
                manifest = inspect_flax_orbax_dir(artifact, root)
            elif kind == "tensorflow-savedmodel":
                manifest = inspect_saved_model_dir(artifact, root)
            elif kind == "keras":
                manifest = inspect_keras_archive(artifact, root)
            elif kind == "coreml":
                manifest = inspect_coreml_artifact(artifact, root)
            if manifest is not None:
                manifests.append(manifest)
                inspected_bytes += int(manifest.get("size_bytes", 0) or 0)
                if inspected_bytes > MAX_SOURCE_FORMAT_TOTAL_BYTES:
                    errors.append(
                        "source-format aggregate byte budget exceeded while reading stable snapshots: "
                        f"{inspected_bytes} > {MAX_SOURCE_FORMAT_TOTAL_BYTES}"
                    )
                    break
        except (OSError, SkillError) as exc:
            errors.append(f"{rel_display(artifact, root)}: {exc}")
    return manifests, errors


def inspect_safetensors_checkpoint(
    root: Path,
    configs: dict[str, dict[str, Any]],
    tensors: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    safetensors_files = sorted({str(tensor.get("file")) for tensor in tensors if tensor.get("file")})
    if not safetensors_files or "config.json" in configs:
        return None
    paths = {str(rec["path"]) for rec in file_records}
    config_files = sorted(configs)
    sidecar_files = sorted(
        path for path in paths
        if Path(path).name.lower().startswith(("readme", "license", "copying", "notice"))
        or path.endswith((".json", ".txt", ".md", ".model", ".tiktoken"))
    )
    processor_configs = [
        name for name in config_files
        if name in {"tokenizer_config.json", "preprocessor_config.json", "processor_config.json", "feature_extractor_config.json"}
    ]
    manifest: dict[str, Any] = {
        "format": "safetensors-checkpoint",
        "path": "." if root.is_dir() else root.name,
        "safetensors_files": safetensors_files,
        "tensor_count": len(tensors),
        "tensor_key_samples": [str(tensor.get("key")) for tensor in tensors[:20]],
        "config_files": config_files,
        "processor_config_files": processor_configs,
        "sidecar_files": sidecar_files,
        "limitations": [
            "Safetensors headers prove tensor names and shapes only; they do not prove architecture, preprocessing, tokenizer, or source provenance."
        ],
        "hold_conditions": [
            "Missing config.json; architecture routing is weight-key-only and must remain blocked.",
            "Record source/base-model provenance, license, tokenizer or processor metadata, and source-framework oracle outputs before mapping weights.",
        ],
    }
    if not processor_configs:
        manifest["hold_conditions"].append("Missing tokenizer/processor metadata; preprocessing and postprocessing cannot be inferred.")
    if not sidecar_files:
        manifest["hold_conditions"].append("Missing model card, license, or provenance sidecar files.")
    return manifest


def acceptable_license_declaration(value: str) -> bool:
    normalized = " ".join(value.strip().lower().split())
    if (
        normalized in UNACCEPTABLE_LICENSE_VALUES
        or normalized.startswith(("http://", "https://", "licenseref-"))
        or len(normalized) > 256
    ):
        return False
    return re.fullmatch(r"[a-z0-9][a-z0-9 .+_()/,:-]*", normalized) is not None


def extract_license(
    root: Path,
    configs: dict[str, dict[str, Any]],
    file_records: list[dict[str, Any]],
    artifact_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    declarations: list[tuple[str, str, str]] = []
    for name, cfg in configs.items():
        for key in ("license", "license_name", "license_link"):
            value = cfg.get(key)
            if isinstance(value, str):
                declarations.append((name, key, value.strip()))
    base = root if root.is_dir() else root.parent
    inventoried_paths = {
        str(record.get("path"))
        for record in file_records
        if isinstance(record.get("path"), str)
    }
    readme = base / "README.md"
    if "README.md" in inventoried_paths:
        try:
            head = read_text_prefix(readme, 10_000)
            match = re.search(r"(?im)^license:\s*['\"]?([^\n'\"]+)", head)
            if match:
                declarations.append(("README.md", "license", match.group(1).strip()))
        except (OSError, SkillError):
            pass
    license_paths = sorted(
        path
        for path in inventoried_paths
        if len(Path(path).parts) == 1
        and Path(path).name.lower().startswith(("license", "copying", "notice"))
    )
    license_files = [Path(path).name for path in license_paths]
    identity_paths = {
        str(record.get("path"))
        for record in (artifact_identity or {}).get("manifest", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    require_identity_binding = artifact_identity is not None

    accepted: list[dict[str, str]] = []
    declared = sorted(
        {
            (source, value)
            for source, _, value in declarations
            if value
        }
    )
    for source, field, value in declarations:
        if (
            field != "license_link"
            and acceptable_license_declaration(value)
            and (not require_identity_binding or source in identity_paths)
        ):
            accepted.append({
                "kind": "declaration",
                "source": source,
                "value": value,
            })

    file_evidence: list[dict[str, Any]] = []
    for path_text in license_paths:
        evidence: dict[str, Any] = {"path": path_text, "acceptable": False}
        try:
            raw = read_regular_bytes(
                base / path_text,
                STATIC_TEXT_LIMIT,
                label=path_text,
            )
            text = raw.decode("utf-8")
            normalized_text = " ".join(text.lower().split())
            evidence.update({
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
            is_notice = Path(path_text).name.lower().startswith("notice")
            evidence["acceptable"] = (
                not is_notice
                and len(normalized_text) >= 64
                and any(marker in normalized_text for marker in LICENSE_TEXT_MARKERS)
                and (not require_identity_binding or path_text in identity_paths)
            )
            if evidence["acceptable"]:
                accepted.append({
                    "kind": "license-file",
                    "source": path_text,
                    "value": evidence["sha256"],
                })
        except (OSError, SkillError, UnicodeDecodeError) as exc:
            evidence["error"] = str(exc)
        file_evidence.append(evidence)

    reasons: list[str] = []
    if not declarations and not license_files:
        reasons.append("no license declaration or top-level license text was found")
    elif not accepted:
        reasons.append(
            "license evidence is placeholder, restricted, unreadable, unbound to the artifact, "
            "or not recognizable as license terms"
        )
    requires_review = not bool(accepted)
    return {
        "declared": [
            {"source": source, "value": value}
            for source, value in declared
        ],
        "license_files": license_files,
        "file_evidence": file_evidence,
        "accepted_evidence": accepted,
        "status": "review-required" if requires_review else "acceptable-evidence",
        "requires_review": requires_review,
        "compatibility_assessed": False,
        "reasons": reasons,
    }


def detect_risks(root: Path, configs: dict[str, dict[str, Any]], file_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    config = configs.get("config.json", {})
    if config.get("auto_map"):
        risks.append({"severity": "high", "type": "remote-code", "detail": "config.json contains auto_map; do not enable trust_remote_code without review."})
    suffixes = {rec["suffix"] for rec in file_records}
    paths = {rec["path"] for rec in file_records}
    if suffixes.intersection({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}):
        risks.append({"severity": "high", "type": "unsafe-serialization", "detail": "Pickle-capable weight artifacts are present; inspect/convert in isolation."})
    py_files = sorted(p for p in paths if p.endswith(".py"))
    if py_files:
        risks.append({"severity": "medium", "type": "custom-code", "detail": f"Custom Python files present ({len(py_files)}); review statically before import."})
    if any(Path(p).name in {"setup.py", "pyproject.toml", "requirements.txt"} for p in paths):
        risks.append({"severity": "medium", "type": "dependency-install", "detail": "Package/install metadata is present; do not install blindly."})
    if not any(p.endswith(".safetensors") for p in paths) and suffixes.intersection({".bin", ".pt", ".pth"}):
        risks.append({"severity": "high", "type": "no-safe-weights", "detail": "No safetensors file was found alongside executable-capable weight formats."})
    return risks


def match_architecture_profile(
    registry: dict[str, Any],
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a pinned hybrid profile from strong model identity signals.

    Profiles describe composable architecture capabilities. They never match on
    generic config keys or weight substrings: an exact model type or a specific
    architecture class is required before the profile can influence routing.
    """
    config = configs.get("config.json", {})
    model_type = str(config.get("model_type", "")).lower().replace("-", "_")
    arch_values = config.get("architectures", [])
    if isinstance(arch_values, str):
        arch_values = [arch_values]
    arch_text = " ".join(str(value).lower() for value in arch_values if isinstance(value, str))
    config_keys = set(config)
    matches: list[dict[str, Any]] = []

    for raw_profile in registry.get("hybrid_profiles", []):
        profile_config_keys = {
            str(value) for value in raw_profile.get("when_any_config", [])
        }
        if profile_config_keys and not profile_config_keys.intersection(config_keys):
            continue
        aliases = {
            str(value).lower().replace("-", "_")
            for value in raw_profile.get("model_type_aliases", [])
        }
        evidence: list[str] = []
        match_score = 0.0
        if model_type and model_type in aliases:
            match_score += 20.0
            evidence.append(f"exact hybrid model_type={model_type}")
        matched_patterns = [
            str(pattern)
            for pattern in raw_profile.get("class_patterns", [])
            if str(pattern).lower() in arch_text
        ]
        if matched_patterns:
            match_score += 12.0
            evidence.append("hybrid architecture class: " + ", ".join(matched_patterns))
        if not match_score:
            continue

        components: list[dict[str, Any]] = []
        for raw_component in raw_profile.get("components", []):
            required_keys = {str(value) for value in raw_component.get("when_any_config", [])}
            if required_keys and not required_keys.intersection(config_keys):
                continue
            components.append(dict(raw_component))

        primary_family = str(raw_profile.get("primary_family", ""))
        ordered_families = [str(component.get("family")) for component in components]
        if not primary_family or primary_family not in ordered_families:
            continue
        secondary_families = [family for family in ordered_families if family != primary_family]
        traits = list(dict.fromkeys(
            str(trait)
            for component in components
            for trait in component.get("traits", [])
        ))
        required_runbooks = list(dict.fromkeys(
            str(component.get("runbook"))
            for component in components
            if component.get("runbook")
        ))
        matches.append({
            "id": raw_profile.get("id"),
            "match_score": match_score,
            "evidence": evidence,
            "primary_family": primary_family,
            "secondary_families": secondary_families,
            "families": [primary_family, *secondary_families],
            "traits": traits,
            "required_runbooks": required_runbooks,
            "components": components,
            "official_reference": raw_profile.get("official_reference"),
            "notes": raw_profile.get("notes", ""),
        })

    matches.sort(key=lambda item: (-float(item["match_score"]), str(item["id"])))
    if not matches:
        return None
    selected = matches[0]
    selected["conflicting_profiles"] = [
        {
            "id": other.get("id"),
            "match_score": other.get("match_score"),
            "evidence": other.get("evidence", []),
        }
        for other in matches[1:]
    ]
    return selected


def architecture_scores(
    registry: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    tensor_keys: list[str],
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = configs.get("config.json", {})
    model_type = str(config.get("model_type", "")).lower().replace("-", "_")
    arch_values = config.get("architectures", [])
    if isinstance(arch_values, str):
        arch_values = [arch_values]
    arch_text = " ".join(str(x).lower() for x in arch_values if isinstance(x, str))
    key_blob = "\n".join(tensor_keys[:20000]).lower()
    config_keys = set(config.keys())
    profile = profile or match_architecture_profile(registry, configs)
    profile_families = set(profile.get("families", [])) if profile else set()
    routing_policy = registry.get("routing_policy", {})
    primary_bonus = float(routing_policy.get("profile_primary_bonus", 20.0))
    secondary_bonus = float(routing_policy.get("profile_secondary_bonus", 2.0))
    candidates: list[dict[str, Any]] = []
    for family in registry.get("families", []):
        score = 0.0
        evidence: list[str] = []
        aliases = [str(x).lower().replace("-", "_") for x in family.get("model_type_aliases", [])]
        if model_type and model_type in aliases:
            score += 10
            evidence.append(f"exact model_type={model_type}")
        elif model_type and any(alias in model_type or model_type in alias for alias in aliases):
            score += 5
            evidence.append(f"partial model_type={model_type}")
        matched_classes = [
            str(pattern)
            for pattern in family.get("class_patterns", [])
            if str(pattern).lower() in arch_text
        ]
        if matched_classes:
            score += 4
            evidence.append("architecture class patterns: " + ", ".join(matched_classes))
        matched_config = [sig for sig in family.get("config_signals", []) if sig in config_keys]
        score += min(len(matched_config), 5) * 0.8
        if matched_config:
            evidence.append("config keys: " + ", ".join(matched_config[:5]))
        matched_weights = [sig for sig in family.get("weight_signals", []) if str(sig).lower() in key_blob]
        score += min(len(matched_weights), 5) * 1.5
        if matched_weights:
            evidence.append("weight signals: " + ", ".join(matched_weights[:5]))
        if profile and family["id"] in profile_families:
            if family["id"] == profile["primary_family"]:
                score += primary_bonus
                evidence.append(f"hybrid profile {profile['id']} primary capability")
            else:
                score += secondary_bonus
                evidence.append(f"hybrid profile {profile['id']} secondary capability")
        if score > 0:
            candidate = {
                "family": family["id"],
                "score": round(score, 2),
                "runbook": family.get("runbook"),
                "targets": family.get("targets", []),
                "state": family.get("state"),
                "evidence": evidence,
                "notes": family.get("notes", ""),
            }
            if profile and family["id"] in profile_families:
                candidate["profile_id"] = profile["id"]
                candidate["traits"] = next(
                    (
                        component.get("traits", [])
                        for component in profile.get("components", [])
                        if component.get("family") == family["id"]
                    ),
                    [],
                )
            candidates.append(candidate)
    candidates.sort(key=lambda x: (-x["score"], x["family"]))
    confidence_reference = float(routing_policy.get("confidence_reference_score", 20.0))
    for item in candidates:
        item["confidence"] = round(min(float(item["score"]) / confidence_reference, 1.0), 3)
    return candidates[:5]


def architecture_routing_decision(
    registry: dict[str, Any],
    candidates: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = registry.get("routing_policy", {})
    minimum_score = float(policy.get("minimum_score", 8.0))
    minimum_margin = float(policy.get("minimum_margin", 2.0))
    winner = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    winner_score = float(winner["score"]) if winner else 0.0
    runner_up_score = float(runner_up["score"]) if runner_up else 0.0
    margin = winner_score - runner_up_score if winner else 0.0
    reasons: list[str] = []

    if winner is None:
        reasons.append("no supported family produced architecture evidence")
    elif winner_score < minimum_score:
        reasons.append(
            f"top score {winner_score:.2f} is below the absolute minimum {minimum_score:.2f}"
        )
    if winner and runner_up and margin < minimum_margin:
        reasons.append(
            f"winner margin {margin:.2f} over {runner_up['family']} is below the minimum {minimum_margin:.2f}"
        )
    if profile and winner and winner["family"] != profile.get("primary_family"):
        reasons.append(
            f"hybrid profile {profile['id']} expects primary family {profile['primary_family']} but scoring selected {winner['family']}"
        )
    if profile and profile.get("conflicting_profiles"):
        conflicting_ids = ", ".join(
            str(item.get("id")) for item in profile["conflicting_profiles"]
        )
        reasons.append(
            f"hybrid identity signals conflict between {profile['id']} and {conflicting_ids}"
        )

    return {
        "status": "ambiguous" if reasons else "recommended",
        "minimum_score": minimum_score,
        "minimum_margin": minimum_margin,
        "winner_family": winner.get("family") if winner else None,
        "winner_score": round(winner_score, 2),
        "runner_up_family": runner_up.get("family") if runner_up else None,
        "runner_up_score": round(runner_up_score, 2) if runner_up else None,
        "winner_margin": round(margin, 2) if runner_up else None,
        "reasons": reasons,
    }


def recommendation_blockers(
    risks: list[dict[str, str]],
    tensor_count: int,
    source_format_manifests: list[dict[str, Any]] | None = None,
    routing_decision: dict[str, Any] | None = None,
    integrity_errors: list[str] | None = None,
    artifact_identity: dict[str, Any] | None = None,
    license_info: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    high_risks = {risk["type"] for risk in risks if risk.get("severity") == "high"}
    format_names = {manifest.get("format") for manifest in (source_format_manifests or [])}
    if tensor_count == 0 and not format_names:
        blockers.append("no safetensors were inspected; architecture routing is metadata-only")
    if format_names:
        blockers.append(
            "source-format static intake is triage-only; conversion support requires unsupported-op reports and MLX parity fixtures"
        )
    manifest_errors = [
        str(error)
        for manifest in (source_format_manifests or [])
        for error in manifest.get("errors", [])
    ]
    if manifest_errors or integrity_errors:
        blockers.append(
            "artifact integrity validation failed; repair every malformed, inconsistent, "
            "unreadable, or over-budget model artifact before recommendation"
        )
    if "gguf" in format_names and any(
        any("Missing source/base-model provenance metadata" in condition for condition in manifest.get("hold_conditions", []))
        for manifest in (source_format_manifests or [])
        if manifest.get("format") == "gguf"
    ):
        blockers.append("GGUF source/base-model provenance is missing or incomplete")
    if "safetensors-checkpoint" in format_names:
        blockers.append("safetensors checkpoint is missing required config or provenance metadata")
    if "inventory-truncated" in high_risks:
        blockers.append(
            "file inventory was truncated; complete static inventory is required before recommendation"
        )
    if high_risks.intersection({"remote-code", "unsafe-serialization", "no-safe-weights"}):
        blockers.append("high-risk remote-code or unsafe-serialization flags require manual review")
    identity_fingerprint = (artifact_identity or {}).get("fingerprint")
    identity_manifest = (artifact_identity or {}).get("manifest")
    if not (
        (artifact_identity or {}).get("status") == "verified"
        and (artifact_identity or {}).get("immutable") is True
        and isinstance(identity_fingerprint, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", identity_fingerprint)
        and isinstance(identity_manifest, list)
        and bool(identity_manifest)
    ):
        blockers.append(
            "immutable artifact identity is missing or incomplete; a complete stable "
            "SHA-256 artifact manifest is required before recommendation"
        )
    if not (
        (license_info or {}).get("status") == "acceptable-evidence"
        and (license_info or {}).get("requires_review") is False
        and bool((license_info or {}).get("accepted_evidence"))
    ):
        blockers.append(
            "license evidence is missing or unacceptable; provide an explicit non-placeholder "
            "license declaration or a readable artifact-bound license text"
        )
    if routing_decision and routing_decision.get("status") == "ambiguous":
        reasons = "; ".join(str(reason) for reason in routing_decision.get("reasons", []))
        blockers.append(f"architecture routing is ambiguous: {reasons or 'manual family review is required'}")
    return blockers


def make_markdown(report: dict[str, Any]) -> str:
    summary = report["tensor_summary"]
    candidates = report["architecture_candidates"]
    artifact_identity = report.get("artifact_identity", {})
    license_info = report.get("license", {})
    lines = [
        "# Static model inspection",
        "",
        f"- Source: `{report['source']['input']}`",
        f"- Local path: `{report['local_path']}`",
        f"- Artifact identity: `{artifact_identity.get('status', 'missing')}`"
        + (
            f" (`{artifact_identity['fingerprint']}`)"
            if artifact_identity.get("fingerprint")
            else ""
        ),
        f"- License evidence: `{license_info.get('status', 'review-required')}`",
        f"- Files: {report['file_summary']['count']}" + (" (inventory truncated)" if report['file_summary']['truncated'] else ""),
        f"- Tensors: {summary['count']}",
        f"- Parameters represented in inspected safetensors: {summary['parameters']:,}",
        f"- Estimated tensor bytes: {summary['estimated_bytes']:,}",
        "",
        "## Architecture candidates",
        "",
    ]
    if candidates:
        lines.extend(["| Family | Score | Confidence | Runbook |", "|---|---:|---:|---|"])
        for c in candidates:
            lines.append(f"| {c['family']} | {c['score']} | {c['confidence']:.3f} | `{c['runbook']}` |")
    else:
        lines.append("No architecture family reached a non-zero score. Manual routing is required.")
    decision = report.get("routing_decision", {})
    lines += [
        "",
        "## Architecture routing decision",
        "",
        f"- Status: `{decision.get('status', 'unknown')}`",
        f"- Absolute evidence threshold: {decision.get('minimum_score', 'unknown')}",
        f"- Required winner margin: {decision.get('minimum_margin', 'unknown')}",
    ]
    if report.get("recommended_families"):
        lines.append("- Routed families: " + ", ".join(f"`{family}`" for family in report["recommended_families"]))
        lines.append("- Required runbooks: " + ", ".join(f"`{runbook}`" for runbook in report["recommended_runbooks"]))
    profile = report.get("architecture_profile")
    if profile:
        lines += [
            f"- Hybrid profile: `{profile['id']}`",
            "- Composable traits: " + ", ".join(f"`{trait}`" for trait in profile.get("traits", [])),
            f"- Pinned MLX reference: {profile.get('official_reference')}",
        ]
    source_formats = report.get("source_format_summary", {})
    if source_formats.get("count"):
        lines += ["", "## Source format manifests", ""]
        for manifest in source_formats.get("manifests", []):
            lines.append(f"- `{manifest.get('path')}`: {manifest.get('format')} static metadata")
            for condition in manifest.get("hold_conditions", [])[:3]:
                lines.append(f"  - hold: {condition}")
    if report.get("recommendation_blockers"):
        lines += ["", "## Recommendation blockers", ""]
        for blocker in report["recommendation_blockers"]:
            lines.append(f"- {blocker}")
    lines += ["", "## Risk flags", ""]
    if report["risks"]:
        for risk in report["risks"]:
            lines.append(f"- **{risk['severity']} / {risk['type']}**: {risk['detail']}")
    else:
        lines.append("- No static high-signal risk flag detected. This is not a security guarantee.")
    lines += ["", "## Next actions", "", "1. Confirm license and pinned source revision.", "2. Review any custom Python statically; do not execute it during intake.", "3. Freeze source-oracle fixtures.", "4. Generate a port plan with `make_port_plan.py`."]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        root, source = resolve_model(
            args.model,
            args.allow_network,
            args.revision,
            args.download_weights,
            args.include_local_paths,
        )
        registry = load_structured(args.registry)
        files, truncated = inventory(root, args.max_files, args.hash_small_files)
        files = exclude_generated_reports(
            root,
            files,
            args.output,
            args.markdown,
            model_input=args.model,
        )
        configs = load_configs(root)
        config_base = root if root.is_dir() else root.parent
        artifact_identity = build_artifact_identity(
            root,
            files,
            truncated,
            extra_paths=[config_base / name for name in sorted(configs)],
        )
        license_info = extract_license(root, configs, files, artifact_identity)
        tensors, tensor_metadata, tensor_errors = inspect_tensors(root)
        source_format_manifests, source_format_errors = inspect_source_formats(root)
        if not args.include_local_paths:
            tensor_errors = [portable_diagnostic(error, root) for error in tensor_errors]
            source_format_errors = [
                portable_diagnostic(error, root) for error in source_format_errors
            ]
        checkpoint_manifest = inspect_safetensors_checkpoint(root, configs, tensors, files)
        if checkpoint_manifest:
            source_format_manifests.append(checkpoint_manifest)
        tensor_keys = [t["key"] for t in tensors]
        profile = match_architecture_profile(registry, configs)
        candidates = architecture_scores(registry, configs, tensor_keys, profile)
        routing_decision = architecture_routing_decision(registry, candidates, profile)
        risks = detect_risks(root, configs, files)
        if truncated:
            risks.append({
                "severity": "high",
                "type": "inventory-truncated",
                "detail": (
                    f"File inventory reached --max-files={args.max_files}; uninspected files may contain "
                    "architecture, provenance, or safety evidence."
                ),
            })
            routing_decision = {
                **routing_decision,
                "status": "ambiguous",
                "reasons": [
                    *routing_decision.get("reasons", []),
                    "file inventory is truncated",
                ],
            }
        blockers = recommendation_blockers(
            risks,
            len(tensors),
            source_format_manifests,
            routing_decision,
            [*tensor_errors, *source_format_errors],
            artifact_identity,
            license_info,
        )
        recommended = (
            candidates[0]
            if candidates and routing_decision["status"] == "recommended" and not blockers
            else None
        )
        if recommended and profile:
            recommended_families = list(profile["families"])
            recommended_runbooks = list(profile["required_runbooks"])
        elif recommended:
            recommended_families = [str(recommended["family"])]
            recommended_runbooks = [str(recommended["runbook"])] if recommended.get("runbook") else []
        else:
            recommended_families = []
            recommended_runbooks = []
        source_formats = sorted({manifest.get("format", "unknown") for manifest in source_format_manifests})
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inspection_mode": "static-no-model-import",
            "source": source,
            "artifact_identity": artifact_identity,
            "local_path": (
                str(root)
                if args.include_local_paths
                else (root.name or ".")
                if source.get("kind") == "local"
                else "<temporary-huggingface-snapshot>"
            ),
            "configs": configs,
            "file_summary": {
                "count": len(files),
                "truncated": truncated,
                "extensions": dict(sorted(_counts(rec["suffix"] or "<none>" for rec in files).items())),
                "files": files,
            },
            "tensor_summary": {
                "count": len(tensors),
                "parameters": sum(t["parameters"] for t in tensors),
                "estimated_bytes": sum(t["estimated_bytes"] for t in tensors),
                "dtypes": dict(sorted(_counts(str(t["dtype"]) for t in tensors).items())),
                "files": dict(sorted(_counts(t["file"] for t in tensors).items())),
                "metadata": tensor_metadata,
                "errors": tensor_errors,
                "integrity_ok": not tensor_errors,
            },
            "tensors": tensors,
            "source_format_summary": {
                "count": len(source_format_manifests),
                "formats": source_formats,
                "errors": source_format_errors,
                "integrity_ok": not source_format_errors and not any(
                    manifest.get("errors") for manifest in source_format_manifests
                ),
                "aggregate": {
                    "artifact_count": len(source_format_manifests),
                    "inspected_bytes": sum(
                        int(manifest.get("size_bytes", 0) or 0)
                        for manifest in source_format_manifests
                    ),
                    "artifact_limit": MAX_SOURCE_FORMAT_ARTIFACTS,
                    "byte_limit": MAX_SOURCE_FORMAT_TOTAL_BYTES,
                },
                "manifests": source_format_manifests,
            },
            "architecture_candidates": candidates,
            "architecture_profile": profile,
            "architecture_traits": list(profile.get("traits", [])) if profile else [],
            "routing_decision": routing_decision,
            "recommended_family": recommended["family"] if recommended else None,
            "recommended_runbook": recommended["runbook"] if recommended else None,
            "recommended_families": recommended_families,
            "recommended_runbooks": recommended_runbooks,
            "recommendation_blockers": blockers,
            "license": license_info,
            "risks": risks,
            "limitations": [
                "Static inspection does not prove model behavior or license compatibility.",
                "Parameter totals may double-count tied tensors and omit non-safetensors weights.",
                "Remote metadata-only inspection cannot see tensor keys without downloading weights.",
            ],
        }
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        if args.markdown:
            atomic_write_text(args.markdown, make_markdown(report))
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
