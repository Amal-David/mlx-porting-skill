#!/usr/bin/env python3
"""Convert safetensors checkpoints to explicit MLX parameter names."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import stat
import struct
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from _common import SkillError
from _weight_map_common import (
    POLICY_TO_DTYPE,
    SCHEMA_VERSION,
    effective_policy as _effective_policy,
    infer_reshape as _infer_reshape,
    name as _name,
    normalize_axis as _axis,
    parse_conversion_map,
    parse_strict_json,
    shape as _shape,
    validate_mapping_against_source as _validate_mapping_against_source,
)


REPORT_SCHEMA_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSORS = 1_000_000
MAX_DIRECTORY_ENTRIES = 20_000
MAX_TENSOR_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MAX_WORKING_SET_BYTES = 64 * 1024 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
OPTIONAL_LIBS_ENV = "MLX_PORTING_DISABLE_OPTIONAL_TENSOR_LIBS"

DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
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
}
NUMPY_DTYPES = {
    "BOOL": np.dtype("?"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F64": np.dtype("<f8"),
}
NUMPY_TO_SAFETENSORS = {
    np.dtype("?"): "BOOL",
    np.dtype("i1"): "I8",
    np.dtype("u1"): "U8",
    np.dtype("i2"): "I16",
    np.dtype("u2"): "U16",
    np.dtype("f2"): "F16",
    np.dtype("i4"): "I32",
    np.dtype("u4"): "U32",
    np.dtype("f4"): "F32",
    np.dtype("i8"): "I64",
    np.dtype("u8"): "U64",
    np.dtype("f8"): "F64",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a safetensors checkpoint using an explicit WEIGHT_MAP.json",
    )
    parser.add_argument("--source", required=True, help="Model directory, or source manifest in draft mode")
    parser.add_argument("--mapping", help="Resolved WEIGHT_MAP.json (schema_version 2)")
    parser.add_argument("--output", help="Conversion output directory")
    parser.add_argument(
        "--emit-draft-map",
        nargs="?",
        const="-",
        metavar="PATH",
        help="Emit a draft map to PATH (or stdout when omitted)",
    )
    parser.add_argument("--scaffold-manifest", help="Target scaffold manifest used only in draft mode")
    parser.add_argument(
        "--max-working-set-bytes",
        type=int,
        default=DEFAULT_MAX_WORKING_SET_BYTES,
        help=(
            "Fail before tensor allocation when aggregate materialization or the "
            "estimated peak working set exceeds this many bytes"
        ),
    )
    return parser.parse_args(argv)


def strict_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SkillError(f"value is not strict JSON: {exc}") from exc


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _validate_components(path: Path, *, label: str, allow_missing: bool) -> Path:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    missing_seen = False
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if not allow_missing:
                raise SkillError(f"{label} does not exist: {absolute}")
            missing_seen = True
            continue
        except OSError as exc:
            raise SkillError(f"could not inspect {label} path component {current}: {exc}") from exc
        if missing_seen:
            raise SkillError(f"{label} path changed while it was inspected: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise SkillError(f"{label} path contains symlink component: {current}")
        if current != absolute and not stat.S_ISDIR(metadata.st_mode):
            raise SkillError(f"{label} parent component is not a directory: {current}")
    return absolute


def validate_existing_path(path: Path, *, label: str, kind: str) -> Path:
    absolute = _validate_components(path, label=label, allow_missing=False)
    try:
        metadata = os.lstat(absolute)
    except OSError as exc:
        raise SkillError(f"could not inspect {label} {absolute}: {exc}") from exc
    expected = stat.S_ISDIR(metadata.st_mode) if kind == "directory" else stat.S_ISREG(metadata.st_mode)
    if not expected:
        raise SkillError(f"{label} must be a regular non-symlink {kind}: {absolute}")
    return absolute


def prepare_output_directory(path: Path) -> Path:
    absolute = _validate_components(path, label="output", allow_missing=True)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise SkillError(f"could not create output directory {current}: {exc}") from exc
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise SkillError(f"output path contains symlink component: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SkillError(f"output path component is not a directory: {current}")
    return absolute


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
        raise SkillError("this platform does not provide O_NOFOLLOW; refusing checkpoint reads")
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


def _verify_snapshot(path: Path, descriptor: int, initial: os.stat_result, label: str) -> None:
    final = os.fstat(descriptor)
    try:
        current = os.lstat(path)
    except FileNotFoundError as exc:
        raise SkillError(f"{label} changed while it was read") from exc
    if not stat.S_ISREG(current.st_mode) or not _same_snapshot(initial, final) or not _same_snapshot(initial, current):
        raise SkillError(f"{label} changed while it was read")


def _read_exact(descriptor: int, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
        if not chunk:
            raise SkillError(f"{label} ended while it was read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_bounded_json_capture(path: Path, *, label: str) -> tuple[Any, dict[str, Any]]:
    """Parse and attest one stable, bounded byte snapshot without reopening it."""
    path = validate_existing_path(path, label=label, kind="file")
    descriptor, metadata = _open_regular_nofollow(path, label)
    try:
        if metadata.st_size > MAX_JSON_BYTES:
            raise SkillError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
        raw = _read_exact(descriptor, metadata.st_size, label)
        _verify_snapshot(path, descriptor, metadata, label)
    finally:
        os.close(descriptor)
    value = parse_strict_json(raw, label=label)
    digest = {
        "name": path.name,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return value, digest


def read_bounded_json(path: Path, *, label: str) -> Any:
    return read_bounded_json_capture(path, label=label)[0]


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Validate a safetensors manifest without importing safetensors."""
    label = path.name
    descriptor, metadata = _open_regular_nofollow(path, label)
    try:
        if metadata.st_size < 8:
            raise SkillError(f"invalid safetensors header in {label}: missing length")
        raw_length = _read_exact(descriptor, 8, label)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > MAX_HEADER_BYTES:
            raise SkillError(f"suspicious safetensors header length {header_length} in {label}")
        if 8 + header_length > metadata.st_size:
            raise SkillError(f"truncated safetensors header in {label}")
        raw_header = _read_exact(descriptor, header_length, label)
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise SkillError(f"{label} ended while it was hashed")
            digest.update(chunk)
            remaining -= len(chunk)
        _verify_snapshot(path, descriptor, metadata, label)
    finally:
        os.close(descriptor)
    try:
        value = parse_strict_json(raw_header, label=f"safetensors header in {label}")
    except SkillError:
        raise
    if not isinstance(value, dict):
        raise SkillError(f"safetensors header in {label} must be an object")
    file_metadata = value.pop("__metadata__", {})
    if not isinstance(file_metadata, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in file_metadata.items()
    ):
        raise SkillError(f"safetensors metadata in {label} must map strings to strings")
    if len(value) > MAX_TENSORS:
        raise SkillError(f"safetensors tensor-count limit exceeded in {label}")
    payload_size = metadata.st_size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    tensors: dict[str, dict[str, Any]] = {}
    for key, spec in value.items():
        if not isinstance(key, str) or not key or not isinstance(spec, dict):
            raise SkillError(f"invalid safetensors tensor record in {label}")
        if set(spec) != {"dtype", "shape", "data_offsets"}:
            raise SkillError(f"safetensors tensor {key!r} in {label} has an invalid field set")
        dtype = spec.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise SkillError(f"safetensors tensor {key!r} in {label} has unknown dtype {dtype!r}")
        shape = _shape(spec.get("shape"), label=f"safetensors tensor {key!r} shape")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(offset) is int for offset in offsets)
        ):
            raise SkillError(f"safetensors tensor {key!r} in {label} has invalid data_offsets")
        start, end = offsets
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if expected > MAX_TENSOR_BYTES:
            raise SkillError(f"safetensors tensor {key!r} in {label} exceeds the tensor-byte limit")
        if start < 0 or end < start or end > payload_size or end - start != expected:
            raise SkillError(f"safetensors tensor {key!r} in {label} has out-of-range data_offsets")
        tensors[key] = {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}
        intervals.append((start, end, key))
    cursor = 0
    for start, end, key in sorted(intervals, key=lambda item: (item[0], item[1], item[2])):
        if start != cursor:
            raise SkillError(
                f"safetensors tensor {key!r} in {label} creates overlapping or non-contiguous offsets"
            )
        cursor = end
    if cursor != payload_size:
        raise SkillError(f"safetensors payload in {label} is not fully described by its header")
    return {
        "path": path,
        "header_length": header_length,
        "payload_offset": 8 + header_length,
        "tensors": tensors,
        "metadata": file_metadata,
        "snapshot": metadata,
        "digest": {
            "name": path.name,
            "size_bytes": metadata.st_size,
            "sha256": digest.hexdigest(),
        },
    }


def _bf16_to_float32(raw: bytes, shape: list[int]) -> np.ndarray:
    upper = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (upper << 16).view(np.float32).reshape(shape).copy()


def _float32_to_bf16(array: np.ndarray) -> bytes:
    bits = np.asarray(array, dtype=np.float32).reshape(-1).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2").tobytes(order="C")


def read_safetensors_tensor_pure(file_info: dict[str, Any], key: str) -> np.ndarray:
    spec = file_info["tensors"][key]
    start, end = spec["data_offsets"]
    descriptor, metadata = _open_regular_nofollow(file_info["path"], file_info["path"].name)
    try:
        if not _same_snapshot(file_info["snapshot"], metadata):
            raise SkillError(f"{file_info['path'].name} changed after manifest validation")
        os.lseek(descriptor, file_info["payload_offset"] + start, os.SEEK_SET)
        raw = _read_exact(descriptor, end - start, file_info["path"].name)
        _verify_snapshot(file_info["path"], descriptor, metadata, file_info["path"].name)
    finally:
        os.close(descriptor)
    if spec["dtype"] == "BF16":
        return _bf16_to_float32(raw, spec["shape"])
    dtype = NUMPY_DTYPES[spec["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(spec["shape"]).copy()


def load_tensor(file_info: dict[str, Any], key: str) -> np.ndarray:
    """Use safetensors lazily when available, with the validated pure reader as fallback."""
    if os.environ.get(OPTIONAL_LIBS_ENV) != "1":
        try:
            from safetensors import safe_open  # type: ignore

            with safe_open(str(file_info["path"]), framework="np", device="cpu") as handle:
                value = np.asarray(handle.get_tensor(key))
            current = os.lstat(file_info["path"])
            if not _same_snapshot(file_info["snapshot"], current):
                raise SkillError(f"{file_info['path'].name} changed after manifest validation")
            if list(value.shape) != file_info["tensors"][key]["shape"]:
                raise SkillError(f"safetensors returned an unexpected shape for {key!r}")
            return value
        except ImportError:
            pass
        except SkillError:
            raise
        except Exception:
            # NumPy has no native BF16 dtype on supported CI versions. The pure
            # reader intentionally decodes it to float32 while retaining the
            # manifest dtype for a loss-bounded BF16 write.
            pass
    return read_safetensors_tensor_pure(file_info, key)


def _encode_array(array: np.ndarray, dtype: str) -> bytes:
    contiguous = np.ascontiguousarray(array)
    if dtype == "BF16":
        return _float32_to_bf16(contiguous)
    try:
        target = NUMPY_DTYPES[dtype]
    except KeyError as exc:
        raise SkillError(f"cannot write safetensors dtype {dtype!r}") from exc
    return contiguous.astype(target, copy=False).tobytes(order="C")


def _write_new_bytes(path: Path, data: bytes) -> None:
    path = _absolute_lexical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise SkillError(f"refusing to overwrite existing output: {path.name}") from exc
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SkillError(f"could not make progress writing output: {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)


def write_safetensors_pure(
    path: Path,
    tensors: dict[str, np.ndarray],
    dtypes: dict[str, str] | None = None,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write deterministic MLX-loadable safetensors without safetensors or MLX."""
    if not tensors:
        raise SkillError("refusing to write an empty safetensors file")
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    for key in sorted(tensors):
        if not isinstance(key, str) or not key:
            raise SkillError("safetensors output keys must be non-empty strings")
        array = np.asarray(tensors[key])
        dtype = dtypes[key] if dtypes is not None else NUMPY_TO_SAFETENSORS.get(array.dtype)
        if dtype not in DTYPE_BYTES:
            raise SkillError(f"unsupported output dtype for tensor {key!r}: {dtype!r}")
        payload_size = math.prod(array.shape) * DTYPE_BYTES[str(dtype)]
        records[key] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + payload_size],
        }
        offset += payload_size
    header: dict[str, Any] = dict(records)
    header["__metadata__"] = dict(sorted((metadata or {"format": "mlx"}).items()))
    raw_header = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    if len(raw_header) > MAX_HEADER_BYTES:
        raise SkillError("generated safetensors header exceeds the header-size limit")
    path = _absolute_lexical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise SkillError(f"refusing to overwrite existing output: {path.name}") from exc
    try:
        chunks = [struct.pack("<Q", len(raw_header)), raw_header]
        for key in sorted(tensors):
            dtype = str(records[key]["dtype"])
            chunks.append(_encode_array(np.asarray(tensors[key]), dtype))
            for data in chunks:
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise SkillError(f"could not make progress writing output: {path.name}")
                    view = view[written:]
            chunks.clear()
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)


def read_safetensors_index(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value, digest = read_bounded_json_capture(path, label="safetensors shard index")
    if not isinstance(value, dict) or set(value) - {"metadata", "weight_map"}:
        raise SkillError(f"safetensors shard index {path.name} has an invalid top-level schema")
    metadata = value.get("metadata", {})
    weight_map = value.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict) or not weight_map:
        raise SkillError(f"safetensors shard index {path.name} has invalid metadata or weight_map")
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(shard, str) or not shard:
            raise SkillError(f"safetensors shard index {path.name} has an invalid weight_map entry")
        portable = PurePosixPath(shard)
        if portable.is_absolute() or len(portable.parts) != 1 or ".." in portable.parts or "\\" in shard:
            raise SkillError(f"safetensors shard index {path.name} has unsafe shard path {shard!r}")
        if not shard.endswith(".safetensors"):
            raise SkillError(f"safetensors shard index {path.name} references a non-safetensors shard")
    total_size = metadata.get("total_size")
    if total_size is not None and (type(total_size) is not int or total_size < 0):
        raise SkillError(f"safetensors shard index {path.name} has invalid metadata.total_size")
    return value, digest


def discover_checkpoint(source: Path) -> dict[str, Any]:
    source = validate_existing_path(source, label="source", kind="directory")
    try:
        with os.scandir(source) as iterator:
            entries = []
            for index, entry in enumerate(iterator, start=1):
                if index > MAX_DIRECTORY_ENTRIES:
                    raise SkillError(f"source directory exceeds {MAX_DIRECTORY_ENTRIES} entries")
                entries.append(entry)
    except OSError as exc:
        raise SkillError(f"could not enumerate source directory: {exc}") from exc
    safetensor_paths: list[Path] = []
    index_paths: list[Path] = []
    for entry in sorted(entries, key=lambda item: item.name):
        if not (entry.name.endswith(".safetensors") or entry.name.endswith(".safetensors.index.json")):
            continue
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise SkillError(f"checkpoint input must be a regular non-symlink file: {entry.name}")
        path = source / entry.name
        if entry.name.endswith(".safetensors.index.json"):
            index_paths.append(path)
        else:
            safetensor_paths.append(path)
    if not safetensor_paths:
        raise SkillError("source directory contains no top-level *.safetensors files")
    if len(index_paths) > 1:
        raise SkillError("multiple safetensors shard indexes were found")
    if len(safetensor_paths) > 1 and not index_paths:
        raise SkillError("multiple safetensors shards were found without a shard index")
    files: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    tensors: dict[str, dict[str, Any]] = {}
    for path in safetensor_paths:
        info = read_safetensors_header(path)
        files[path.name] = info
        for key, spec in info["tensors"].items():
            if key in owners:
                raise SkillError(f"duplicate safetensors tensor key {key!r} in {owners[key]} and {path.name}")
            owners[key] = path.name
            tensors[key] = {**spec, "file": path.name}
    index_value = None
    index_digest = None
    if index_paths:
        index_path = index_paths[0]
        index_value, index_digest = read_safetensors_index(index_path)
        weight_map = index_value["weight_map"]
        discovered = set(files)
        listed = {str(name) for name in weight_map.values()}
        missing = sorted(listed - discovered)
        unexpected = sorted(discovered - listed)
        if missing:
            raise SkillError(
                f"safetensors shard index {index_path.name} references missing shards: {', '.join(missing)}"
            )
        if unexpected:
            raise SkillError(
                f"safetensors shard index {index_path.name} does not account for shards: {', '.join(unexpected)}"
            )
        for key, shard in weight_map.items():
            if shard not in files or key not in files[shard]["tensors"] or owners.get(key) != shard:
                raise SkillError(f"safetensors shard index {index_path.name} maps {key!r} to the wrong shard")
        omitted = sorted(set(tensors) - set(weight_map))
        if omitted:
            raise SkillError(f"safetensors shard index {index_path.name} omits tensors: {', '.join(omitted[:20])}")
        total_size = index_value.get("metadata", {}).get("total_size")
        indexed_bytes = sum(
            math.prod(spec["shape"]) * DTYPE_BYTES[spec["dtype"]]
            for key, spec in tensors.items()
            if key in weight_map
        )
        if total_size is not None and total_size != indexed_bytes:
            raise SkillError(
                f"safetensors shard index {index_path.name} metadata.total_size {total_size} "
                f"does not match indexed tensor bytes {indexed_bytes}"
            )
    input_digests = [files[path.name]["digest"] for path in safetensor_paths]
    if index_digest is not None:
        input_digests.append(index_digest)
    return {
        "root": source,
        "files": files,
        "tensors": tensors,
        "input_digests": sorted(input_digests, key=lambda item: item["name"]),
    }


def apply_unary_array(array: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    op = transform["op"]
    if op in {"identity", "rename", "cast"}:
        return array
    if op in {"transpose", "permute"}:
        return np.transpose(array, transform["axes"])
    if op == "reshape":
        return np.reshape(array, _infer_reshape(list(array.shape), transform["shape"]))
    if op == "squeeze":
        axis = _axis(transform["axis"], array.ndim)
        return np.squeeze(array, axis=axis)
    if op == "unsqueeze":
        axis = _axis(transform["axis"], array.ndim, insertion=True)
        return np.expand_dims(array, axis=axis)
    if op == "slice":
        axis = _axis(transform["axis"], array.ndim)
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(
            transform.get("start"), transform.get("end"), transform.get("step", 1)
        )
        return array[tuple(slices)]
    raise SkillError(f"{op!r} is not a unary transform")


def _source_array(checkpoint: dict[str, Any], source: str) -> tuple[np.ndarray, str]:
    tensor = checkpoint["tensors"][source]
    info = checkpoint["files"][tensor["file"]]
    return load_tensor(info, source), tensor["dtype"]


def _numpy_materialized_itemsize(dtype: str) -> int:
    # NumPy has no native BF16 storage in the dependency-free path.
    return 4 if dtype == "BF16" else DTYPE_BYTES[dtype]


def estimate_memory_budget(
    mapping: dict[str, Any], checkpoint: dict[str, Any], limit_bytes: int
) -> dict[str, int]:
    """Reject unsafe aggregate and working-set estimates before any tensor load."""
    if type(limit_bytes) is not int or limit_bytes <= 0:
        raise SkillError("--max-working-set-bytes must be a positive integer")
    source_total = 0
    target_storage_total = 0
    target_encoded_total = 0
    max_entry_source = 0
    max_entry_target_storage = 0
    max_entry_target_encoded = 0
    for entry in mapping["entries"]:
        source_dtypes = [
            checkpoint["tensors"][record["source"]]["dtype"]
            for record in entry["sources"]
        ]
        entry_source = sum(
            math.prod(record["shape"]) * _numpy_materialized_itemsize(source_dtype)
            for record, source_dtype in zip(entry["sources"], source_dtypes, strict=True)
        )
        source_total += entry_source
        max_entry_source = max(max_entry_source, entry_source)
        entry_target_storage = 0
        entry_target_encoded = 0
        for target in entry["targets"]:
            target_policy = _effective_policy(
                mapping["global_policy"], entry, entry["transforms"], target
            )
            if target_policy == "keep":
                if len(set(source_dtypes)) != 1:
                    raise SkillError(
                        "merge with dtype_policy keep requires identical source dtypes"
                    )
                output_dtype = source_dtypes[0]
            else:
                output_dtype = POLICY_TO_DTYPE[target_policy]
            elements = math.prod(target["shape"])
            entry_target_storage += elements * _numpy_materialized_itemsize(output_dtype)
            entry_target_encoded += elements * DTYPE_BYTES[output_dtype]
        target_storage_total += entry_target_storage
        target_encoded_total += entry_target_encoded
        max_entry_target_storage = max(max_entry_target_storage, entry_target_storage)
        max_entry_target_encoded = max(max_entry_target_encoded, entry_target_encoded)

    aggregate_materialized = source_total + target_storage_total
    materialization_peak = (
        target_storage_total + max_entry_source + 2 * max_entry_target_storage
    )
    streaming_writer_peak = (
        target_storage_total + max_entry_target_storage + max_entry_target_encoded
    )
    mlx_writer_peak = 3 * target_storage_total
    estimated_peak = max(materialization_peak, streaming_writer_peak, mlx_writer_peak)
    if aggregate_materialized > limit_bytes:
        raise SkillError(
            "aggregate materialization memory bound exceeded before allocation: "
            f"estimated {aggregate_materialized} bytes > limit {limit_bytes} bytes"
        )
    if estimated_peak > limit_bytes:
        raise SkillError(
            "working-set memory bound exceeded before allocation: "
            f"estimated peak {estimated_peak} bytes > limit {limit_bytes} bytes"
        )
    return {
        "limit_bytes": limit_bytes,
        "aggregate_materialized_bytes": aggregate_materialized,
        "estimated_peak_bytes": estimated_peak,
        "source_materialized_bytes": source_total,
        "target_materialized_bytes": target_storage_total,
        "target_encoded_bytes": target_encoded_total,
    }


def _cast_array(array: np.ndarray, source_dtype: str, policy: str) -> tuple[np.ndarray, str]:
    if policy == "keep":
        return np.asarray(array), source_dtype
    dtype = POLICY_TO_DTYPE[policy]
    if dtype == "F16":
        return np.asarray(array, dtype=np.float16), dtype
    return np.asarray(array, dtype=np.float32), dtype


def materialize_targets(
    mapping: dict[str, Any], checkpoint: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, int], dict[str, str]]:
    outputs: dict[str, np.ndarray] = {}
    output_dtypes: dict[str, str] = {}
    transform_counts: dict[str, int] = {}
    policies: dict[str, str] = {}
    for entry in mapping["entries"]:
        for transform in entry["transforms"]:
            if transform["op"] != "cast":
                transform_counts[transform["op"]] = transform_counts.get(transform["op"], 0) + 1
        if entry["kind"] == "single":
            array, source_dtype = _source_array(checkpoint, entry["sources"][0]["source"])
            for transform in entry["transforms"]:
                array = apply_unary_array(array, transform)
            target = entry["targets"][0]
            policy = _effective_policy(mapping["global_policy"], entry, entry["transforms"], target)
            array, dtype = _cast_array(array, source_dtype, policy)
            target_arrays = [(target, array, dtype, policy)]
        elif entry["kind"] == "split":
            array, source_dtype = _source_array(checkpoint, entry["sources"][0]["source"])
            pieces: list[np.ndarray] | None = None
            for transform in entry["transforms"]:
                if transform["op"] == "split":
                    axis = _axis(transform["axis"], array.ndim)
                    cuts = np.cumsum(transform["sizes"][:-1], dtype=np.int64).tolist()
                    pieces = list(np.split(array, cuts, axis=axis))
                elif pieces is None:
                    array = apply_unary_array(array, transform)
            assert pieces is not None
            target_arrays = []
            for target, piece in zip(entry["targets"], pieces, strict=True):
                policy = _effective_policy(mapping["global_policy"], entry, entry["transforms"], target)
                cast_piece, dtype = _cast_array(piece, source_dtype, policy)
                target_arrays.append((target, cast_piece, dtype, policy))
        else:
            source_arrays = [_source_array(checkpoint, item["source"]) for item in entry["sources"]]
            arrays = [item[0] for item in source_arrays]
            source_dtypes = [item[1] for item in source_arrays]
            merged: np.ndarray | None = None
            for transform in entry["transforms"]:
                if transform["op"] == "merge":
                    axis = _axis(transform["axis"], arrays[0].ndim)
                    merged = np.concatenate(arrays, axis=axis)
                elif merged is not None:
                    merged = apply_unary_array(merged, transform)
            assert merged is not None
            target = entry["targets"][0]
            policy = _effective_policy(mapping["global_policy"], entry, entry["transforms"], target)
            if policy == "keep" and len(set(source_dtypes)) != 1:
                raise SkillError("merge with dtype_policy keep requires identical source dtypes")
            merged, dtype = _cast_array(merged, source_dtypes[0], policy)
            target_arrays = [(target, merged, dtype, policy)]
        for target, array, dtype, policy in target_arrays:
            if list(array.shape) != target["shape"]:
                raise SkillError(
                    f"post-transform shape mismatch for {target['target']}: "
                    f"declared {target['shape']}, actual {list(array.shape)}"
                )
            outputs[target["target"]] = np.ascontiguousarray(array)
            output_dtypes[target["target"]] = dtype
            policies[target["target"]] = policy
            if policy != "keep":
                transform_counts["cast"] = transform_counts.get("cast", 0) + 1
    return outputs, output_dtypes, dict(sorted(transform_counts.items())), dict(sorted(policies.items()))


def _save_with_mlx(path: Path, tensors: dict[str, np.ndarray], dtypes: dict[str, str]) -> bool:
    if os.environ.get(OPTIONAL_LIBS_ENV) == "1":
        return False
    try:
        import mlx.core as mx  # type: ignore
    except ImportError:
        return False
    converted: dict[str, Any] = {}
    for key in sorted(tensors):
        if dtypes[key] == "BF16":
            converted[key] = mx.array(tensors[key], dtype=mx.bfloat16)
        else:
            converted[key] = mx.array(tensors[key])
    mx.save_safetensors(str(path), converted, metadata={"format": "mlx"})
    return True


def write_output_weights(
    path: Path, tensors: dict[str, np.ndarray], dtypes: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    if path.exists():
        raise SkillError(f"refusing to overwrite existing output: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".converted-", suffix=".safetensors", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        if _save_with_mlx(temporary, tensors, dtypes):
            writer = "mlx.core.save_safetensors"
        else:
            write_safetensors_pure(temporary, tensors, dtypes, metadata={"format": "mlx"})
            writer = "pure-python-safetensors"
        validated = read_safetensors_header(temporary)
        if set(validated["tensors"]) != set(tensors):
            raise SkillError("written safetensors manifest does not match converted targets")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise SkillError(f"refusing to overwrite existing output: {path.name}") from exc
        digest = {**validated["digest"], "name": path.name}
        return writer, digest
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _manifest_tensor_map(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise SkillError(f"{label} must be an object")
    raw = value.get("tensors")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [{**spec, "key": key} for key, spec in raw.items() if isinstance(spec, dict)]
    else:
        raise SkillError(f"{label} must contain a tensors list or object")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SkillError(f"{label} tensors[{index}] must be an object")
        name = item.get("key", item.get("name"))
        name = _name(name, label=f"{label} tensors[{index}] name")
        if name in result:
            raise SkillError(f"{label} contains duplicate tensor {name!r}")
        result[name] = {
            "shape": _shape(item.get("shape"), label=f"{label} tensor {name!r} shape"),
            "dtype": item.get("dtype"),
        }
    return result


def emit_draft_map(source_path: Path, scaffold_path: Path) -> dict[str, Any]:
    source = _manifest_tensor_map(read_bounded_json(source_path, label="source manifest"), label="source manifest")
    target = _manifest_tensor_map(
        read_bounded_json(scaffold_path, label="scaffold manifest"), label="scaffold manifest"
    )
    entries: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for name in sorted(set(source) & set(target)):
        if source[name]["shape"] == target[name]["shape"]:
            entries.append({
                "source": name,
                "source_shape": source[name]["shape"],
                "target": name,
                "target_shape": target[name]["shape"],
                "transforms": [{"op": "rename"}],
            })
        else:
            unresolved.append({
                "source": name,
                "source_shape": source[name]["shape"],
                "target": name,
                "target_shape": target[name]["shape"],
                "reason": "exact name matched but shape differed",
            })
    for name in sorted(set(source) - set(target)):
        unresolved.append({
            "source": name,
            "source_shape": source[name]["shape"],
            "reason": "no exact target-name match",
        })
    for name in sorted(set(target) - set(source)):
        unresolved.append({
            "target": name,
            "target_shape": target[name]["shape"],
            "reason": "no exact source-name match",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "draft": True,
        "dtype_policy": "keep",
        "entries": entries,
        "ignore": [],
        "unresolved": unresolved,
    }


def run_conversion(
    source_path: Path,
    mapping_path: Path,
    output_path: Path,
    *,
    max_working_set_bytes: int = DEFAULT_MAX_WORKING_SET_BYTES,
) -> dict[str, Any]:
    checkpoint = discover_checkpoint(source_path)
    mapping_path = validate_existing_path(mapping_path, label="mapping", kind="file")
    raw_mapping, mapping_digest = read_bounded_json_capture(mapping_path, label="WEIGHT_MAP")
    mapping = parse_conversion_map(raw_mapping)
    _validate_mapping_against_source(mapping, checkpoint["tensors"])
    memory_budget = estimate_memory_budget(mapping, checkpoint, max_working_set_bytes)
    tensors, dtypes, transforms, policies = materialize_targets(mapping, checkpoint)
    output = prepare_output_directory(output_path)
    weights_path = output / "model.safetensors"
    manifest_path = output / "target-manifest.json"
    report_path = output / "conversion-report.json"
    for path in (weights_path, manifest_path, report_path):
        if path.exists() or path.is_symlink():
            raise SkillError(f"refusing to overwrite existing output: {path.name}")
    writer, weights_digest = write_output_weights(weights_path, tensors, dtypes)
    target_manifest = {
        "schema_version": 1,
        "format": "safetensors",
        "file": weights_path.name,
        "tensors": [
            {"key": key, "shape": list(tensors[key].shape), "dtype": dtypes[key]}
            for key in sorted(tensors)
        ],
    }
    manifest_bytes = strict_json_bytes(target_manifest)
    _write_new_bytes(manifest_path, manifest_bytes)
    manifest_digest = {
        "name": manifest_path.name,
        "size_bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "counts": {
            "ignored_source_tensors": len(mapping["ignore"]),
            "mapped_source_tensors": len(mapping["mapped_sources"]),
            "mapping_entries": len(mapping["entries"]),
            "source_tensors": len(checkpoint["tensors"]),
            "target_tensors": len(tensors),
        },
        "transforms_applied": transforms,
        "dtype_policy": {
            "global": mapping["global_policy"],
            "targets": policies,
        },
        "memory_budget": memory_budget,
        "inputs": {
            "mapping": mapping_digest,
            "source_files": checkpoint["input_digests"],
        },
        "outputs": {
            "weights": {**weights_digest, "writer": writer},
            "target_manifest": manifest_digest,
        },
    }
    _write_new_bytes(report_path, strict_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.emit_draft_map is not None:
            if args.mapping is not None:
                raise SkillError("--mapping is not used with --emit-draft-map")
            if not args.scaffold_manifest:
                raise SkillError("--emit-draft-map requires --scaffold-manifest")
            draft = emit_draft_map(Path(args.source), Path(args.scaffold_manifest))
            data = strict_json_bytes(draft)
            destination = args.emit_draft_map
            if destination == "-" and args.output:
                destination = args.output
            if destination == "-":
                sys.stdout.buffer.write(data)
            else:
                output_path = _validate_components(Path(destination), label="draft map output", allow_missing=True)
                if output_path.exists() or output_path.is_symlink():
                    raise SkillError(f"refusing to overwrite existing output: {output_path.name}")
                prepare_output_directory(output_path.parent)
                _write_new_bytes(output_path, data)
            return 0
        if args.scaffold_manifest is not None:
            raise SkillError("--scaffold-manifest is only valid with --emit-draft-map")
        if not args.mapping or not args.output:
            raise SkillError("conversion requires --mapping and --output")
        report = run_conversion(
            Path(args.source),
            Path(args.mapping),
            Path(args.output),
            max_working_set_bytes=args.max_working_set_bytes,
        )
        sys.stdout.buffer.write(strict_json_bytes(report))
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: conversion failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
