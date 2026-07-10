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


SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSORS = 1_000_000
MAX_DIRECTORY_ENTRIES = 20_000
MAX_DIMENSIONS = 64
MAX_DIM = (1 << 63) - 1
MAX_TENSOR_BYTES = 16 * 1024 * 1024 * 1024
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
DTYPE_POLICIES = {"keep", "f16", "bf16", "f32"}
POLICY_TO_DTYPE = {"f16": "F16", "bf16": "BF16", "f32": "F32"}


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
    return parser.parse_args(argv)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


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


def read_bounded_json(path: Path, *, label: str) -> Any:
    path = validate_existing_path(path, label=label, kind="file")
    descriptor, metadata = _open_regular_nofollow(path, label)
    try:
        if metadata.st_size > MAX_JSON_BYTES:
            raise SkillError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
        raw = _read_exact(descriptor, metadata.st_size, label)
        _verify_snapshot(path, descriptor, metadata, label)
    finally:
        os.close(descriptor)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except SkillError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def hash_regular_file(path: Path, *, label: str) -> dict[str, Any]:
    descriptor, metadata = _open_regular_nofollow(path, label)
    digest = hashlib.sha256()
    try:
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
    return {"name": path.name, "size_bytes": metadata.st_size, "sha256": digest.hexdigest()}


def _shape(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_DIMENSIONS
        or not all(type(dim) is int and 0 <= dim <= MAX_DIM for dim in value)
    ):
        raise SkillError(f"{label} must be a bounded list of non-negative integers")
    elements = math.prod(value)
    if elements > MAX_DIM:
        raise SkillError(f"{label} exceeds the element-count limit")
    return list(value)


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
        _verify_snapshot(path, descriptor, metadata, label)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw_header.decode("utf-8"), object_pairs_hook=_strict_object)
    except SkillError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"invalid safetensors header JSON in {label}: {exc}") from exc
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
    payloads: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    for key in sorted(tensors):
        if not isinstance(key, str) or not key:
            raise SkillError("safetensors output keys must be non-empty strings")
        array = np.asarray(tensors[key])
        dtype = dtypes[key] if dtypes is not None else NUMPY_TO_SAFETENSORS.get(array.dtype)
        if dtype not in DTYPE_BYTES:
            raise SkillError(f"unsupported output dtype for tensor {key!r}: {dtype!r}")
        raw = _encode_array(array, str(dtype))
        payloads[key] = raw
        records[key] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
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
    body = struct.pack("<Q", len(raw_header)) + raw_header + b"".join(payloads[key] for key in sorted(payloads))
    _write_new_bytes(path, body)


def read_safetensors_index(path: Path) -> dict[str, Any]:
    value = read_bounded_json(path, label="safetensors shard index")
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
    return value


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
    if index_paths:
        index_path = index_paths[0]
        index_value = read_safetensors_index(index_path)
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
    input_digests = [hash_regular_file(path, label=path.name) for path in safetensor_paths]
    if index_paths:
        input_digests.append(hash_regular_file(index_paths[0], label=index_paths[0].name))
    return {
        "root": source,
        "files": files,
        "tensors": tensors,
        "input_digests": sorted(input_digests, key=lambda item: item["name"]),
    }


def _require_fields(value: Any, required: set[str], optional: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SkillError(f"{label} must be an object")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise SkillError(f"{label} has invalid fields; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SkillError(f"{label} must be a non-empty string")
    return value


def _policy(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value not in DTYPE_POLICIES:
        raise SkillError(f"{label} must be one of {sorted(DTYPE_POLICIES)}")
    return value


def validate_transforms(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SkillError(f"{label} must be a list")
    result: list[dict[str, Any]] = []
    schemas = {
        "identity": (set(), set()),
        "rename": (set(), set()),
        "transpose": ({"axes"}, set()),
        "permute": ({"axes"}, set()),
        "reshape": ({"shape"}, set()),
        "squeeze": ({"axis"}, set()),
        "unsqueeze": ({"axis"}, set()),
        "slice": ({"axis"}, {"start", "end", "step"}),
        "cast": ({"dtype"}, set()),
        "split": ({"axis", "sizes"}, set()),
        "merge": ({"axis"}, set()),
    }
    for index, transform in enumerate(value):
        if not isinstance(transform, dict) or not isinstance(transform.get("op"), str):
            raise SkillError(f"{label}[{index}] must have a string op")
        op = transform["op"]
        if op not in schemas:
            raise SkillError(f"{label}[{index}] has unsupported transform op {op!r}")
        required, optional = schemas[op]
        _require_fields(transform, {"op", *required}, optional, label=f"{label}[{index}]")
        if op in {"transpose", "permute"}:
            axes = transform["axes"]
            if not isinstance(axes, list) or not all(type(axis) is int for axis in axes):
                raise SkillError(f"{label}[{index}].axes must be an integer list")
        elif op == "reshape":
            shape = transform["shape"]
            if not isinstance(shape, list) or not all(type(dim) is int and dim >= -1 for dim in shape):
                raise SkillError(f"{label}[{index}].shape must contain integers >= -1")
        elif op in {"squeeze", "unsqueeze", "split", "merge", "slice"}:
            if type(transform["axis"]) is not int:
                raise SkillError(f"{label}[{index}].axis must be an integer")
        if op == "slice":
            for field in ("start", "end", "step"):
                if field in transform and type(transform[field]) is not int:
                    raise SkillError(f"{label}[{index}].{field} must be an integer")
            if transform.get("step", 1) <= 0:
                raise SkillError(f"{label}[{index}].step must be positive")
        elif op == "cast":
            _policy(transform["dtype"], label=f"{label}[{index}].dtype")
            if transform["dtype"] == "keep":
                raise SkillError(f"{label}[{index}] cast dtype must name a concrete float dtype")
        elif op == "split":
            sizes = transform["sizes"]
            if not isinstance(sizes, list) or not sizes or not all(type(size) is int and size >= 0 for size in sizes):
                raise SkillError(f"{label}[{index}].sizes must be a non-empty non-negative integer list")
        result.append(dict(transform))
    return result


def _infer_reshape(old: list[int], new: list[int]) -> list[int]:
    if new.count(-1) > 1:
        raise SkillError(f"reshape has more than one -1: {new}")
    elements = math.prod(old)
    result = list(new)
    if -1 in result:
        known = math.prod(dim for dim in result if dim != -1)
        if known == 0 or elements % known:
            raise SkillError(f"cannot infer reshape {old} -> {new}")
        result[result.index(-1)] = elements // known
    if math.prod(result) != elements:
        raise SkillError(f"reshape changes element count: {old} -> {result}")
    return result


def _axis(axis: int, rank: int, *, insertion: bool = False) -> int:
    limit = rank + 1 if insertion else rank
    if axis < 0:
        axis += limit
    if axis < 0 or axis >= limit:
        raise SkillError(f"axis {axis} is out of range for rank {rank}")
    return axis


def apply_unary_shape(shape: list[int], transform: dict[str, Any]) -> list[int]:
    current = list(shape)
    op = transform["op"]
    if op in {"identity", "rename", "cast"}:
        return current
    if op in {"transpose", "permute"}:
        axes = transform["axes"]
        if sorted(axes) != list(range(len(current))):
            raise SkillError(f"invalid axes {axes} for shape {current}")
        return [current[index] for index in axes]
    if op == "reshape":
        return _infer_reshape(current, transform["shape"])
    if op == "squeeze":
        axis = _axis(transform["axis"], len(current))
        if current[axis] != 1:
            raise SkillError(f"cannot squeeze non-unit axis {axis} of {current}")
        current.pop(axis)
        return current
    if op == "unsqueeze":
        axis = _axis(transform["axis"], len(current), insertion=True)
        current.insert(axis, 1)
        return current
    if op == "slice":
        axis = _axis(transform["axis"], len(current))
        start, end, step = slice(
            transform.get("start"), transform.get("end"), transform.get("step", 1)
        ).indices(current[axis])
        current[axis] = len(range(start, end, step))
        return current
    raise SkillError(f"{op!r} is not a unary transform")


def apply_unary_array(array: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    op = transform["op"]
    if op in {"identity", "rename", "cast"}:
        return array
    if op in {"transpose", "permute"}:
        return np.transpose(array, transform["axes"])
    if op == "reshape":
        return np.reshape(array, _infer_reshape(list(array.shape), transform["shape"]))
    if op == "squeeze":
        return np.squeeze(array, axis=transform["axis"])
    if op == "unsqueeze":
        return np.expand_dims(array, axis=transform["axis"])
    if op == "slice":
        slices = [slice(None)] * array.ndim
        slices[transform["axis"]] = slice(
            transform.get("start"), transform.get("end"), transform.get("step", 1)
        )
        return array[tuple(slices)]
    raise SkillError(f"{op!r} is not a unary transform")


def _effective_policy(
    global_policy: str,
    entry: dict[str, Any],
    transforms: list[dict[str, Any]],
    target: dict[str, Any] | None = None,
) -> str:
    casts = [transform["dtype"] for transform in transforms if transform["op"] == "cast"]
    if len(set(casts)) > 1:
        raise SkillError("one mapping entry contains conflicting cast transforms")
    explicit = target.get("dtype_policy") if target is not None else None
    if explicit is None:
        explicit = entry.get("dtype_policy")
    if explicit is not None:
        explicit = _policy(explicit, label="entry dtype_policy")
    if casts and explicit is not None and casts[0] != explicit:
        raise SkillError("cast transform conflicts with the entry dtype_policy")
    return str(explicit or (casts[0] if casts else global_policy))


def parse_conversion_map(value: Any) -> dict[str, Any]:
    mapping = _require_fields(
        value,
        {"schema_version", "dtype_policy", "entries", "ignore", "unresolved"},
        {"draft"},
        label="WEIGHT_MAP",
    )
    if type(mapping["schema_version"]) is not int or mapping["schema_version"] != SCHEMA_VERSION:
        raise SkillError(f"WEIGHT_MAP schema_version must be integer {SCHEMA_VERSION}")
    if mapping.get("draft") is True:
        raise SkillError("refusing to convert with a draft WEIGHT_MAP")
    if "draft" in mapping and not isinstance(mapping["draft"], bool):
        raise SkillError("WEIGHT_MAP draft must be boolean")
    global_policy = _policy(mapping["dtype_policy"], label="WEIGHT_MAP dtype_policy")
    if not isinstance(mapping["unresolved"], list):
        raise SkillError("WEIGHT_MAP unresolved must be a list")
    if mapping["unresolved"]:
        raise SkillError("refusing to convert with non-empty WEIGHT_MAP unresolved entries")
    if not isinstance(mapping["ignore"], list):
        raise SkillError("WEIGHT_MAP ignore must be a list")
    ignore: list[dict[str, str]] = []
    ignored_names: set[str] = set()
    for index, item in enumerate(mapping["ignore"]):
        record = _require_fields(item, {"source", "reason"}, set(), label=f"WEIGHT_MAP ignore[{index}]")
        source = _name(record["source"], label=f"WEIGHT_MAP ignore[{index}].source")
        reason = record["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise SkillError(f"WEIGHT_MAP ignore[{index}].reason must be non-empty")
        if source in ignored_names:
            raise SkillError(f"duplicate ignored source tensor: {source}")
        ignored_names.add(source)
        ignore.append({"source": source, "reason": reason.strip()})
    if not isinstance(mapping["entries"], list) or not mapping["entries"]:
        raise SkillError("WEIGHT_MAP entries must be a non-empty list")
    entries: list[dict[str, Any]] = []
    mapped_sources: set[str] = set()
    mapped_targets: set[str] = set()
    for index, raw in enumerate(mapping["entries"]):
        label = f"WEIGHT_MAP entries[{index}]"
        if not isinstance(raw, dict):
            raise SkillError(f"{label} must be an object")
        if "sources" in raw:
            entry = _require_fields(
                raw,
                {"sources", "target", "target_shape", "transforms"},
                {"dtype_policy"},
                label=label,
            )
            kind = "merge"
            if not isinstance(entry["sources"], list) or len(entry["sources"]) < 2:
                raise SkillError(f"{label}.sources must contain at least two source records")
            sources = []
            for source_index, item in enumerate(entry["sources"]):
                record = _require_fields(item, {"source", "shape"}, set(), label=f"{label}.sources[{source_index}]")
                sources.append({
                    "source": _name(record["source"], label=f"{label}.sources[{source_index}].source"),
                    "shape": _shape(record["shape"], label=f"{label}.sources[{source_index}].shape"),
                })
            targets = [{
                "target": _name(entry["target"], label=f"{label}.target"),
                "shape": _shape(entry["target_shape"], label=f"{label}.target_shape"),
            }]
        elif "targets" in raw:
            entry = _require_fields(
                raw,
                {"source", "source_shape", "targets", "transforms"},
                {"dtype_policy"},
                label=label,
            )
            kind = "split"
            sources = [{
                "source": _name(entry["source"], label=f"{label}.source"),
                "shape": _shape(entry["source_shape"], label=f"{label}.source_shape"),
            }]
            if not isinstance(entry["targets"], list) or len(entry["targets"]) < 2:
                raise SkillError(f"{label}.targets must contain at least two target records")
            targets = []
            for target_index, item in enumerate(entry["targets"]):
                record = _require_fields(
                    item,
                    {"target", "shape"},
                    {"dtype_policy"},
                    label=f"{label}.targets[{target_index}]",
                )
                target_record: dict[str, Any] = {
                    "target": _name(record["target"], label=f"{label}.targets[{target_index}].target"),
                    "shape": _shape(record["shape"], label=f"{label}.targets[{target_index}].shape"),
                }
                if "dtype_policy" in record:
                    target_record["dtype_policy"] = _policy(
                        record["dtype_policy"], label=f"{label}.targets[{target_index}].dtype_policy"
                    )
                targets.append(target_record)
        else:
            entry = _require_fields(
                raw,
                {"source", "target", "source_shape", "target_shape", "transforms"},
                {"dtype_policy"},
                label=label,
            )
            kind = "single"
            sources = [{
                "source": _name(entry["source"], label=f"{label}.source"),
                "shape": _shape(entry["source_shape"], label=f"{label}.source_shape"),
            }]
            targets = [{
                "target": _name(entry["target"], label=f"{label}.target"),
                "shape": _shape(entry["target_shape"], label=f"{label}.target_shape"),
            }]
        transforms = validate_transforms(entry["transforms"], label=f"{label}.transforms")
        merge_count = sum(transform["op"] == "merge" for transform in transforms)
        split_count = sum(transform["op"] == "split" for transform in transforms)
        if kind == "merge" and (merge_count != 1 or split_count):
            raise SkillError(f"{label} grouped sources require exactly one merge transform")
        if kind == "split" and (split_count != 1 or merge_count):
            raise SkillError(f"{label} grouped targets require exactly one split transform")
        if kind == "single" and (merge_count or split_count):
            raise SkillError(f"{label} single-source entry cannot contain split or merge")
        if "dtype_policy" in entry:
            _policy(entry["dtype_policy"], label=f"{label}.dtype_policy")
        for source_record in sources:
            source_name = source_record["source"]
            if source_name in mapped_sources:
                raise SkillError(f"source tensor mapped more than once: {source_name}")
            if source_name in ignored_names:
                raise SkillError(f"source tensor is both mapped and ignored: {source_name}")
            mapped_sources.add(source_name)
        for target_record in targets:
            target_name = target_record["target"]
            if target_name in mapped_targets:
                raise SkillError(f"target tensor mapped more than once: {target_name}")
            mapped_targets.add(target_name)
        normalized = {
            "kind": kind,
            "sources": sources,
            "targets": targets,
            "transforms": transforms,
        }
        if "dtype_policy" in entry:
            normalized["dtype_policy"] = entry["dtype_policy"]
        entries.append(normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "global_policy": global_policy,
        "entries": entries,
        "ignore": ignore,
        "mapped_sources": mapped_sources,
        "mapped_targets": mapped_targets,
    }


def _validate_entry_shapes(entry: dict[str, Any]) -> list[list[int]]:
    transforms = entry["transforms"]
    if entry["kind"] == "single":
        current = list(entry["sources"][0]["shape"])
        for transform in transforms:
            current = apply_unary_shape(current, transform)
        return [current]
    if entry["kind"] == "split":
        current = list(entry["sources"][0]["shape"])
        outputs: list[list[int]] | None = None
        for transform in transforms:
            if transform["op"] == "split":
                axis = _axis(transform["axis"], len(current))
                sizes = transform["sizes"]
                if len(sizes) != len(entry["targets"]) or sum(sizes) != current[axis]:
                    raise SkillError("split sizes must match the source axis and target count")
                outputs = []
                for size in sizes:
                    shape = list(current)
                    shape[axis] = size
                    outputs.append(shape)
            elif outputs is None:
                current = apply_unary_shape(current, transform)
            elif transform["op"] not in {"identity", "rename", "cast"}:
                raise SkillError("only identity, rename, or cast may follow split")
        if outputs is None:
            raise SkillError("split entry is missing its split transform")
        return outputs
    shapes = [list(item["shape"]) for item in entry["sources"]]
    merged: list[int] | None = None
    for transform in transforms:
        if transform["op"] == "merge":
            axis = _axis(transform["axis"], len(shapes[0]))
            if any(len(shape) != len(shapes[0]) for shape in shapes):
                raise SkillError("merge sources must have the same rank")
            for shape in shapes[1:]:
                if any(shape[index] != shapes[0][index] for index in range(len(shape)) if index != axis):
                    raise SkillError("merge source shapes differ outside the merge axis")
            merged = list(shapes[0])
            merged[axis] = sum(shape[axis] for shape in shapes)
        elif merged is None:
            if transform["op"] not in {"identity", "rename", "cast"}:
                raise SkillError("only identity, rename, or cast may precede merge")
        else:
            merged = apply_unary_shape(merged, transform)
    if merged is None:
        raise SkillError("merge entry is missing its merge transform")
    return [merged]


def validate_mapping_against_source(mapping: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    source_names = set(checkpoint["tensors"])
    missing = sorted(mapping["mapped_sources"] - source_names)
    ignored = {item["source"] for item in mapping["ignore"]}
    unknown_ignored = sorted(ignored - source_names)
    unmapped = sorted(source_names - mapping["mapped_sources"] - ignored)
    if missing:
        raise SkillError(f"mapped source tensors are missing: {missing}")
    if unknown_ignored:
        raise SkillError(f"ignored source tensors are missing: {unknown_ignored}")
    if unmapped:
        raise SkillError(f"source tensors are unmapped and not ignored: {unmapped}")
    for entry in mapping["entries"]:
        for source_record in entry["sources"]:
            actual = checkpoint["tensors"][source_record["source"]]["shape"]
            if actual != source_record["shape"]:
                raise SkillError(
                    f"declared source shape mismatch for {source_record['source']}: "
                    f"mapping {source_record['shape']}, checkpoint {actual}"
                )
        actual_targets = _validate_entry_shapes(entry)
        for target_record, actual in zip(entry["targets"], actual_targets, strict=True):
            if actual != target_record["shape"]:
                raise SkillError(
                    f"post-transform shape mismatch for {target_record['target']}: "
                    f"declared {target_record['shape']}, computed {actual}"
                )


def _source_array(checkpoint: dict[str, Any], source: str) -> tuple[np.ndarray, str]:
    tensor = checkpoint["tensors"][source]
    info = checkpoint["files"][tensor["file"]]
    return load_tensor(info, source), tensor["dtype"]


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
                    axis = transform["axis"]
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
                    merged = np.concatenate(arrays, axis=transform["axis"])
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


def write_output_weights(path: Path, tensors: dict[str, np.ndarray], dtypes: dict[str, str]) -> str:
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
        return writer
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


def run_conversion(source_path: Path, mapping_path: Path, output_path: Path) -> dict[str, Any]:
    checkpoint = discover_checkpoint(source_path)
    mapping_path = validate_existing_path(mapping_path, label="mapping", kind="file")
    raw_mapping = read_bounded_json(mapping_path, label="WEIGHT_MAP")
    mapping = parse_conversion_map(raw_mapping)
    validate_mapping_against_source(mapping, checkpoint)
    tensors, dtypes, transforms, policies = materialize_targets(mapping, checkpoint)
    output = prepare_output_directory(output_path)
    weights_path = output / "model.safetensors"
    manifest_path = output / "target-manifest.json"
    report_path = output / "conversion-report.json"
    for path in (weights_path, manifest_path, report_path):
        if path.exists() or path.is_symlink():
            raise SkillError(f"refusing to overwrite existing output: {path.name}")
    writer = write_output_weights(weights_path, tensors, dtypes)
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
    mapping_digest = hash_regular_file(mapping_path, label="WEIGHT_MAP")
    weights_digest = hash_regular_file(weights_path, label=weights_path.name)
    manifest_digest = hash_regular_file(manifest_path, label=manifest_path.name)
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
        report = run_conversion(Path(args.source), Path(args.mapping), Path(args.output))
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
