#!/usr/bin/env python3
"""Compare bounded NPY/NPZ tensor oracles with chunked numerical metrics."""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import stat
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from ast import literal_eval as parse_python_literal
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

from _common import SkillError, dump_json, load_structured, validate_comparison_tolerances


NPY_MAGIC = b"\x93NUMPY"
NPY_SINGLE_KEY = "array"
ZIP_EOCD = struct.Struct("<4s4H2IH")
ZIP_CENTRAL_FILE_HEADER = struct.Struct("<4s6H3I5H2I")
ZIP_MAX_COMMENT_BYTES = 65_535
MAX_TENSOR_FILE_BYTES = 1024 * 1024 * 1024
MAX_NPZ_MEMBERS = 1_024
MAX_NPZ_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_TENSOR_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_NPZ_COMPRESSION_RATIO = 1_000
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_NPY_DIMENSIONS = 64
IO_CHUNK_BYTES = 1024 * 1024
COMPARE_CHUNK_BYTES = 4 * 1024 * 1024
SUPPORTED_ZIP_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


@dataclass(frozen=True)
class NpyDescriptor:
    shape: tuple[int, ...]
    dtype: Any
    fortran_order: bool
    data_offset: int
    data_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensors in bounded .npy/.npz archives")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--mapping", help="JSON/YAML mapping of source key to target key")
    parser.add_argument("--include", help="Regex selecting source keys")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--cosine-min", type=float, default=-1.0, help="Fail if cosine is below this; -1 disables")
    parser.add_argument("--output")
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="Permit zero selected tensors")
    return parser.parse_args()


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise SkillError(f"{label} is truncated")
    return data


def _open_regular_tensor_file(path: Path) -> tuple[BinaryIO, int]:
    if path.is_symlink():
        raise SkillError(f"tensor input must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SkillError(f"could not open tensor input {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SkillError(f"tensor input must be a regular non-symlink file: {path}")
        if metadata.st_size > MAX_TENSOR_FILE_BYTES:
            raise SkillError(
                f"tensor file size limit exceeded for {path.name}: "
                f"{metadata.st_size} > {MAX_TENSOR_FILE_BYTES}"
            )
        return os.fdopen(descriptor, "rb"), metadata.st_size
    except BaseException:
        os.close(descriptor)
        raise


def _parse_npy_header(
    handle: BinaryIO,
    file_size: int,
    np: Any,
    *,
    label: str,
) -> NpyDescriptor:
    handle.seek(0)
    if _read_exact(handle, len(NPY_MAGIC), label) != NPY_MAGIC:
        raise SkillError(f"{label} is not an NPY payload")
    version_raw = _read_exact(handle, 2, label)
    version = (version_raw[0], version_raw[1])
    if version == (1, 0):
        header_size = struct.unpack("<H", _read_exact(handle, 2, label))[0]
        encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        header_size = struct.unpack("<I", _read_exact(handle, 4, label))[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise SkillError(f"{label} uses unsupported NPY version {version[0]}.{version[1]}")
    if header_size > MAX_NPY_HEADER_BYTES:
        raise SkillError(
            f"NPY header size limit exceeded for {label}: {header_size} > {MAX_NPY_HEADER_BYTES}"
        )
    header_raw = _read_exact(handle, header_size, label)
    try:
        header = parse_python_literal(header_raw.decode(encoding).strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise SkillError(f"{label} has an invalid NPY header") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise SkillError(f"{label} has an invalid NPY header shape")
    shape = header.get("shape")
    if (
        not isinstance(shape, tuple)
        or len(shape) > MAX_NPY_DIMENSIONS
        or any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape)
    ):
        raise SkillError(f"{label} has an invalid or over-dimensional NPY shape")
    fortran_order = header.get("fortran_order")
    if not isinstance(fortran_order, bool):
        raise SkillError(f"{label} has an invalid NPY storage order")
    try:
        dtype = np.dtype(header.get("descr"))
    except (TypeError, ValueError) as exc:
        raise SkillError(f"{label} has an invalid NPY dtype") from exc
    if dtype.hasobject or dtype.kind not in "buifc" or dtype.itemsize <= 0:
        raise SkillError(f"{label} must contain a non-object numeric or boolean NPY dtype")

    if any(dimension > MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES for dimension in shape):
        raise SkillError(f"NPY shape exceeds the bounded member limit for {label}")
    if 0 in shape:
        element_count = 0
    else:
        element_count = 1
        max_elements = MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES // dtype.itemsize
        for dimension in shape:
            if dimension and element_count > max_elements // dimension:
                raise SkillError(
                    f"NPY tensor expansion limit exceeded for {label}: "
                    f"more than {MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES} data bytes"
                )
            element_count *= dimension
    data_bytes = element_count * dtype.itemsize
    data_offset = handle.tell()
    if data_offset + data_bytes != file_size:
        relation = "truncated" if data_offset + data_bytes > file_size else "has trailing bytes"
        raise SkillError(f"{label} {relation} relative to its declared NPY shape")
    return NpyDescriptor(
        shape=shape,
        dtype=dtype,
        fortran_order=fortran_order,
        data_offset=data_offset,
        data_bytes=data_bytes,
    )


def _find_zip_eocd(handle: BinaryIO, file_size: int, label: str) -> tuple[int, tuple[Any, ...]]:
    if file_size < ZIP_EOCD.size:
        raise SkillError(f"{label} is missing a ZIP end-of-central-directory record")
    tail_size = min(file_size, ZIP_EOCD.size + ZIP_MAX_COMMENT_BYTES)
    handle.seek(file_size - tail_size)
    tail = _read_exact(handle, tail_size, label)
    cursor = len(tail)
    while True:
        relative = tail.rfind(b"PK\x05\x06", 0, cursor)
        if relative < 0:
            raise SkillError(f"{label} is missing a ZIP end-of-central-directory record")
        if relative + ZIP_EOCD.size <= len(tail):
            fields = ZIP_EOCD.unpack_from(tail, relative)
            comment_size = fields[-1]
            if relative + ZIP_EOCD.size + comment_size == len(tail):
                return file_size - tail_size + relative, fields
        cursor = relative


def _preflight_npz_central_directory(handle: BinaryIO, file_size: int, label: str) -> tuple[int, int]:
    eocd_offset, fields = _find_zip_eocd(handle, file_size, label)
    (
        signature,
        disk_number,
        central_directory_disk,
        entries_on_disk,
        entry_count,
        central_directory_size,
        central_directory_offset,
        _comment_size,
    ) = fields
    if signature != b"PK\x05\x06":
        raise SkillError(f"{label} has an invalid ZIP end-of-directory signature")
    if disk_number != 0 or central_directory_disk != 0 or entries_on_disk != entry_count:
        raise SkillError(f"{label} uses an unsupported multi-disk ZIP layout")
    if entry_count == 0xFFFF or central_directory_size == 0xFFFFFFFF or central_directory_offset == 0xFFFFFFFF:
        raise SkillError(f"{label} uses an unsupported ZIP64 central directory")
    if entry_count > MAX_NPZ_MEMBERS:
        raise SkillError(f"NPZ member limit exceeded for {label}: {entry_count} > {MAX_NPZ_MEMBERS}")
    if central_directory_size > MAX_NPZ_CENTRAL_DIRECTORY_BYTES:
        raise SkillError(
            f"NPZ central directory limit exceeded for {label}: "
            f"{central_directory_size} > {MAX_NPZ_CENTRAL_DIRECTORY_BYTES}"
        )
    if central_directory_offset + central_directory_size != eocd_offset:
        raise SkillError(f"{label} has inconsistent ZIP central-directory bounds")

    handle.seek(central_directory_offset)
    directory = _read_exact(handle, central_directory_size, label)
    cursor = 0
    observed_entries = 0
    total_uncompressed = 0
    while cursor < len(directory):
        if cursor + ZIP_CENTRAL_FILE_HEADER.size > len(directory):
            raise SkillError(f"{label} has a truncated ZIP central-directory member")
        member = ZIP_CENTRAL_FILE_HEADER.unpack_from(directory, cursor)
        if member[0] != b"PK\x01\x02":
            raise SkillError(f"{label} has an invalid ZIP central-directory member")
        flags = member[3]
        compression = member[4]
        compressed_size = member[8]
        uncompressed_size = member[9]
        filename_size, extra_size, comment_size = member[10:13]
        record_size = ZIP_CENTRAL_FILE_HEADER.size + filename_size + extra_size + comment_size
        if cursor + record_size > len(directory):
            raise SkillError(f"{label} has an out-of-bounds ZIP central-directory member")
        observed_entries += 1
        if observed_entries > MAX_NPZ_MEMBERS:
            raise SkillError(f"NPZ member limit exceeded for {label}: more than {MAX_NPZ_MEMBERS}")
        if flags & 0x1:
            raise SkillError(f"{label} contains an encrypted NPZ member")
        if compression not in SUPPORTED_ZIP_COMPRESSION:
            raise SkillError(f"{label} contains an unsupported NPZ compression method")
        if uncompressed_size > MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES:
            raise SkillError(
                f"NPZ member expansion limit exceeded for {label}: "
                f"{uncompressed_size} > {MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES}"
            )
        total_uncompressed += uncompressed_size
        if total_uncompressed > MAX_TENSOR_TOTAL_UNCOMPRESSED_BYTES:
            raise SkillError(
                f"NPZ total expansion limit exceeded for {label}: "
                f"{total_uncompressed} > {MAX_TENSOR_TOTAL_UNCOMPRESSED_BYTES}"
            )
        if uncompressed_size and (
            compressed_size == 0 or uncompressed_size / compressed_size > MAX_NPZ_COMPRESSION_RATIO
        ):
            raise SkillError(f"NPZ compression ratio is suspicious for {label}")
        cursor += record_size
    if observed_entries != entry_count:
        raise SkillError(f"{label} ZIP member count disagrees with its central directory")
    return entry_count, total_uncompressed


def _validated_npz_members(
    archive: zipfile.ZipFile,
    *,
    expected_count: int,
    expected_total: int,
    label: str,
) -> dict[str, zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) != expected_count or len(members) > MAX_NPZ_MEMBERS:
        raise SkillError(f"{label} ZIP member count changed after preflight")
    result: dict[str, zipfile.ZipInfo] = {}
    total_uncompressed = 0
    for member in members:
        member_path = PurePosixPath(member.filename)
        if (
            member.is_dir()
            or member_path.is_absolute()
            or ".." in member_path.parts
            or str(member_path) != member.filename
            or not member.filename.endswith(".npy")
            or len(member.filename) <= 4
        ):
            raise SkillError(f"unsafe or non-NPY member in {label}: {member.filename}")
        unix_mode = member.external_attr >> 16
        if unix_mode & 0o170000 == 0o120000:
            raise SkillError(f"symlink member in {label}: {member.filename}")
        if member.flag_bits & 0x1 or member.compress_type not in SUPPORTED_ZIP_COMPRESSION:
            raise SkillError(f"unsupported member encoding in {label}: {member.filename}")
        if member.file_size > MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES:
            raise SkillError(
                f"NPZ member expansion limit exceeded for {member.filename}: "
                f"{member.file_size} > {MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES}"
            )
        total_uncompressed += member.file_size
        if total_uncompressed > MAX_TENSOR_TOTAL_UNCOMPRESSED_BYTES:
            raise SkillError(
                f"NPZ total expansion limit exceeded for {label}: "
                f"{total_uncompressed} > {MAX_TENSOR_TOTAL_UNCOMPRESSED_BYTES}"
            )
        if member.file_size and (
            member.compress_size == 0 or member.file_size / member.compress_size > MAX_NPZ_COMPRESSION_RATIO
        ):
            raise SkillError(f"NPZ compression ratio is suspicious for {member.filename}")
        key = member.filename[:-4]
        if key in result:
            raise SkillError(f"duplicate NPZ tensor key in {label}: {key}")
        result[key] = member
    if total_uncompressed != expected_total:
        raise SkillError(f"{label} ZIP expansion metadata changed after preflight")
    return result


@contextlib.contextmanager
def _mapped_npy(handle: BinaryIO, descriptor: NpyDescriptor, np: Any) -> Iterator[Any]:
    if descriptor.data_bytes == 0:
        yield np.empty(
            descriptor.shape,
            dtype=descriptor.dtype,
            order="F" if descriptor.fortran_order else "C",
        )
        return
    mapped = np.memmap(
        handle,
        dtype=descriptor.dtype,
        mode="r",
        offset=descriptor.data_offset,
        shape=descriptor.shape,
        order="F" if descriptor.fortran_order else "C",
    )
    try:
        yield mapped
    finally:
        mmap_handle = getattr(mapped, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


class TensorArchive:
    def __init__(self, path: str | Path, np: Any) -> None:
        self.path = Path(path).expanduser()
        self.np = np
        self.handle: BinaryIO | None = None
        self.file_size = 0
        self.kind = ""
        self.npy_descriptor: NpyDescriptor | None = None
        self.zip_archive: zipfile.ZipFile | None = None
        self.members: dict[str, zipfile.ZipInfo] = {}

    def __enter__(self) -> "TensorArchive":
        self.handle, self.file_size = _open_regular_tensor_file(self.path)
        try:
            self.handle.seek(0)
            prefix = self.handle.read(len(NPY_MAGIC))
            if prefix == NPY_MAGIC:
                if self.file_size > MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES:
                    raise SkillError(
                        f"NPY member expansion limit exceeded for {self.path.name}: "
                        f"{self.file_size} > {MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES}"
                    )
                self.kind = "npy"
                self.npy_descriptor = _parse_npy_header(
                    self.handle,
                    self.file_size,
                    self.np,
                    label=self.path.name,
                )
            elif prefix[:2] == b"PK":
                self.kind = "npz"
                expected_count, expected_total = _preflight_npz_central_directory(
                    self.handle,
                    self.file_size,
                    self.path.name,
                )
                self.handle.seek(0)
                self.zip_archive = zipfile.ZipFile(self.handle, mode="r")
                self.members = _validated_npz_members(
                    self.zip_archive,
                    expected_count=expected_count,
                    expected_total=expected_total,
                    label=self.path.name,
                )
            else:
                raise SkillError(f"tensor input is neither NPY nor NPZ: {self.path.name}")
            return self
        except (SkillError, OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            if isinstance(exc, SkillError):
                raise
            raise SkillError(f"could not preflight tensor input {self.path.name}: {exc}") from exc

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.zip_archive is not None:
            self.zip_archive.close()
            self.zip_archive = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    @property
    def files(self) -> list[str]:
        return [NPY_SINGLE_KEY] if self.kind == "npy" else list(self.members)

    @contextlib.contextmanager
    def open_array(self, key: str) -> Iterator[Any]:
        if self.handle is None:
            raise SkillError("tensor archive is closed")
        if self.kind == "npy":
            if key != NPY_SINGLE_KEY or self.npy_descriptor is None:
                raise SkillError(f"missing NPY tensor key: {key}")
            with _mapped_npy(self.handle, self.npy_descriptor, self.np) as array:
                yield array
            return
        member = self.members.get(key)
        if member is None or self.zip_archive is None:
            raise SkillError(f"missing NPZ tensor key: {key}")
        try:
            with tempfile.TemporaryFile() as extracted:
                observed_size = 0
                with self.zip_archive.open(member, mode="r") as source:
                    while chunk := source.read(IO_CHUNK_BYTES):
                        observed_size += len(chunk)
                        if (
                            observed_size > member.file_size
                            or observed_size > MAX_TENSOR_MEMBER_UNCOMPRESSED_BYTES
                        ):
                            raise SkillError(f"NPZ member expanded beyond its bound: {member.filename}")
                        extracted.write(chunk)
                if observed_size != member.file_size:
                    raise SkillError(f"NPZ member size disagrees with metadata: {member.filename}")
                extracted.flush()
                descriptor = _parse_npy_header(
                    extracted,
                    observed_size,
                    self.np,
                    label=f"{self.path.name}:{member.filename}",
                )
                with _mapped_npy(extracted, descriptor, self.np) as array:
                    yield array
        except (SkillError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            if isinstance(exc, SkillError):
                raise
            raise SkillError(f"could not read NPZ member {member.filename}: {exc}") from exc


def compare_arrays(a: Any, b: Any, np: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    if a.shape != b.shape:
        raise SkillError("compare_arrays requires equal tensor shapes")
    accumulator_dtype = np.dtype(
        np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64
    )
    chunk_elements = max(
        1,
        COMPARE_CHUNK_BYTES // max(accumulator_dtype.itemsize, a.dtype.itemsize, b.dtype.itemsize),
    )
    iterator = np.nditer(
        [a, b],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        order="C",
        buffersize=chunk_elements,
    )
    finite = True
    allclose = True
    arrays_equal = True
    element_count = 0
    chunk_count = 0
    abs_sum = 0.0
    max_abs = 0.0
    max_rel = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0
    dot: complex | float = 0j if accumulator_dtype.kind == "c" else 0.0
    source_integer = a.dtype.kind in "bui"
    target_integer = b.dtype.kind in "bui"
    integer_pair = source_integer and target_integer
    mixed_integer_category = source_integer != target_integer
    with np.errstate(all="ignore"):
        for source_chunk, target_chunk in iterator:
            chunk_count += 1
            # NumPy may choose float64 as the common comparison dtype for an
            # integer/float pair and collapse distinct uint64 values. Treat an
            # integer-vs-floating/complex representation as incompatible; the
            # caller can normalize dtypes explicitly before parity comparison.
            chunk_equal = not mixed_integer_category and bool(
                np.array_equal(source_chunk, target_chunk)
            )
            arrays_equal = arrays_equal and chunk_equal
            aa = source_chunk.astype(accumulator_dtype, copy=False)
            bb = target_chunk.astype(accumulator_dtype, copy=False)
            finite = finite and bool(np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)))
            diff = np.abs(aa - bb)
            if integer_pair and not chunk_equal:
                # float64 cannot distinguish every int64/uint64 value. Integer
                # parity is exact, and every exact mismatch has an absolute
                # difference of at least one even when the reporting cast
                # rounds adjacent large values together.
                mismatch = np.not_equal(source_chunk, target_chunk)
                diff = np.maximum(diff, mismatch)
            source_abs = np.abs(aa)
            target_abs = np.abs(bb)
            element_count += int(diff.size)
            if diff.size:
                chunk_max_abs = float(np.max(diff))
                max_abs = math.nan if math.isnan(chunk_max_abs) else max(max_abs, chunk_max_abs)
                abs_sum += float(np.sum(diff, dtype=np.float64))
                relative = diff / np.maximum(source_abs, 1e-30)
                chunk_max_rel = float(np.max(relative))
                max_rel = math.nan if math.isnan(chunk_max_rel) else max(max_rel, chunk_max_rel)
                if integer_pair or mixed_integer_category:
                    allclose = allclose and chunk_equal
                else:
                    allclose = allclose and bool(
                        np.all(diff <= (atol + rtol * target_abs))
                    )
                norm_a_squared += float(np.sum(source_abs * source_abs, dtype=np.float64))
                norm_b_squared += float(np.sum(target_abs * target_abs, dtype=np.float64))
                dot += np.sum(np.conjugate(aa) * bb)
    if element_count != int(a.size):
        raise SkillError("chunked tensor iteration did not cover the complete tensor")
    norm_a = math.sqrt(norm_a_squared)
    norm_b = math.sqrt(norm_b_squared)
    norm = norm_a * norm_b
    cosine = float(np.real(dot) / norm) if norm else (1.0 if arrays_equal else 0.0)
    return {
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": abs_sum / element_count if element_count else 0.0,
        "max_rel": max_rel,
        "cosine": cosine,
        "allclose": allclose,
        "dtype_compatible": not mixed_integer_category,
        "chunks": chunk_count,
    }


def _json_metric(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _format_metric(value: float, digits: int) -> str:
    numeric = float(value)
    return f"{numeric:.{digits}g}" if math.isfinite(numeric) else "non-finite"


def main() -> int:
    args = parse_args()
    try:
        try:
            import numpy as np
        except ImportError as exc:
            raise SkillError("numpy is required for compare_tensors.py") from exc
        validate_comparison_tolerances(args.atol, args.rtol, args.cosine_min)
        mapping: dict[str, str] = {}
        if args.mapping:
            raw = load_structured(args.mapping)
            if isinstance(raw, dict) and "mapping" in raw:
                raw = raw["mapping"]
            if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                raise SkillError("mapping must be an object of source-key -> target-key")
            mapping = raw
        pattern = re.compile(args.include) if args.include else None
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        with TensorArchive(args.source, np) as src, TensorArchive(args.target, np) as dst:
            source_keys = [key for key in src.files if pattern is None or pattern.search(key)]
            if not source_keys and not args.allow_empty:
                failures.append("no source tensors selected; use --allow-empty only for intentional report-only checks")
            for key in source_keys:
                target_key = mapping.get(key, key)
                if target_key not in dst.files:
                    failures.append(f"missing target key {target_key} for source {key}")
                    rows.append({"source": key, "target": target_key, "ok": False, "error": "missing target"})
                    continue
                with src.open_array(key) as a, dst.open_array(target_key) as b:
                    if a.shape != b.shape:
                        failures.append(f"shape mismatch {key} {a.shape} vs {target_key} {b.shape}")
                        rows.append({
                            "source": key,
                            "target": target_key,
                            "source_shape": list(a.shape),
                            "target_shape": list(b.shape),
                            "ok": False,
                            "error": "shape mismatch",
                        })
                        continue
                    metrics = compare_arrays(a, b, np, atol=args.atol, rtol=args.rtol)
                    cosine_ok = args.cosine_min < 0 or metrics["cosine"] >= args.cosine_min
                    ok = (
                        metrics["finite"]
                        and metrics["dtype_compatible"]
                        and metrics["allclose"]
                        and cosine_ok
                    )
                    row = {
                        "source": key,
                        "target": target_key,
                        "shape": list(a.shape),
                        "source_dtype": str(a.dtype),
                        "target_dtype": str(b.dtype),
                        "finite": metrics["finite"],
                        "dtype_compatible": metrics["dtype_compatible"],
                        "max_abs": _json_metric(metrics["max_abs"]),
                        "mean_abs": _json_metric(metrics["mean_abs"]),
                        "max_rel": _json_metric(metrics["max_rel"]),
                        "cosine": _json_metric(metrics["cosine"]),
                        "allclose": metrics["allclose"],
                        "ok": ok,
                    }
                    rows.append(row)
                    if not ok:
                        mismatch_kind = (
                            "dtype-category mismatch"
                            if not metrics["dtype_compatible"]
                            else "numerical mismatch"
                        )
                        failures.append(
                            f"{mismatch_kind} {key}: "
                            f"max_abs={_format_metric(metrics['max_abs'], 6)}, "
                            f"max_rel={_format_metric(metrics['max_rel'], 6)}, "
                            f"cosine={_format_metric(metrics['cosine'], 8)}"
                        )
            unmapped_target = sorted(set(dst.files) - {mapping.get(key, key) for key in source_keys})
        report = {
            "schema_version": 1,
            "ok": not failures,
            "atol": args.atol,
            "rtol": args.rtol,
            "cosine_min": args.cosine_min,
            "compared": len(rows),
            "rows": rows,
            "uncompared_target_keys": unmapped_target,
            "failures": failures,
            "report_only": bool(args.no_fail),
        }
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        return 0 if (not failures or args.no_fail) else 1
    except (SkillError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
