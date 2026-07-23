#!/usr/bin/env python3
"""Generate and verify MANIFEST.json: a sha256 integrity record of every distributed file.

The manifest covers the whole repository tree (skill, tests, CI, docs) minus local
scaffolding and the manifest itself. `check` recomputes hashes from disk and fails loud
on any added, removed, or changed file so the integrity record cannot silently rot.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import MAX_STRUCTURED_BYTES, SkillError, dump_json

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = "mlx-porting-skill"
SCHEMA_VERSION = 1
MANIFEST_FIELDS = frozenset({
    "schema_version",
    "artifact",
    "version",
    "generated_at",
    "file_count",
    "files",
})
REGULAR_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256", "executable"})
SYMLINK_FIELDS = frozenset({"path", "type", "target", "size_bytes", "sha256"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", None)
DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", None)
OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
STAT_SUPPORTS_DIR_FD = os.stat in os.supports_dir_fd
READLINK_SUPPORTS_DIR_FD = os.readlink in os.supports_dir_fd
LISTDIR_SUPPORTS_FD = os.listdir in os.supports_fd
READ_CHUNK_BYTES = 1024 * 1024

# Mirrors .gitignore: local editor/agent scaffolding and build noise are not distributed.
EXCLUDE_DIRS = {
    ".git", ".superplan", ".agents", ".amazonq", ".claude", ".codex",
    ".cursor", ".gemini", ".opencode", ".pagecast", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", ".wrangler", "dist-pages",
}
EXCLUDE_NAMES = {".DS_Store", ".dev.vars", "MANIFEST.json"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log"}
# Atomic-write staging files the skill's own tools create and unlink in place.
# convert_checkpoint.py stages converted weights as ".converted-*.safetensors"
# before linking them atomically, so a check that races with an in-progress
# conversion must not try to hash a file that is actively being written. These
# are transient-by-construction and never part of the distribution, so skipping
# them cannot hide drift in a real shipped file (mirrors the existing ".tmp"
# staging exclusion that already covers atomic_write_text).
STAGING_PREFIX = ".converted-"
STAGING_SUFFIX = ".safetensors"
SHIPPED_SYMLINKS = {
    Path(".agents/skills/mlx-model-porting"),
    Path(".claude/skills/mlx-model-porting"),
}
MAX_VERSION_BYTES = 256


def _require_descriptor_support() -> None:
    if not isinstance(NOFOLLOW_FLAG, int) or NOFOLLOW_FLAG == 0:
        raise SkillError(
            "this platform does not provide O_NOFOLLOW; refusing to read distribution files"
        )
    if not isinstance(DIRECTORY_FLAG, int) or DIRECTORY_FLAG == 0:
        raise SkillError(
            "this platform does not provide O_DIRECTORY; refusing to traverse distribution files"
        )
    if (
        not OPEN_SUPPORTS_DIR_FD
        or not STAT_SUPPORTS_DIR_FD
        or not READLINK_SUPPORTS_DIR_FD
        or not LISTDIR_SUPPORTS_FD
    ):
        raise SkillError(
            "this platform lacks required directory-descriptor APIs; refusing to traverse distribution files"
        )


def _open_directory_at(parent_descriptor: int | None, name: str | Path, label: str) -> int:
    _require_descriptor_support()
    flags = os.O_RDONLY | int(NOFOLLOW_FLAG) | int(DIRECTORY_FLAG)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        if parent_descriptor is None:
            descriptor = os.open(name, flags)
        else:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SkillError(f"could not open {label} without following symlinks: {exc}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise SkillError(f"{label} must be a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_path(path: Path, label: str) -> int:
    """Pin every component of the canonical directory path with no-follow openat calls."""
    absolute = Path(os.path.realpath(path))
    if not absolute.anchor:
        raise SkillError(f"{label} must resolve to an absolute directory path")
    current = _open_directory_at(None, absolute.anchor, f"{label} anchor")
    try:
        for part in absolute.parts[1:]:
            child = _open_directory_at(current, part, f"{label} component {part}")
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_regular_nofollow_at(
    parent_descriptor: int,
    name: str,
    label: str,
) -> tuple[int, os.stat_result]:
    """Open one leaf through a pinned parent without following the leaf."""
    _require_descriptor_support()
    flags = os.O_RDONLY | int(NOFOLLOW_FLAG)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SkillError(f"{label} must not be a symlink") from exc
        raise SkillError(f"could not open {label} without following symlinks: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SkillError(f"{label} must be a regular file")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


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


def _verify_unchanged_descriptor(
    descriptor: int,
    parent_descriptor: int,
    name: str,
    initial: os.stat_result,
    label: str,
) -> None:
    final = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise SkillError(f"{label} changed while it was being read") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or not _same_snapshot(initial, final)
        or not _same_snapshot(initial, current)
    ):
        raise SkillError(f"{label} changed while it was being read")


def _read_regular_bytes_at(
    parent_descriptor: int,
    name: str,
    label: str,
    limit: int,
) -> bytes:
    descriptor, metadata = _open_regular_nofollow_at(parent_descriptor, name, label)
    try:
        if metadata.st_size > limit:
            raise SkillError(f"{label} exceeds size limit: {metadata.st_size} > {limit} bytes")
        remaining = metadata.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise SkillError(f"{label} changed while it was being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SkillError(f"{label} changed while it was being read")
        _verify_unchanged_descriptor(descriptor, parent_descriptor, name, metadata, label)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash_regular_file_at(
    parent_descriptor: int,
    name: str,
    label: str,
) -> tuple[os.stat_result, str]:
    descriptor, metadata = _open_regular_nofollow_at(parent_descriptor, name, label)
    digest = hashlib.sha256()
    try:
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise SkillError(f"{label} changed while it was being hashed")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SkillError(f"{label} changed while it was being hashed")
        _verify_unchanged_descriptor(descriptor, parent_descriptor, name, metadata, label)
        return metadata, digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_symlink_target(link_path: str, target: str, label: str) -> None:
    """Reject link text that is absolute or lexically walks above the artifact root."""
    portable_target = target.replace("\\", "/")
    drive, _ = os.path.splitdrive(target)
    if (
        not target
        or target.startswith(("/", "\\"))
        or drive
        or re.match(r"^[A-Za-z]:", portable_target)
    ):
        raise SkillError(f"{label} target must be a relative path inside the distribution root")

    destination_parts = link_path.split("/")[:-1]
    for part in portable_target.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not destination_parts:
                raise SkillError(f"{label} target escapes the distribution root")
            destination_parts.pop()
            continue
        destination_parts.append(part)


def _read_safe_symlink_target_at(
    parent_descriptor: int,
    name: str,
    relative_path: Path,
) -> str:
    label = f"symlink {relative_path}"
    initial = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISLNK(initial.st_mode):
        raise SkillError(f"{label} changed while it was being inspected")
    target = os.readlink(name, dir_fd=parent_descriptor)
    final = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISLNK(final.st_mode) or not _same_snapshot(initial, final):
        raise SkillError(f"{label} changed while it was being inspected")
    _validate_symlink_target(str(relative_path), target, label)
    return target


def _symlink_record_at(
    parent_descriptor: int,
    name: str,
    relative_path: Path,
) -> dict[str, Any]:
    target = _read_safe_symlink_target_at(parent_descriptor, name, relative_path)
    encoded_target = target.encode("utf-8")
    return {
        "path": str(relative_path),
        "type": "symlink",
        "target": target,
        "size_bytes": len(encoded_target),
        "sha256": hashlib.sha256(encoded_target).hexdigest(),
    }


def _special_node_kind(mode: int) -> str:
    if stat.S_ISFIFO(mode):
        return "FIFO"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character device"
    if stat.S_ISBLK(mode):
        return "block device"
    return "unsupported special node"


def _shipped_symlink_record(root_descriptor: int, relative_path: Path) -> dict[str, Any] | None:
    current = root_descriptor
    opened: list[int] = []
    try:
        for part in relative_path.parts[:-1]:
            try:
                current = _open_directory_at(current, part, f"directory {relative_path.parent}")
            except FileNotFoundError:
                return None
            opened.append(current)
        try:
            metadata = os.stat(
                relative_path.name,
                dir_fd=current,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return _symlink_record_at(current, relative_path.name, relative_path)
        if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise SkillError(
                f"unsupported {_special_node_kind(metadata.st_mode)} at {relative_path}"
            )
        return None
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _build_directory_records(
    directory_descriptor: int,
    relative_directory: Path,
    records: list[dict[str, Any]],
    yielded: set[Path],
) -> None:
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as exc:
        raise SkillError(f"could not enumerate directory {relative_directory or Path('.')}: {exc}") from exc
    for name in names:
        relative_path = relative_directory / name
        if relative_path in yielded:
            continue
        try:
            metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise SkillError(f"distribution entry changed during traversal: {relative_path}") from exc

        mode = metadata.st_mode
        excluded = (
            name in EXCLUDE_DIRS
            or name in EXCLUDE_NAMES
            or Path(name).suffix in EXCLUDE_SUFFIXES
            or (name.startswith(STAGING_PREFIX) and name.endswith(STAGING_SUFFIX))
        )
        if stat.S_ISDIR(mode):
            if excluded:
                continue
            child = _open_directory_at(
                directory_descriptor,
                name,
                f"directory {relative_path}",
            )
            try:
                _build_directory_records(child, relative_path, records, yielded)
            finally:
                os.close(child)
            continue
        if not stat.S_ISLNK(mode) and not stat.S_ISREG(mode):
            raise SkillError(f"unsupported {_special_node_kind(mode)} at {relative_path}")
        if excluded:
            continue
        if stat.S_ISLNK(mode):
            records.append(_symlink_record_at(directory_descriptor, name, relative_path))
            continue

        final_metadata, digest = _hash_regular_file_at(
            directory_descriptor,
            name,
            f"distributed file {relative_path}",
        )
        records.append({
            "path": str(relative_path),
            "size_bytes": final_metadata.st_size,
            "sha256": digest,
            "executable": bool(final_metadata.st_mode & 0o111),
        })


def build_files(root: Path) -> list[dict[str, Any]]:
    root_descriptor = _open_directory_path(root, "distribution root")
    try:
        records: list[dict[str, Any]] = []
        yielded: set[Path] = set()
        for relative_path in sorted(SHIPPED_SYMLINKS, key=str):
            record = _shipped_symlink_record(root_descriptor, relative_path)
            if record is not None:
                yielded.add(relative_path)
                records.append(record)
        _build_directory_records(root_descriptor, Path(), records, yielded)
        records.sort(key=lambda record: record["path"])
        return records
    finally:
        os.close(root_descriptor)


def read_version(root: Path) -> str:
    root_descriptor = _open_directory_path(root, "distribution root")
    try:
        try:
            raw = _read_regular_bytes_at(
                root_descriptor,
                "VERSION",
                "VERSION",
                MAX_VERSION_BYTES,
            )
        except FileNotFoundError:
            return "0.0.0"
    finally:
        os.close(root_descriptor)
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SkillError("VERSION must be valid UTF-8") from exc


def generate(root: Path) -> dict[str, Any]:
    files = build_files(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "version": read_version(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise SkillError(f"{label} fields are invalid: {'; '.join(details)}")


def _validate_file_record(record: Any, index: int) -> str:
    label = f"manifest files[{index}]"
    if not isinstance(record, dict):
        raise SkillError(f"{label} must be an object")

    record_type = record.get("type")
    expected_fields = SYMLINK_FIELDS if record_type == "symlink" else REGULAR_FILE_FIELDS
    _require_exact_fields(record, expected_fields, label)

    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise SkillError(f"{label}.path must be a non-empty string")
    path_parts = path.split("/")
    if path.startswith("/") or "\\" in path or any(part in {"", ".", ".."} for part in path_parts):
        raise SkillError(f"{label}.path must be a normalized relative POSIX path")

    size_bytes = record.get("size_bytes")
    if type(size_bytes) is not int or size_bytes < 0:
        raise SkillError(f"{label}.size_bytes must be a non-negative integer")

    digest = record.get("sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise SkillError(f"{label}.sha256 must be a lowercase 64-character SHA-256 digest")

    if record_type == "symlink":
        target = record.get("target")
        if not isinstance(target, str) or not target:
            raise SkillError(f"{label}.target must be a non-empty string")
        _validate_symlink_target(path, target, label)
    elif "type" in record:
        raise SkillError(f"{label}.type must be 'symlink' when present")
    elif type(record.get("executable")) is not bool:
        raise SkillError(f"{label}.executable must be a boolean")
    return path


def validate_manifest(payload: Any, root: Path) -> list[dict[str, Any]]:
    """Validate schema-1 metadata and return records safe for integrity diffing."""
    if not isinstance(payload, dict):
        raise SkillError("manifest must be a JSON object")
    _require_exact_fields(payload, MANIFEST_FIELDS, "manifest")

    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise SkillError(f"manifest schema_version must be integer {SCHEMA_VERSION}")

    artifact = payload.get("artifact")
    if not isinstance(artifact, str) or artifact != ARTIFACT:
        raise SkillError(f"manifest artifact must be {ARTIFACT!r}")

    expected_version = read_version(root)
    version = payload.get("version")
    if not isinstance(version, str) or version != expected_version:
        raise SkillError(f"manifest version must match VERSION ({expected_version!r})")

    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        raise SkillError("manifest generated_at must be a timezone-aware ISO-8601 string")
    try:
        generated_timestamp = datetime.fromisoformat(generated_at)
    except ValueError as exc:
        raise SkillError("manifest generated_at must be a timezone-aware ISO-8601 string") from exc
    if generated_timestamp.utcoffset() is None or generated_timestamp.utcoffset().total_seconds() != 0:
        raise SkillError("manifest generated_at must use UTC")

    files = payload.get("files")
    if not isinstance(files, list):
        raise SkillError("manifest files must be an array")
    file_count = payload.get("file_count")
    if type(file_count) is not int or file_count < 0:
        raise SkillError("manifest file_count must be a non-negative integer")
    if file_count != len(files):
        raise SkillError(
            f"manifest file_count does not match files length: {file_count} != {len(files)}"
        )

    paths = [_validate_file_record(record, index) for index, record in enumerate(files)]
    if paths != sorted(paths):
        raise SkillError("manifest files must be sorted by path")
    if len(paths) != len(set(paths)):
        raise SkillError("manifest files must contain unique paths")
    return files


def read_manifest(path: Path) -> Any:
    """Read a bounded regular JSON manifest without following a replacement symlink."""
    parent_descriptor = _open_directory_path(path.parent, "manifest parent")
    try:
        try:
            raw = _read_regular_bytes_at(
                parent_descriptor,
                path.name,
                "manifest",
                MAX_STRUCTURED_BYTES,
            )
        except FileNotFoundError as exc:
            raise SkillError(f"manifest not found: {path}; run `manifest.py generate` first") from exc
    finally:
        os.close(parent_descriptor)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError("manifest must be valid UTF-8") from exc
    return json.loads(text)


def diff(committed: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, list[str]]:
    committed_by = {f["path"]: f for f in committed}
    current_by = {f["path"]: f for f in current}
    added = sorted(set(current_by) - set(committed_by))
    removed = sorted(set(committed_by) - set(current_by))
    changed = sorted(
        p for p in set(committed_by) & set(current_by)
        if committed_by[p].get("sha256") != current_by[p].get("sha256")
        or committed_by[p].get("size_bytes") != current_by[p].get("size_bytes")
        or committed_by[p].get("type") != current_by[p].get("type")
        or committed_by[p].get("target") != current_by[p].get("target")
        or committed_by[p].get("executable") != current_by[p].get("executable")
    )
    return {"added": added, "removed": removed, "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify MANIFEST.json")
    parser.add_argument("action", choices=["generate", "check"])
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root to scan")
    parser.add_argument("--output", help="Manifest path (default <root>/MANIFEST.json)")
    try:
        args = parser.parse_args()
        root = Path(os.path.abspath(args.root))
        manifest_path = Path(args.output) if args.output else root / "MANIFEST.json"
        if args.action == "generate":
            report = generate(root)
            dump_json(report, manifest_path)
            print(f"wrote {report['file_count']} entries to {manifest_path}")
            return 0
        committed = validate_manifest(read_manifest(manifest_path), root)
        delta = diff(committed, build_files(root))
        drifted = any(delta.values())
        print(json.dumps({"ok": not drifted, "manifest": str(manifest_path), **delta}, indent=2))
        return 1 if drifted else 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
