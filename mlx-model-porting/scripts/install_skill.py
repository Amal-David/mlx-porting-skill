#!/usr/bin/env python3
"""Install the manifest-attested ``mlx-model-porting`` skill payload.

Interrupted force replacements are recovered only through a fsynced ownership
journal. Unjournaled backup-looking siblings are reported and left untouched.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

# Running the installer must not create an unlisted cache inside its own source
# tree before the manifest inventory is checked.
sys.dont_write_bytecode = True

from _common import SkillError  # noqa: E402
import manifest as manifest_contract  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR.parent
MANIFEST_NAME = "MANIFEST.json"
READ_CHUNK_BYTES = 1024 * 1024
NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
INSTALL_JOURNAL_SCHEMA_VERSION = 1
MAX_INSTALL_JOURNAL_BYTES = 64 * 1024

CLIENT_PRESETS = {
    "claude-code": (".claude/skills", "symlink"),
    "codex": (".agents/skills", "symlink"),
    "cursor": (".cursor/skills", "copy"),
    "gemini": (".gemini/skills", "copy"),
    "windsurf": (".windsurf/skills", "copy"),
    "copilot": (".github/skills", "copy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the manifest-attested mlx-model-porting skill"
    )
    dest_group = parser.add_mutually_exclusive_group(required=True)
    dest_group.add_argument("--dest", help="Agent Skills root, e.g. .agents/skills")
    dest_group.add_argument(
        "--client",
        choices=list(CLIENT_PRESETS),
        help="Use a documented client preset root and mode",
    )
    parser.add_argument("--mode", choices=["copy", "symlink"], help="Override the install mode")
    parser.add_argument("--name", default=SOURCE.name)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def destination_and_mode(args: argparse.Namespace) -> tuple[str, str]:
    if args.client:
        preset_dest, preset_mode = CLIENT_PRESETS[args.client]
        return preset_dest, args.mode or preset_mode
    return args.dest, args.mode or "copy"


def resolve_install_path(path: Path, label: str) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"{label} contains a symlink cycle: {path}") from exc


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def reject_symlink_cycle(path: Path) -> None:
    if not path.is_symlink():
        return
    try:
        os.stat(path)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SkillError(f"Target contains a symlink cycle: {path}") from exc
        if exc.errno != errno.ENOENT:
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


def _require_nofollow_support() -> None:
    if not NOFOLLOW_FLAG or not DIRECTORY_FLAG:
        raise SkillError(
            "this platform lacks no-follow file APIs; refusing a manifest-attested copy install"
        )


def _validate_symlink_target(link_path: str, target: object, label: str) -> str:
    if not isinstance(target, str) or not target:
        raise SkillError(f"{label}.target must be a non-empty string")
    if "\x00" in target:
        raise SkillError(f"{label}.target must not contain NUL")
    portable = target.replace("\\", "/")
    if (
        target.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", portable)
        or os.path.splitdrive(target)[0]
    ):
        raise SkillError(f"{label} target must be relative and stay inside the skill root")
    destination = list(PurePosixPath(link_path).parent.parts)
    for part in portable.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not destination:
                raise SkillError(f"{label} target escapes the skill root")
            destination.pop()
        else:
            destination.append(part)
    return target


def load_distribution_manifest(
    manifest_path: Path | None = None,
    source: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Return the strict manifest allowlist for one distributable skill."""
    source = SOURCE if source is None else source
    manifest_path = source.parent / MANIFEST_NAME if manifest_path is None else manifest_path
    payload = manifest_contract.read_manifest(manifest_path)
    files = manifest_contract.validate_manifest(payload, source.parent)

    prefix = f"{source.name}/"
    entries: dict[str, dict[str, object]] = {}
    for index, raw_record in enumerate(files):
        label = f"distribution manifest files[{index}]"
        record = dict(raw_record)
        full_path = str(record["path"])
        if not full_path.startswith(prefix):
            continue
        relative = full_path[len(prefix):]
        if "\x00" in relative:
            raise SkillError(f"{label}.path must not contain NUL")
        if record.get("type") == "symlink":
            target = _validate_symlink_target(relative, record.get("target"), label)
            try:
                encoded = target.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise SkillError(f"{label}.target must be valid UTF-8") from exc
            if (
                record["size_bytes"] != len(encoded)
                or record["sha256"] != hashlib.sha256(encoded).hexdigest()
            ):
                raise SkillError(f"{label} symlink target metadata does not match its target text")
        if relative in entries:
            raise SkillError(f"distribution manifest has duplicate path: {full_path}")
        entries[relative] = record

    if not entries:
        raise SkillError(f"distribution manifest has no {source.name}/ payload")
    for required in ("SKILL.md", "LICENSE"):
        if required not in entries or entries[required].get("type") == "symlink":
            raise SkillError(f"distribution manifest is missing regular {source.name}/{required}")

    entry_paths = {PurePosixPath(path) for path in entries}
    for relative in entries:
        path = PurePosixPath(relative)
        if any(parent in entry_paths for parent in path.parents):
            raise SkillError(
                f"distribution manifest places {relative} below another payload entry"
            )
    return dict(sorted(entries.items()))


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


def _expected_inventory(entries: dict[str, dict[str, object]]) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for relative, record in entries.items():
        path = PurePosixPath(relative)
        for parent in path.parents:
            if str(parent) != ".":
                inventory[str(parent)] = "directory"
        inventory[relative] = "symlink" if record.get("type") == "symlink" else "file"
    return inventory


def _open_parent_descriptors(root: Path, relative: PurePosixPath) -> tuple[list[int], int, str]:
    _require_nofollow_support()
    directory_flags = os.O_RDONLY | NOFOLLOW_FLAG | DIRECTORY_FLAG | getattr(os, "O_CLOEXEC", 0)
    try:
        current = os.open(root, directory_flags)
    except OSError as exc:
        raise SkillError(
            f"could not open source root without following symlinks: {root}: {exc}"
        ) from exc
    descriptors = [current]
    try:
        for part in relative.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        return descriptors, current, relative.name
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise SkillError(
            f"source path contains a missing or symlinked directory: {relative}"
        ) from exc


def _close_descriptors(descriptors: list[int]) -> None:
    for descriptor in reversed(descriptors):
        os.close(descriptor)


def _read_verified_regular(
    root: Path,
    relative: str,
    record: dict[str, object],
    destination: BinaryIO | None = None,
) -> None:
    path = PurePosixPath(relative)
    descriptors, parent, name = _open_parent_descriptors(root, path)
    flags = os.O_RDONLY | NOFOLLOW_FLAG | getattr(os, "O_CLOEXEC", 0)
    try:
        try:
            descriptor = os.open(name, flags, dir_fd=parent)
        except OSError as exc:
            raise SkillError(
                f"could not open manifest file without following symlinks: {relative}"
            ) from exc
        try:
            initial = os.fstat(descriptor)
            if not stat.S_ISREG(initial.st_mode):
                raise SkillError(f"manifest file is not a regular file: {relative}")
            if bool(initial.st_mode & 0o111) != record["executable"]:
                raise SkillError(f"manifest executable flag does not match source: {relative}")
            digest = hashlib.sha256()
            remaining = initial.st_size
            while remaining:
                chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
                if not chunk:
                    raise SkillError(f"source file changed while it was being read: {relative}")
                digest.update(chunk)
                if destination is not None:
                    destination.write(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise SkillError(f"source file changed while it was being read: {relative}")
            final = os.fstat(descriptor)
            try:
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise SkillError(
                    f"source file changed while it was being read: {relative}"
                ) from exc
            if (
                not stat.S_ISREG(current.st_mode)
                or not _same_snapshot(initial, final)
                or not _same_snapshot(initial, current)
            ):
                raise SkillError(f"source file changed while it was being read: {relative}")
            if initial.st_size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                raise SkillError(f"source file does not match distribution manifest: {relative}")
        finally:
            os.close(descriptor)
    finally:
        _close_descriptors(descriptors)


def _read_verified_symlink(
    root: Path,
    relative: str,
    record: dict[str, object],
) -> str:
    path = PurePosixPath(relative)
    descriptors, parent, name = _open_parent_descriptors(root, path)
    try:
        initial = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISLNK(initial.st_mode):
            raise SkillError(f"manifest symlink is not a symlink: {relative}")
        target = os.readlink(name, dir_fd=parent)
        final = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISLNK(final.st_mode) or not _same_snapshot(initial, final):
            raise SkillError(f"source symlink changed while it was being read: {relative}")
        if target != record["target"]:
            raise SkillError(f"source symlink does not match distribution manifest: {relative}")
        _validate_resolved_symlink(root, relative, target, "source symlink")
        return target
    finally:
        _close_descriptors(descriptors)


def _validate_resolved_symlink(root: Path, relative: str, target: str, label: str) -> None:
    _validate_symlink_target(relative, target, label)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = (root / relative).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"{label} contains a cycle or invalid target: {relative}") from exc
    if not inside(resolved, resolved_root):
        raise SkillError(f"{label} resolves outside the skill root: {relative} -> {target}")


def verify_source_tree(source: Path, entries: dict[str, dict[str, object]]) -> None:
    """Reject source drift and verify every manifest payload hash without following links."""
    expected_records = []
    for relative, record in entries.items():
        normalized = dict(record)
        normalized["path"] = relative
        expected_records.append(normalized)
    current_records = manifest_contract.build_files(source)
    delta = manifest_contract.diff(expected_records, current_records)
    if any(delta.values()):
        details = []
        if delta["added"]:
            details.append(f"unlisted source content: {', '.join(delta['added'])}")
        if delta["removed"]:
            details.append(f"missing manifest content: {', '.join(delta['removed'])}")
        if delta["changed"]:
            details.append(f"changed manifest content: {', '.join(delta['changed'])}")
        raise SkillError(
            "source tree does not match the distribution manifest; " + "; ".join(details)
        )

    for relative, record in entries.items():
        if record.get("type") == "symlink":
            _read_verified_symlink(source, relative, record)


def tree_signature(root: Path) -> dict[str, tuple[str, str, int]]:
    """Describe an exact installed tree, rejecting unsupported filesystem nodes."""
    root_metadata = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise SkillError(f"installed tree root must be a directory: {root}")
    signature: dict[str, tuple[str, str, int]] = {
        "./": ("directory", "", stat.S_IMODE(root_metadata.st_mode)),
    }
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            relative = path.relative_to(root)
            key = relative.as_posix()
            metadata = entry.stat(follow_symlinks=False)
            mode = metadata.st_mode
            permissions = stat.S_IMODE(mode)
            if stat.S_ISLNK(mode):
                # Symlink permission bits are platform-defined and cannot be
                # portably set. Link identity is its exact target text.
                signature[key] = ("symlink", os.readlink(path), 0)
            elif stat.S_ISDIR(mode):
                signature[f"{key}/"] = ("directory", "", permissions)
                pending.append(path)
            elif stat.S_ISREG(mode):
                digest = hashlib.sha256()
                flags = os.O_RDONLY | NOFOLLOW_FLAG | getattr(os, "O_CLOEXEC", 0)
                try:
                    descriptor = os.open(path, flags)
                except OSError as exc:
                    raise SkillError(
                        f"could not hash installed file without following links: {relative}"
                    ) from exc
                try:
                    while chunk := os.read(descriptor, READ_CHUNK_BYTES):
                        digest.update(chunk)
                finally:
                    os.close(descriptor)
                signature[key] = ("file", f"{metadata.st_size}:{digest.hexdigest()}", permissions)
            else:
                raise SkillError(
                    f"unsupported {_special_node_kind(mode)} in installed tree: {relative}"
                )
    return signature


def expected_tree_signature(
    entries: dict[str, dict[str, object]],
) -> dict[str, tuple[str, str, int]]:
    signature: dict[str, tuple[str, str, int]] = {"./": ("directory", "", 0o755)}
    for relative, kind in _expected_inventory(entries).items():
        record = entries.get(relative)
        if kind == "directory":
            signature[f"{relative}/"] = ("directory", "", 0o755)
        elif kind == "symlink":
            signature[relative] = ("symlink", str(record["target"]), 0)
        else:
            permissions = 0o755 if record["executable"] else 0o644
            signature[relative] = (
                "file",
                f"{record['size_bytes']}:{record['sha256']}",
                permissions,
            )
    return signature


def already_installed(
    target: Path,
    mode: str,
    entries: dict[str, dict[str, object]] | None = None,
) -> bool:
    if mode == "symlink":
        return target.is_symlink() and target.resolve() == SOURCE.resolve()
    if not target.is_dir() or target.is_symlink():
        return False
    entries = load_distribution_manifest() if entries is None else entries
    return tree_signature(target) == expected_tree_signature(entries)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | DIRECTORY_FLAG | NOFOLLOW_FLAG)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _node_identity(path: Path) -> dict[str, int]:
    metadata = os.lstat(path)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "type": stat.S_IFMT(metadata.st_mode),
    }


def _valid_identity(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"device", "inode", "type"}
        and all(type(value[key]) is int and value[key] >= 0 for key in value)
    )


def _identity_matches(path: Path, expected: object) -> bool:
    if not _valid_identity(expected):
        return False
    try:
        return _node_identity(path) == expected
    except FileNotFoundError:
        return False


def _install_journal_path(target: Path) -> Path:
    return target.parent / f".{target.name}.install-journal.json"


def _create_force_backup_placeholder(target: Path, nonce: str) -> Path:
    prefix = f".{target.name}.backup-{nonce}-"
    if target.is_dir() and not target.is_symlink():
        return Path(tempfile.mkdtemp(prefix=prefix, dir=target.parent))
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=target.parent)
    os.close(descriptor)
    return Path(raw_path)


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write force-install recovery journal")
        remaining = remaining[written:]


def _write_recovery_journal(journal: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            journal,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW_FLAG,
            0o600,
        )
        created = True
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _fsync_directory(journal.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                journal.unlink()
                _fsync_directory(journal.parent)
            except OSError:
                pass
        raise


def _remove_recovery_journal(journal: Path) -> None:
    journal.unlink()
    _fsync_directory(journal.parent)


def _read_recovery_journal(journal: Path) -> object | None:
    if not _path_present(journal):
        return None
    descriptor = -1
    try:
        descriptor = os.open(journal, os.O_RDONLY | NOFOLLOW_FLAG)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_INSTALL_JOURNAL_BYTES:
            raise ValueError("journal is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = MAX_INSTALL_JOURNAL_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            raise ValueError("journal exceeds the size limit")
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        print(
            f"warning: unrecognized force-install recovery journal left untouched: {journal}",
            file=sys.stderr,
        )
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validated_recovery_paths(
    target: Path,
    payload: object,
) -> tuple[Path, Path] | None:
    if not isinstance(payload, dict):
        return None
    nonce = payload.get("nonce")
    if (
        payload.get("schema_version") != INSTALL_JOURNAL_SCHEMA_VERSION
        or payload.get("operation") != "force-replace"
        or not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
        or payload.get("target") != str(target)
        or not _valid_identity(payload.get("original_identity"))
        or not _valid_identity(payload.get("stage_identity"))
        or not _valid_identity(payload.get("placeholder_identity"))
    ):
        return None
    raw_backup = payload.get("backup")
    raw_stage = payload.get("stage")
    if not isinstance(raw_backup, str) or not isinstance(raw_stage, str):
        return None
    backup = Path(raw_backup)
    stage = Path(raw_stage)
    if (
        backup.parent != target.parent
        or stage.parent != target.parent
        or not backup.name.startswith(f".{target.name}.backup-{nonce}-")
        or not stage.name.startswith(f".{SOURCE.name}.install-")
    ):
        return None
    return backup, stage


def _warn_legacy_force_backups(target: Path) -> None:
    if not target.parent.is_dir():
        return
    deterministic_name = f".{target.name}.backup"
    prefix = f".{target.name}.backup-"
    backups = sorted(
        Path(entry.path)
        for entry in os.scandir(target.parent)
        if entry.name == deterministic_name or entry.name.startswith(prefix)
    )
    for backup in backups:
        print(
            f"warning: legacy stranded force-install backup left untouched: {backup}",
            file=sys.stderr,
        )


def _warn_recovery_journal(target: Path) -> None:
    journal = _install_journal_path(target)
    if _path_present(journal):
        print(
            f"warning: force-install recovery journal left untouched: {journal}",
            file=sys.stderr,
        )


def recover_interrupted_install(target: Path) -> None:
    """Recover only states proven by this target's ownership journal."""
    journal = _install_journal_path(target)
    payload = _read_recovery_journal(journal)
    paths = _validated_recovery_paths(target, payload)
    if payload is not None and paths is None:
        print(
            f"warning: unrecognized force-install recovery journal left untouched: {journal}",
            file=sys.stderr,
        )
    elif paths is not None:
        backup, stage = paths
        original_identity = payload["original_identity"]
        stage_identity = payload["stage_identity"]
        placeholder_identity = payload["placeholder_identity"]
        target_is_original = _identity_matches(target, original_identity)
        target_is_stage = _identity_matches(target, stage_identity)
        backup_is_original = _identity_matches(backup, original_identity)
        backup_is_placeholder = _identity_matches(backup, placeholder_identity)
        stage_is_owned = _identity_matches(stage, stage_identity)

        action = None
        if not _path_present(target) and backup_is_original:
            os.replace(backup, target)
            _fsync_directory(target.parent)
            if stage_is_owned:
                remove_path(stage)
                _fsync_directory(target.parent)
            action = "restored"
        elif target_is_stage and backup_is_original:
            remove_path(backup)
            _fsync_directory(target.parent)
            action = "completed cleanup for"
        elif (
            target_is_original
            and backup_is_placeholder
            and (stage_is_owned or not _path_present(stage))
        ):
            if stage_is_owned:
                remove_path(stage)
            remove_path(backup)
            _fsync_directory(target.parent)
            action = "discarded prepared"
        elif target_is_original and not _path_present(backup):
            if stage_is_owned:
                remove_path(stage)
                _fsync_directory(target.parent)
            action = "rolled back"
        elif target_is_stage and not _path_present(backup):
            action = "completed cleanup for"

        if action is None:
            print(
                f"warning: force-install recovery journal state mismatch left untouched: {journal}",
                file=sys.stderr,
            )
        else:
            _remove_recovery_journal(journal)
            print(
                f"warning: {action} interrupted force install: {target}",
                file=sys.stderr,
            )
    _warn_legacy_force_backups(target)


def _stage_copy(
    dest_root: Path,
    entries: dict[str, dict[str, object]],
) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=f".{SOURCE.name}.install-", dir=dest_root))
    try:
        stage.chmod(0o755)
        directories = [
            Path(path)
            for path, kind in _expected_inventory(entries).items()
            if kind == "directory"
        ]
        for directory in sorted(directories, key=lambda path: (len(path.parts), path.as_posix())):
            destination = stage / directory
            destination.mkdir()
            destination.chmod(0o755)
        for relative, record in entries.items():
            if record.get("type") == "symlink":
                continue
            destination = stage / relative
            with destination.open("xb") as handle:
                _read_verified_regular(SOURCE, relative, record, handle)
            destination.chmod(0o755 if record["executable"] else 0o644)
        for relative, record in entries.items():
            if record.get("type") != "symlink":
                continue
            target = _read_verified_symlink(SOURCE, relative, record)
            (stage / relative).symlink_to(target)
        for relative, record in entries.items():
            if record.get("type") == "symlink":
                _validate_resolved_symlink(stage, relative, str(record["target"]), "staged symlink")
        # Recheck the full source inventory after staging so source drift that
        # happened during the copy cannot be silently accepted.
        verify_source_tree(SOURCE, entries)
        if tree_signature(stage) != expected_tree_signature(entries):
            raise SkillError("staged install does not exactly match the manifest allowlist")
        return stage
    except BaseException:
        remove_path(stage)
        raise


def stage_install(
    dest_root: Path,
    mode: str,
    entries: dict[str, dict[str, object]] | None = None,
) -> Path:
    if mode == "copy":
        entries = load_distribution_manifest() if entries is None else entries
        return _stage_copy(dest_root, entries)
    stage = Path(tempfile.mkdtemp(prefix=f".{SOURCE.name}.install-", dir=dest_root))
    remove_path(stage)
    try:
        relative = os.path.relpath(SOURCE, dest_root)
        stage.symlink_to(relative, target_is_directory=True)
        return stage
    except BaseException:
        remove_path(stage)
        raise


def install_atomically(target: Path, stage: Path, force: bool) -> None:
    if not _path_present(target):
        os.replace(stage, target)
        return
    if not force:
        raise SkillError(f"Target already exists: {target}; use --force to replace")
    nonce = secrets.token_hex(16)
    backup = _create_force_backup_placeholder(target, nonce)
    journal = _install_journal_path(target)
    payload: dict[str, object] = {
        "schema_version": INSTALL_JOURNAL_SCHEMA_VERSION,
        "operation": "force-replace",
        "nonce": nonce,
        "target": str(target),
        "backup": str(backup),
        "stage": str(stage),
        "original_identity": _node_identity(target),
        "stage_identity": _node_identity(stage),
        "placeholder_identity": _node_identity(backup),
    }
    try:
        _write_recovery_journal(journal, payload)
    except BaseException:
        if _identity_matches(backup, payload["placeholder_identity"]):
            remove_path(backup)
            _fsync_directory(target.parent)
        raise
    try:
        os.replace(target, backup)
    except BaseException:
        if _identity_matches(backup, payload["placeholder_identity"]):
            remove_path(backup)
            _fsync_directory(target.parent)
        _remove_recovery_journal(journal)
        raise
    _fsync_directory(target.parent)
    try:
        os.replace(stage, target)
    except BaseException:
        os.replace(backup, target)
        _fsync_directory(target.parent)
        _remove_recovery_journal(journal)
        raise
    _fsync_directory(target.parent)
    remove_path(backup)
    _fsync_directory(target.parent)
    _remove_recovery_journal(journal)


def main() -> int:
    args = parse_args()
    try:
        if not args.name or "/" in args.name or "\\" in args.name or args.name in {".", ".."}:
            raise SkillError("--name must be a single safe directory name")
        dest_arg, mode = destination_and_mode(args)
        source_root = resolve_install_path(SOURCE, "Source")
        dest_root = resolve_install_path(Path(dest_arg), "Destination")
        target = dest_root / args.name
        if inside(dest_root, source_root) or inside(target, source_root):
            raise SkillError("Destination and target must not be inside the source skill directory")
        if inside(source_root, target):
            raise SkillError("Target must not contain the source skill directory")
        try:
            target.relative_to(dest_root)
        except ValueError as exc:
            raise SkillError("Target escapes destination root") from exc
        reject_symlink_cycle(target)

        if not args.dry_run:
            dest_root.mkdir(parents=True, exist_ok=True)
            recover_interrupted_install(target)
            reject_symlink_cycle(target)

        manifest_entries = None
        if mode == "copy":
            manifest_entries = load_distribution_manifest()
            verify_source_tree(SOURCE, manifest_entries)

        print(f"source: {SOURCE}")
        print(f"destination: {dest_root}")
        print(f"target: {target}")
        print(f"mode: {mode}")
        if args.dry_run:
            _warn_recovery_journal(target)
            _warn_legacy_force_backups(target)
            return 0
        if already_installed(target, mode, manifest_entries):
            print("already installed")
            return 0
        stage = stage_install(dest_root, mode, manifest_entries)
        try:
            install_atomically(target, stage, args.force)
        finally:
            remove_path(stage)
        print("installed")
        return 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
