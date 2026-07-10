#!/usr/bin/env python3
"""Shared helpers for the MLX model porting skill scripts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import statistics
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, BinaryIO


class SkillError(RuntimeError):
    """Raised for expected, actionable skill-tool errors."""


DEFAULT_IGNORED_TREES = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".wrangler",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
})
MAX_CAPTURE_BYTES = 1024 * 1024
CAPTURE_CHUNK_BYTES = 64 * 1024
MAX_STRUCTURED_BYTES = 16 * 1024 * 1024


def bounded_files(
    root: Path,
    max_files: int,
    *,
    ignored_trees: set[str] | frozenset[str] = DEFAULT_IGNORED_TREES,
) -> tuple[list[Path], bool]:
    """Enumerate a bounded tree without following links or entering noisy trees."""
    if max_files <= 0:
        raise SkillError("file traversal limit must be positive")
    if root.is_file():
        return [root], False
    root_resolved = root.resolve()
    pending: list[tuple[str, Path]] = [("directory", root)]
    files: list[Path] = []
    visited_entries = 0
    max_entries = max(1_024, max_files * 8)
    entry_limit_hit = False
    truncated = False
    while pending:
        kind, path = pending.pop()
        if kind == "file":
            files.append(path)
            if len(files) > max_files:
                return files[:max_files], True
            continue
        if entry_limit_hit:
            truncated = True
            continue
        directory = path
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise SkillError(f"could not enumerate {directory}: {exc}") from exc
        children: list[tuple[str, Path]] = []
        bounded_entries: list[os.DirEntry[str]] = []
        with entries:
            for entry in entries:
                visited_entries += 1
                if visited_entries > max_entries:
                    entry_limit_hit = True
                    truncated = True
                    break
                bounded_entries.append(entry)
            for entry in sorted(bounded_entries, key=lambda item: item.name):
                path = Path(entry.path)
                if entry.is_symlink():
                    try:
                        resolved = path.resolve(strict=True)
                        resolved.relative_to(root_resolved)
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise SkillError(f"symlink escapes inspected root: {path}") from exc
                    if resolved.is_file():
                        children.append(("file", path))
                    # Never follow directory symlinks, even when their target stays in-root.
                elif entry.is_dir(follow_symlinks=False):
                    if entry.name not in ignored_trees:
                        children.append(("directory", path))
                elif entry.is_file(follow_symlinks=False):
                    children.append(("file", path))
        pending.extend(reversed(children))
    return files, truncated


def load_structured(path: str | Path) -> Any:
    """Load JSON or YAML. Registry .yaml files are JSON-compatible by design."""
    p = Path(path)
    try:
        if p.stat().st_size > MAX_STRUCTURED_BYTES:
            raise SkillError(f"Structured input exceeds {MAX_STRUCTURED_BYTES} bytes: {p}")
        with p.open("rb") as handle:
            raw = handle.read(MAX_STRUCTURED_BYTES + 1)
        if len(raw) > MAX_STRUCTURED_BYTES:
            raise SkillError(f"Structured input exceeds {MAX_STRUCTURED_BYTES} bytes: {p}")
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillError(f"Could not read {p}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SkillError(
                f"{p} is not JSON-compatible YAML and PyYAML is not installed. "
                "Install PyYAML or convert the file to JSON-compatible YAML."
            ) from exc
        try:
            return yaml.safe_load(text)
        except Exception as yaml_exc:  # pragma: no cover - dependency-specific
            raise SkillError(f"Could not parse {p}: {yaml_exc}; JSON error: {json_exc}") from yaml_exc


_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(?:authorization|proxy[_-]?authorization|api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"token|secret|password|private[_-]?key|database[_-]?url|connection[_-]?string|cookie)(?:$|[_-])",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+\-/=]+")
_AUTHORIZATION_RE = re.compile(
    r"(?i)((?:^|\\[nrt]|[^A-Za-z0-9_])(?:proxy[-_])?authorization\s*[:=]\s*"
    r"(?:basic|bearer)\s+)(?!\[REDACTED\])(?:(?!\\[nrt])[^\s,;])+"
)
_SECRET_HEADER_RE = re.compile(
    r"(?i)((?:^|\\[nrt]|[^A-Za-z0-9_])(?:(?:x[-_])?(?:api[-_]?key|auth[-_]?token|access[-_]?token)|"
    r"(?:set[-_])?cookie)\s*[:=]\s*)(?!\[REDACTED\])(?:(?!\\[nrt])[^\r\n,])+"
)
_SECRET_FLAG_RE = re.compile(
    r'''(?i)(--(?:api[-_]?key|access[-_]?token|auth[-_]?token|token|secret|password|'''
    r'''private[-_]?key|database[-_]?url|connection[-_]?string|secret[-_]?access[-_]?key)\s*[= ]\s*)'''
    r'''(?:"[^"]*"|'[^']*'|(?:(?!\\[nrt])[^\s])+)'''
)
_SECRET_ENV_RE = re.compile(
    r'''(?i)((?:^|\\[nrt]|[^A-Za-z0-9_])(?:[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|'''
    r'''DATABASE_URL|CONNECTION_STRING|SECRET_ACCESS_KEY))\s*=\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|'''
    r'''(?:(?!\\[nrt])[^\s])+)'''
)
_SECRET_OPTION_RE = re.compile(
    r"(?i)^--(?:api[-_]?key|access[-_]?token|auth[-_]?token|token|secret|password|"
    r"private[-_]?key|database[-_]?url|connection[-_]?string|secret[-_]?access[-_]?key)$"
)
_URL_USERINFO_RE = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.-]*://)(?!\[REDACTED\]@)[^\s/:@]+:[^\s/@]+@"
)
_URL_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password)=)"
    r"(?!\[REDACTED\])(?:(?!\\[nrt])[^&#\s])+"
)


def redact_secret_text(value: str) -> str:
    """Redact common credential spellings before text reaches a receipt or log."""
    value = _AUTHORIZATION_RE.sub(r"\1[REDACTED]", value)
    value = _BEARER_RE.sub(r"\1[REDACTED]", value)
    value = _SECRET_HEADER_RE.sub(r"\1[REDACTED]", value)
    value = _SECRET_FLAG_RE.sub(r"\1[REDACTED]", value)
    value = _SECRET_ENV_RE.sub(r"\1[REDACTED]", value)
    value = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", value)
    return _URL_QUERY_SECRET_RE.sub(r"\1[REDACTED]", value)


def redact_secrets(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-compatible copy with credential values removed."""
    if key is not None and _SENSITIVE_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): redact_secrets(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        redacted: list[Any] = []
        redact_next = False
        for item in value:
            if redact_next:
                redacted.append("[REDACTED]")
                redact_next = False
                continue
            redacted.append(redact_secrets(item))
            redact_next = isinstance(item, str) and _SECRET_OPTION_RE.fullmatch(item) is not None
        return redacted
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return value


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file from a same-directory staging file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        try:
            existing = target.lstat()
            mode = stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else 0o644
        except FileNotFoundError:
            mode = 0o644
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def dump_json(data: Any, path: str | Path | None = None) -> str:
    text = json.dumps(redact_secrets(data), indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if path is not None:
        atomic_write_text(path, text)
    return text


def validate_comparison_tolerances(atol: float, rtol: float, cosine_min: float) -> None:
    """Apply the shared fail-closed safety policy for tensor comparison thresholds."""
    for name, value in (("--atol", atol), ("--rtol", rtol)):
        if not math.isfinite(value) or value < 0:
            raise SkillError(f"{name} must be a finite non-negative number")
    if not math.isfinite(cosine_min) or not -1.0 <= cosine_min <= 1.0:
        raise SkillError("--cosine-min must be finite and between -1 and 1")


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate a subprocess and its descendants without invoking a shell."""
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    if process.poll() is not None:
        return
    process.kill()


class _BoundedCapture:
    """Keep the beginning and end of one process stream in bounded memory."""

    def __init__(self, limit: int = MAX_CAPTURE_BYTES):
        self.limit = limit
        self.head_limit = limit // 2
        self.tail_limit = limit - self.head_limit
        self.total = 0
        self.head = bytearray()
        self.tail = bytearray()

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        head_remaining = self.head_limit - len(self.head)
        if head_remaining > 0:
            self.head.extend(chunk[:head_remaining])
            chunk = chunk[head_remaining:]
        if chunk:
            self.tail.extend(chunk)
            if len(self.tail) > self.tail_limit:
                del self.tail[:-self.tail_limit]

    def text(self) -> str:
        if self.total <= self.limit:
            payload = bytes(self.head + self.tail)
            return payload.decode("utf-8", errors="replace")
        head_size = self.head_limit
        tail_size = self.tail_limit
        for _ in range(3):
            omitted = self.total - head_size - tail_size
            marker = (
                f"\n...[truncated {omitted} bytes; kept first {head_size} and last {tail_size} bytes]...\n"
            ).encode("utf-8")
            available = max(0, self.limit - len(marker))
            head_size = available // 2
            tail_size = available - head_size
        omitted = self.total - head_size - tail_size
        marker = (
            f"\n...[truncated {omitted} bytes; kept first {head_size} and last {tail_size} bytes]...\n"
        ).encode("utf-8")
        payload = bytes(self.head[:head_size]) + marker + bytes(self.tail[-tail_size:] if tail_size else b"")
        return payload.decode("utf-8", errors="replace")


def _drain_process_stream(stream: BinaryIO, capture: _BoundedCapture) -> None:
    try:
        while chunk := stream.read(CAPTURE_CHUNK_BYTES):
            capture.add(chunk)
    except (OSError, ValueError):
        pass
    finally:
        stream.close()


def _finish_capture_threads(
    streams: tuple[BinaryIO, BinaryIO],
    threads: tuple[threading.Thread, ...],
    *,
    close_first: bool,
) -> None:
    if close_first:
        for stream in streams:
            try:
                stream.close()
            except OSError:
                pass
    for thread in threads:
        thread.join(timeout=5)
    for thread in threads:
        if thread.is_alive():
            for stream in streams:
                try:
                    stream.close()
                except OSError:
                    pass
            thread.join(timeout=1)


def run_process_capture(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run a bounded command and kill its process group if the timeout expires."""
    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    stdout_capture = _BoundedCapture()
    stderr_capture = _BoundedCapture()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    streams: tuple[BinaryIO, BinaryIO] | None = None
    timed_out = False
    completed_setup_and_wait = False
    started_threads: list[threading.Thread] = []
    try:
        if process.stdout is None or process.stderr is None:
            raise SkillError("bounded process capture requires stdout and stderr pipes")
        streams = (process.stdout, process.stderr)
        for stream, capture, name in (
            (process.stdout, stdout_capture, "mlx-capture-stdout"),
            (process.stderr, stderr_capture, "mlx-capture-stderr"),
        ):
            thread = threading.Thread(
                target=_drain_process_stream,
                args=(stream, capture),
                daemon=True,
                name=name,
            )
            thread.start()
            started_threads.append(thread)
        try:
            process.wait(timeout=timeout)
            # A command can exit after launching children that retain its process
            # group or capture pipes. Never let those descendants outlive the
            # bounded invocation merely because the group leader exited first.
            terminate_process_tree(process)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(process)
            process.wait()
        completed_setup_and_wait = True
    finally:
        if not completed_setup_and_wait:
            terminate_process_tree(process)
            try:
                process.wait(timeout=1)
            except BaseException:
                pass
        if streams is not None:
            _finish_capture_threads(
                streams,
                tuple(started_threads),
                close_first=not completed_setup_and_wait,
            )
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout_capture.text(),
        stderr_capture.text(),
    ), timed_out


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse the deliberately simple YAML frontmatter used by SKILL.md."""
    if not text.startswith("---\n"):
        raise SkillError("SKILL.md must start with YAML frontmatter delimited by ---")
    try:
        raw, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise SkillError("SKILL.md frontmatter is missing the closing --- delimiter") from exc

    # Prefer PyYAML when present. Fall back to a conservative parser for scalar
    # fields and one-level metadata maps used by this repository.
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise SkillError("SKILL.md frontmatter must be a mapping")
        return data, body
    except ImportError:
        result: dict[str, Any] = {}
        current_map: dict[str, str] | None = None
        current_key: str | None = None
        for line in raw.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("  ") and current_map is not None and ":" in line:
                key, value = line.strip().split(":", 1)
                current_map[key.strip()] = _strip_yaml_scalar(value.strip())
                continue
            if ":" not in line:
                raise SkillError(f"Unsupported frontmatter line without PyYAML: {line!r}")
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if not value:
                current_map = {}
                result[key] = current_map
                current_key = key
            else:
                result[key] = _strip_yaml_scalar(value)
                current_map = None
                current_key = None
        return result, body


def _strip_yaml_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def safe_relpath(root: Path, candidate: Path) -> str:
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise SkillError(f"Path escapes allowed root {root}: {candidate}") from exc


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "item"


def applies_to_family(applies_to: list, family: str) -> bool:
    """Whether an ``applies_to`` list matches an architecture family id.

    Single source of truth for the fuzzy family matcher shared by
    ``recommend_optimizations.py``, ``make_port_plan.py``, and the reachability
    guard test, so production and test semantics cannot silently diverge.
    """
    f = family.lower()
    tokens = set(f.replace("-", " ").split())
    for value in [str(x).lower() for x in (applies_to or [])]:
        if value == "all" or value == f or value in f or f in value:
            return True
        if "-" not in value and value in tokens:
            return True
    return False


_BAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)x\s*-\s*(\d+(?:\.\d+)?)x\s*$")


def parse_band(range_str: str) -> tuple[float, float]:
    """Parse a multiplier band like ``1.0x-4.3x`` into numeric bounds."""
    if not isinstance(range_str, str):
        raise ValueError(f"improvement band must be a string, got {type(range_str).__name__}")
    match = _BAND_RE.fullmatch(range_str)
    if not match:
        raise ValueError(f"malformed improvement band {range_str!r}; expected '<floor>x-<ceiling>x'")
    floor, ceiling = (float(match.group(1)), float(match.group(2)))
    if not (math.isfinite(floor) and math.isfinite(ceiling)):
        raise ValueError(f"improvement band {range_str!r} contains a non-finite value")
    if floor <= 0 or ceiling <= 0:
        raise ValueError(f"improvement band {range_str!r} must contain positive multipliers")
    if floor > ceiling:
        raise ValueError(f"improvement band {range_str!r} has floor above ceiling")
    return floor, ceiling


def _format_multiplier(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"{text}x"


STACK_COMPOUND_HYPOTHESIS_FLAG = "unmeasured composition - multiplicative hypothesis, not a claim"


COMPOUND_ASSESSMENT_CLASSIFICATIONS = {
    "performance_observation",
    "promotion_ready",
    "rejected",
}
COMPOUND_PROMOTION_REQUIRED_GATES = (
    "aggregates_recomputed",
    "model_lineage_pinned",
    "target_hash_valid",
    "workload_hash_valid",
    "raw_outputs_valid",
    "quality_valid",
    "stability_passed",
    "stability_threshold_valid",
    "rollback_defined",
    "baseline_compatible",
    "enabled_methods_valid",
    "improvement_beyond_noise",
    "runner_valid",
    "quality_binding_valid",
    "experiment_descriptor_valid",
    "experiment_compatible",
    "source_identity_pinned",
    "baseline_source_identity_match",
    "execution_attested",
)
STACK_MEASURED_METRIC_POLICY = {
    "decode-tokens-per-sec": ("generation_tps", "decode_tps"),
    "wall-time": ("wall_seconds", "wall_seconds_inverse"),
    "kernel-latency": ("wall_seconds", "wall_seconds_inverse"),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
MLX_FINGERPRINT_METRICS = (
    "prompt_tokens",
    "prompt_tps",
    "generation_tokens",
    "generation_tps",
    "peak_memory_gb",
    "ttft_proxy_s",
)
EXTERNAL_FINGERPRINT_METRICS = ("wall_seconds",)
FINGERPRINT_METRIC_SETS = {
    frozenset(MLX_FINGERPRINT_METRICS),
    frozenset(EXTERNAL_FINGERPRINT_METRICS),
}
EXPERIMENT_IDENTITY_FIELDS = (
    "models",
    "target",
    "workload",
    "primary_metric",
    "candidate_baseline_binding",
    "enabled_methods",
)


def _canonical_digest(item: Any) -> str:
    return hashlib.sha256(
        json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _quality_contract_identity_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"validator", "metric", "provenance_mode"}:
        return False
    metric = value.get("metric")
    return (
        isinstance(value.get("validator"), dict)
        and set(value["validator"]) == {"id", "version"}
        and value["validator"].get("id") == "mlx-benchmark-exact-output-parity"
        and not isinstance(value["validator"].get("version"), bool)
        and isinstance(value["validator"].get("version"), int)
        and value["validator"].get("version") == 1
        and value.get("provenance_mode") == "controlled-exact-output-v1"
        and isinstance(metric, dict)
        and set(metric)
        == {
            "id",
            "reference_sha256",
            "candidate_sha256",
            "reference_size_bytes",
            "candidate_size_bytes",
            "exact_match",
        }
        and metric.get("id") == "exact-output-parity"
        and metric.get("exact_match") is True
        and isinstance(metric.get("reference_sha256"), str)
        and SHA256_RE.fullmatch(metric["reference_sha256"]) is not None
        and isinstance(metric.get("candidate_sha256"), str)
        and SHA256_RE.fullmatch(metric["candidate_sha256"]) is not None
        and metric["candidate_sha256"] == metric["reference_sha256"]
        and all(
            not isinstance(metric.get(field), bool)
            and isinstance(metric.get(field), int)
            and metric[field] >= 0
            for field in ("reference_size_bytes", "candidate_size_bytes")
        )
        and metric["candidate_size_bytes"] == metric["reference_size_bytes"]
    )


def _experiment_fingerprint_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"schema_version", "sha256", "payload"}:
        return False
    payload = value.get("payload")
    if value.get("schema_version") != 2 or not isinstance(payload, dict):
        return False
    if set(payload) != {
        "candidate_receipt_sha256",
        "models",
        "target",
        "workload",
        "experiment",
        "primary_metric",
        "candidate_baseline_binding",
        "enabled_methods",
        "aggregate",
        "measured_runs",
        "quality",
    }:
        return False
    binding = payload.get("candidate_baseline_binding")
    models = payload.get("models")
    target = payload.get("target")
    workload = payload.get("workload")
    experiment = payload.get("experiment")
    enabled_methods = payload.get("enabled_methods")
    receipt_sha256 = payload.get("candidate_receipt_sha256")
    model_fields = {"id", "revision", "lineage_id", "source_id", "source_revision"}
    if (
        not isinstance(models, dict)
        or "target" not in models
        or any(
            not isinstance(model, dict)
            or set(model) != model_fields
            or any(not isinstance(model.get(field), str) or not model[field] for field in model_fields)
            for model in models.values()
        )
    ):
        return False
    measured_runs = payload.get("measured_runs")
    aggregate = payload.get("aggregate")
    if (
        not isinstance(receipt_sha256, str)
        or SHA256_RE.fullmatch(receipt_sha256) is None
        or not isinstance(measured_runs, list)
        or not measured_runs
        or not isinstance(aggregate, dict)
    ):
        return False
    metric_names = tuple(sorted(aggregate))
    if frozenset(metric_names) not in FINGERPRINT_METRIC_SETS:
        return False
    metric_values: dict[str, list[float | int]] = {metric: [] for metric in metric_names}
    seen_runs: set[int] = set()
    for run in measured_runs:
        if not isinstance(run, dict) or set(run) != {"run", "metrics", "raw_output"}:
            return False
        run_id = run.get("run")
        metrics = run.get("metrics")
        raw_output = run.get("raw_output")
        if (
            isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id < 1
            or run_id in seen_runs
            or not isinstance(metrics, dict)
            or set(metrics) != set(metric_names)
            or not isinstance(raw_output, dict)
            or set(raw_output) != {"path", "sha256", "size_bytes", "truncated"}
            or not isinstance(raw_output.get("path"), str)
            or not raw_output["path"]
            or not isinstance(raw_output.get("sha256"), str)
            or SHA256_RE.fullmatch(raw_output["sha256"]) is None
            or isinstance(raw_output.get("size_bytes"), bool)
            or not isinstance(raw_output.get("size_bytes"), int)
            or raw_output["size_bytes"] < 0
            or raw_output.get("truncated") is not False
        ):
            return False
        seen_runs.add(run_id)
        for metric in metric_names:
            metric_value = metrics.get(metric)
            if (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(float(metric_value))
            ):
                return False
            metric_values[metric].append(metric_value)
    for metric, values in metric_values.items():
        summary = aggregate.get(metric)
        expected = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if not isinstance(summary, dict) or set(summary) != set(expected):
            return False
        if any(
            isinstance(summary.get(key), bool)
            or not isinstance(summary.get(key), (int, float))
            or not math.isclose(float(summary[key]), float(expected_value), rel_tol=1e-12, abs_tol=1e-12)
            for key, expected_value in expected.items()
        ):
            return False
    quality = payload.get("quality")
    if (
        not isinstance(quality, dict)
        or set(quality) != {"status", "artifact", "result_sha256", "contract_identity"}
        or quality.get("status") != "pass"
        or not isinstance(quality.get("artifact"), dict)
        or set(quality["artifact"]) != {"path", "sha256", "size_bytes"}
        or not isinstance(quality["artifact"].get("path"), str)
        or not quality["artifact"]["path"]
        or not isinstance(quality["artifact"].get("sha256"), str)
        or SHA256_RE.fullmatch(quality["artifact"]["sha256"]) is None
        or isinstance(quality["artifact"].get("size_bytes"), bool)
        or not isinstance(quality["artifact"].get("size_bytes"), int)
        or quality["artifact"]["size_bytes"] < 0
        or quality.get("result_sha256") != quality["artifact"]["sha256"]
        or not _quality_contract_identity_valid(quality.get("contract_identity"))
    ):
        return False
    if (
        not isinstance(target, dict)
        or set(target) != {"descriptor", "sha256"}
        or not isinstance(target.get("descriptor"), dict)
        or not isinstance(target.get("sha256"), str)
        or SHA256_RE.fullmatch(target["sha256"]) is None
    ):
        return False
    if (
        not isinstance(workload, dict)
        or set(workload) != {"id", "artifacts", "parameters", "sha256"}
        or not isinstance(workload.get("id"), str)
        or not workload["id"]
        or not isinstance(workload.get("artifacts"), list)
        or not workload["artifacts"]
        or not isinstance(workload.get("parameters"), dict)
        or not isinstance(workload.get("sha256"), str)
        or SHA256_RE.fullmatch(workload["sha256"]) is None
    ):
        return False
    if (
        not isinstance(experiment, dict)
        or set(experiment) != {"invariant", "invariant_sha256", "variant", "variant_sha256"}
        or not isinstance(experiment.get("invariant"), dict)
        or not isinstance(experiment.get("variant"), dict)
        or any(
            not isinstance(experiment.get(field), str)
            or SHA256_RE.fullmatch(experiment[field]) is None
            for field in ("invariant_sha256", "variant_sha256")
        )
    ):
        return False
    if (
        not isinstance(binding, dict)
        or set(binding) != {"role", "baseline_receipt", "baseline_sha256"}
        or binding.get("role") != "candidate"
        or not isinstance(binding.get("baseline_receipt"), str)
        or not binding["baseline_receipt"]
        or not isinstance(binding.get("baseline_sha256"), str)
        or SHA256_RE.fullmatch(binding["baseline_sha256"]) is None
    ):
        return False
    if (
        payload.get("primary_metric") not in metric_names
        or not isinstance(enabled_methods, list)
        or not enabled_methods
        or any(not isinstance(method, str) or not method for method in enabled_methods)
    ):
        return False
    digest = value.get("sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        return False
    workload_descriptor = {
        "id": workload["id"],
        "artifacts": workload["artifacts"],
        "parameters": workload["parameters"],
    }
    return (
        target["sha256"] == _canonical_digest(target["descriptor"])
        and workload["sha256"] == _canonical_digest(workload_descriptor)
        and experiment["invariant_sha256"] == _canonical_digest(experiment["invariant"])
        and experiment["variant_sha256"] == _canonical_digest(experiment["variant"])
        and experiment["invariant"].get("primary_metric") == payload["primary_metric"]
        and experiment["variant"].get("enabled_methods") == enabled_methods
        and digest == _canonical_digest(payload)
    )


def experiment_identity_from_fingerprint(value: Any) -> dict[str, Any] | None:
    """Return the stable experiment identity shared by compatible repetitions.

    The full fingerprint remains receipt-specific and binds raw measurements and
    quality results.  This projection removes only sample evidence and the
    label-owned quality output path; it retains the exact correctness contract.
    """
    if not _experiment_fingerprint_valid(value):
        return None
    payload = value["payload"]
    experiment = payload["experiment"]
    variant = experiment["variant"]
    config = variant.get("config")
    runner = experiment["invariant"].get("runner")
    argv_template = runner.get("argv_template") if isinstance(runner, dict) else None
    external_runner = (
        isinstance(runner, dict)
        and runner.get("id") == "external-command-wall-time"
        and isinstance(argv_template, list)
        and any(
            isinstance(entry, dict)
            and entry.get("source") == ["variant_config", "quality_output_path"]
            for entry in argv_template
        )
    )
    semantic_config = (
        {key: item for key, item in config.items() if key != "quality_output_path"}
        if external_runner and isinstance(config, dict)
        else config
    )
    semantic_variant = {
        **variant,
        "config": semantic_config,
    }
    semantic_experiment = {
        "invariant": experiment["invariant"],
        "invariant_sha256": experiment["invariant_sha256"],
        "variant": semantic_variant,
        "variant_sha256": _canonical_digest(semantic_variant),
    }
    identity_payload = {
        **{field: payload[field] for field in EXPERIMENT_IDENTITY_FIELDS},
        "experiment": semantic_experiment,
        "quality_contract_identity": payload["quality"]["contract_identity"],
    }
    return {
        "schema_version": 1,
        "sha256": _canonical_digest(identity_payload),
        "payload": identity_payload,
    }


def _measured_on_from_experiment_fingerprint(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Project display metadata solely from the verified experiment identity."""
    payload = fingerprint["payload"]
    descriptor = payload["target"]["descriptor"]
    hardware = descriptor.get("hardware") if isinstance(descriptor.get("hardware"), dict) else {}
    software = descriptor.get("software") if isinstance(descriptor.get("software"), dict) else {}
    target_model = payload["models"]["target"]
    measured_on = {
        "chip": hardware.get("chip"),
        "mac_model": hardware.get("mac_model") or hardware.get("model"),
        "memory_bytes": hardware.get("memory_bytes"),
        "macos": software.get("macos"),
        "python": software.get("python"),
        "mlx": software.get("mlx"),
        "mlx_lm": software.get("mlx_lm"),
        "target_model": f"{target_model['id']}@{target_model['revision']}",
        "target_sha256": payload["target"]["sha256"],
        "workload_id": payload["workload"]["id"],
        "workload_sha256": payload["workload"]["sha256"],
        "experiment_fingerprint_sha256": fingerprint["sha256"],
    }
    return {key: value for key, value in measured_on.items() if value is not None}


def _assessment_rows(receipt_assessments: Any) -> list[dict[str, Any]]:
    if not isinstance(receipt_assessments, dict):
        return []
    rows = receipt_assessments.get("assessments")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _receipt_identity(receipt: Any) -> tuple[str | None, str | None]:
    if isinstance(receipt, str):
        return receipt, None
    if not isinstance(receipt, dict):
        return None, None
    receipt_ref = receipt.get("file")
    label = receipt.get("label")
    return (
        str(receipt_ref) if isinstance(receipt_ref, str) and receipt_ref else None,
        str(label) if isinstance(label, str) and label else None,
    )


def _matching_receipt_assessment(
    receipt: Any,
    receipt_assessments: Any,
) -> dict[str, Any] | None:
    receipt_ref, label = _receipt_identity(receipt)
    for row in _assessment_rows(receipt_assessments):
        assessed_receipt = row.get("receipt")
        assessed_label = row.get("label")
        if receipt_ref and assessed_receipt == receipt_ref:
            return row
        if not receipt_ref and label and assessed_label == label:
            return row
    return None


def _measured_compound(
    receipts: Any,
    receipt_assessments: Any,
    ordered_enabled_methods: list[str],
    stack_metric: str,
    verified_receipts: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Promote a measured stack only through an external, exact-coverage assessment."""
    if not isinstance(receipts, list):
        return None, []
    promotable: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    candidate_set_blocked = False
    for receipt in receipts:
        receipt_ref, label = _receipt_identity(receipt)
        assessment = _matching_receipt_assessment(receipt, receipt_assessments)
        if assessment is None:
            evidence.append({
                "receipt": receipt_ref or label,
                "label": label,
                "classification": "incomplete",
                "reasons": ["missing external receipt assessment"],
                "enabled_methods": [],
                "gates": {},
            })
            candidate_set_blocked = True
            continue

        classification = str(assessment.get("classification", ""))
        assessed_methods = assessment.get("enabled_methods")
        reasons = [str(reason) for reason in assessment.get("reasons", [])]
        base_evidence = {
            "receipt": assessment.get("receipt") or receipt_ref or label,
            "label": assessment.get("label") or label,
            "receipt_sha256": assessment.get("receipt_sha256"),
            "classification": classification,
            "enabled_methods": assessed_methods if isinstance(assessed_methods, list) else [],
            "reasons": reasons,
            "gates": assessment.get("gates") if isinstance(assessment.get("gates"), dict) else {},
            "primary_metric": assessment.get("primary_metric"),
            "experiment_fingerprint": assessment.get("experiment_fingerprint"),
            "recomputed_median_ratios": (
                assessment.get("recomputed_median_ratios")
                if isinstance(assessment.get("recomputed_median_ratios"), dict)
                else {}
            ),
        }
        if classification not in COMPOUND_ASSESSMENT_CLASSIFICATIONS:
            base_evidence["original_classification"] = classification or None
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [
                *reasons,
                "receipt assessment has an invalid classification",
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue
        if classification != "promotion_ready":
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue
        promotion_failures: list[str] = []
        if assessment.get("promotion_ready") is not True:
            promotion_failures.append("assessment promotion_ready boolean is not true")
        if reasons:
            promotion_failures.append("promotion-ready assessment reasons must be empty")
        receipt_sha256 = assessment.get("receipt_sha256")
        if not isinstance(receipt_sha256, str) or SHA256_RE.fullmatch(receipt_sha256) is None:
            promotion_failures.append("assessment receipt_sha256 is not a 64-hex digest")
        gates = assessment.get("gates")
        for gate in COMPOUND_PROMOTION_REQUIRED_GATES:
            if not isinstance(gates, dict) or gates.get(gate) is not True:
                promotion_failures.append(f"assessment gate {gate} is not true")
        if assessed_methods != ordered_enabled_methods:
            promotion_failures.append(
                "assessment ordered enabled_methods do not exactly match the stack"
            )
        fingerprint = assessment.get("experiment_fingerprint")
        if not _experiment_fingerprint_valid(fingerprint):
            promotion_failures.append("assessment experiment_fingerprint is invalid")
        else:
            identity = experiment_identity_from_fingerprint(fingerprint)
            if identity is None:
                promotion_failures.append("assessment experiment identity is invalid")
            if fingerprint["payload"].get("candidate_receipt_sha256") != receipt_sha256:
                promotion_failures.append(
                    "assessment experiment_fingerprint receipt_sha256 does not match the assessment"
                )
            if fingerprint["payload"].get("enabled_methods") != assessed_methods:
                promotion_failures.append(
                    "assessment experiment_fingerprint enabled_methods do not match the assessment"
                )
            if fingerprint["payload"].get("primary_metric") != assessment.get("primary_metric"):
                promotion_failures.append(
                    "assessment experiment_fingerprint primary_metric does not match the assessment"
                )
        if promotion_failures:
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [
                *reasons,
                *promotion_failures,
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue

        metric_policy = STACK_MEASURED_METRIC_POLICY.get(stack_metric)
        if metric_policy is None:
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [
                *reasons,
                f"stack metric {stack_metric} has no controlled assessment ratio mapping",
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue
        required_primary_metric, ratio_key = metric_policy
        if assessment.get("primary_metric") != required_primary_metric:
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [
                *reasons,
                f"assessment primary_metric must be {required_primary_metric} for {stack_metric}",
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue
        ratios = assessment.get("recomputed_median_ratios")
        ratio = ratios.get(ratio_key) if isinstance(ratios, dict) else None
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or float(ratio) <= 0
        ):
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [
                *reasons,
                f"assessment recomputed_median_ratios lacks valid {ratio_key}",
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue
        if float(ratio) <= 1.0:
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "rejected"
            base_evidence["reasons"] = [
                *reasons,
                f"assessment recomputed median ratio {ratio_key}={_format_multiplier(float(ratio))} does not exceed 1.0",
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue

        verified_assessment = (
            verified_receipts.get(receipt_ref)
            if isinstance(verified_receipts, dict) and isinstance(receipt_ref, str)
            else None
        )
        if not isinstance(verified_assessment, dict) or verified_assessment != assessment:
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [
                *reasons,
                "promotion-ready assessment is not backed by a verified receipt recomputation",
            ]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue

        identity = experiment_identity_from_fingerprint(fingerprint)
        if identity is None:
            base_evidence["original_classification"] = "promotion_ready"
            base_evidence["classification"] = "incomplete"
            base_evidence["reasons"] = [*reasons, "assessment experiment identity is invalid"]
            evidence.append(base_evidence)
            candidate_set_blocked = True
            continue
        promotable.append((float(ratio), assessment, fingerprint, identity))

    if candidate_set_blocked:
        return None, evidence

    identity_digests = {identity["sha256"] for _, _, _, identity in promotable}
    if len(identity_digests) > 1:
        for _, assessment, fingerprint, _ in promotable:
            evidence.append({
                "receipt": assessment.get("receipt"),
                "label": assessment.get("label"),
                "receipt_sha256": assessment.get("receipt_sha256"),
                "classification": "incomplete",
                "original_classification": "promotion_ready",
                "enabled_methods": assessment.get("enabled_methods", []),
                "gates": assessment.get("gates", {}),
                "experiment_fingerprint": fingerprint,
                "reasons": ["heterogeneous verified experiment fingerprints"],
            })
        return None, evidence
    if not promotable:
        return None, evidence

    ratio, assessment, fingerprint, identity = min(
        promotable,
        key=lambda item: (item[0], str(item[1].get("receipt", ""))),
    )
    measured = {
        "ratio": _format_multiplier(ratio),
        "provenance": "local_reproduced",
        "assessment_classification": "promotion_ready",
        "enabled_methods": list(ordered_enabled_methods),
        "metric": stack_metric,
        "assessment_primary_metric": STACK_MEASURED_METRIC_POLICY[stack_metric][0],
        "assessment_ratio_key": STACK_MEASURED_METRIC_POLICY[stack_metric][1],
        "receipt": assessment.get("receipt"),
        "receipt_sha256": assessment.get("receipt_sha256"),
        "label": assessment.get("label"),
        "file": assessment.get("receipt"),
        "experiment_fingerprint": fingerprint,
        "experiment_fingerprint_sha256": fingerprint["sha256"],
        "experiment_identity_sha256": identity["sha256"],
        "compatible_repetition_count": len(promotable),
        "compatible_repetitions": [
            {
                "receipt": row.get("receipt"),
                "receipt_sha256": row.get("receipt_sha256"),
                "experiment_fingerprint_sha256": full_fingerprint["sha256"],
            }
            for _, row, full_fingerprint, _ in sorted(
                promotable,
                key=lambda item: str(item[1].get("receipt", "")),
            )
        ],
        "measured_on": _measured_on_from_experiment_fingerprint(fingerprint),
        "basis": "Ratio recomputed by the external receipt assessment.",
        "floor": "1.0x",
    }
    return measured, evidence


def compose_stack_band(
    stack: dict[str, Any],
    guidance_methods: dict[str, Any] | list[dict[str, Any]],
    receipt_assessments: dict[str, Any] | None = None,
    *,
    verified_receipts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive an advisory compound band for an optimization stack.

    Numeric products are emitted only when every stack-step pair is explicitly
    validated-composable and every multiplied band has a unique evidence
    lineage. Unknown, conflicting, mutually-exclusive, or duplicate-lineage
    combinations stay as individual experiments without a compound number.
    """
    if isinstance(guidance_methods, dict):
        methods_by_id = guidance_methods
    else:
        methods_by_id = {str(method.get("id")): method for method in guidance_methods if isinstance(method, dict)}

    primary_metric = stack.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ValueError(f"stack {stack.get('id', '<unknown>')} primary_metric must be a non-empty string")

    steps = stack.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError(f"stack {stack.get('id', '<unknown>')} steps must be a list")
    step_ids: list[str] = []
    seen_step_ids: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"stack {stack.get('id', '<unknown>')} contains a non-mapping step")
        method_id = str(step.get("method", ""))
        if method_id in seen_step_ids:
            raise ValueError(
                f"stack {stack.get('id', '<unknown>')} has duplicate step method {method_id}"
            )
        seen_step_ids.add(method_id)
        step_ids.append(method_id)

    pair_validity: dict[tuple[str, str], str] = {}
    excluded_conflicts: list[list[str]] = []
    mutually_exclusive_pairs: list[list[str]] = []
    for note in stack.get("composition_notes", []) or []:
        if not isinstance(note, dict):
            raise ValueError(f"stack {stack.get('id', '<unknown>')} contains a non-mapping composition note")
        pair = note.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"stack {stack.get('id', '<unknown>')} has malformed composition pair {pair!r}")
        normalized_pair = tuple(sorted((str(pair[0]), str(pair[1]))))
        if normalized_pair[0] == normalized_pair[1]:
            raise ValueError(
                f"stack {stack.get('id', '<unknown>')} has self composition pair {normalized_pair[0]}"
            )
        if normalized_pair in pair_validity:
            raise ValueError(
                f"stack {stack.get('id', '<unknown>')} has duplicate composition pair "
                f"{normalized_pair[0]} + {normalized_pair[1]}"
            )
        validity = str(note.get("validity", "unknown"))
        pair_validity[normalized_pair] = validity
        if validity == "known-conflicting":
            excluded_conflicts.append(list(normalized_pair))
        elif validity == "mutually-exclusive":
            mutually_exclusive_pairs.append(list(normalized_pair))

    per_step: list[dict[str, Any]] = []
    unmeasured_upside: list[str] = []
    other_metric_upside: list[dict[str, str]] = []
    numeric_steps: list[dict[str, Any]] = []
    numeric_authority_holds: list[str] = []
    ceiling = 1.0

    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"stack {stack.get('id', '<unknown>')} contains a non-mapping step")
        method_id = str(step.get("method", ""))
        method = methods_by_id.get(method_id, {})
        improvement_band = method.get("improvement_band") if isinstance(method, dict) else None
        band_range = None
        band_provenance = None
        if isinstance(improvement_band, dict):
            band_range = improvement_band.get("range")
            band_provenance = improvement_band.get("provenance")
        lineage_ids = (
            [str(value) for value in improvement_band.get("evidence_lineage_ids", [])]
            if isinstance(improvement_band, dict)
            else []
        )
        numeric_authority = (
            improvement_band.get("numeric_authority")
            if isinstance(improvement_band, dict)
            else None
        )

        per_step.append({
            "method": method_id,
            "band": (
                band_range
                if isinstance(band_range, str) and numeric_authority == "effective_claims"
                else None
            ),
            "lossiness": step.get("lossiness"),
            "gate": step.get("gate"),
            "evidence_lineage_ids": lineage_ids,
            "numeric_authority": numeric_authority,
        })

        if not isinstance(improvement_band, dict) or band_provenance == "profile_required":
            unmeasured_upside.append(method_id)
            continue
        if band_provenance not in {"source_reported", "local_reproduced"}:
            unmeasured_upside.append(method_id)
            continue
        if improvement_band.get("numeric_authority") != "effective_claims":
            numeric_authority_holds.append(
                f"method {method_id} lacks effective_claims numeric authority"
            )
            unmeasured_upside.append(method_id)
            continue
        band_metric = improvement_band.get("metric")
        if band_metric != primary_metric:
            if isinstance(band_range, str) and isinstance(band_metric, str):
                other_metric_upside.append({
                    "method": method_id,
                    "metric": band_metric,
                    "range": band_range,
                })
            continue
        _, step_ceiling = parse_band(str(band_range))
        numeric_steps.append({
            "method": method_id,
            "ceiling": step_ceiling,
            "evidence_lineage_ids": lineage_ids,
        })

    withheld_reasons: list[str] = list(numeric_authority_holds)
    for left_index, left in enumerate(step_ids):
        for right in step_ids[left_index + 1:]:
            pair = tuple(sorted((left, right)))
            validity = pair_validity.get(pair, "unknown")
            if validity != "validated-composable":
                withheld_reasons.append(
                    f"pair {pair[0]} + {pair[1]} is {validity}; numeric composition requires validated-composable"
                )

    lineage_owner: dict[str, str] = {}
    for numeric_step in numeric_steps:
        method_id = str(numeric_step["method"])
        lineages = numeric_step["evidence_lineage_ids"]
        if not lineages:
            withheld_reasons.append(f"method {method_id} has no evidence lineage id")
            continue
        for lineage_id in lineages:
            previous = lineage_owner.get(lineage_id)
            if lineage_id in lineage_owner:
                withheld_reasons.append(
                    f"duplicate evidence lineage {lineage_id} is shared by {previous} and {method_id}"
                )
            else:
                lineage_owner[lineage_id] = method_id

    numeric_compound_allowed = len(numeric_steps) >= 2 and not withheld_reasons
    if numeric_compound_allowed:
        for numeric_step in numeric_steps:
            ceiling *= float(numeric_step["ceiling"])

    compound = stack.get("compound", {}) if isinstance(stack.get("compound", {}), dict) else {}
    measured_together = compound.get("measured_together") is True
    receipts = compound.get("receipts", [])
    hypothesis_ceiling = {
        "floor": "1.0x",
        "ceiling": _format_multiplier(ceiling) if numeric_compound_allowed else None,
        "metric": primary_metric,
        "provenance": "multiplicative_hypothesis" if numeric_compound_allowed else "withheld",
        "flag": (
            STACK_COMPOUND_HYPOTHESIS_FLAG
            if numeric_compound_allowed
            else "numeric compound withheld until every pair is validated-composable with unique evidence lineage"
        ),
    }
    result = {
        "primary_metric": primary_metric,
        "composition_status": "numeric-hypothesis" if numeric_compound_allowed else "withheld",
        "hypothesis_ceiling": hypothesis_ceiling,
        "withheld_reasons": list(dict.fromkeys(withheld_reasons)),
        "unmeasured_upside": unmeasured_upside,
        "other_metric_upside": other_metric_upside,
        "excluded_conflicts": excluded_conflicts,
        "mutually_exclusive_pairs": mutually_exclusive_pairs,
        "per_step": per_step,
    }
    measured, measured_evidence = (
        _measured_compound(
            receipts,
            receipt_assessments,
            step_ids,
            primary_metric,
            verified_receipts,
        )
        if measured_together
        else (None, [])
    )
    if measured:
        result["measured"] = measured
    if measured_evidence:
        result["measured_evidence"] = measured_evidence
    return result
