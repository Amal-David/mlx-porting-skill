#!/usr/bin/env python3
"""Run a command repeatedly and capture reproducible wall-time/RSS/environment data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import shutil
import signal
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from _common import SkillError, atomic_write_text, dump_json, redact_secret_text


ENVIRONMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RECEIPT_SPEC_FIELDS = {
    "schema_version",
    "label",
    "argv_template",
    "models",
    "workload",
    "variant_config",
    "enabled_methods",
    "comparison_role",
    "rollback_condition",
}
MAX_RECEIPT_RUNS = 100
MAX_RECEIPT_WARMUPS = 20
MAX_RECEIPT_TAIL_BYTES = 64 * 1024
MAX_RECEIPT_REPORT_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_TIMEOUT_SECONDS = 3600.0
MODEL_IDENTITY_FIELDS = {"id", "revision", "lineage_id", "source_id", "source_revision"}
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
RECEIPT_INTERPRETER_FLAGS = ("-I", "-B")
MAX_INTERPRETER_ENVIRONMENT_BYTES = 2 * 1024 * 1024
MAX_INTERPRETER_PROBE_STDERR_BYTES = 64 * 1024
MAX_INTERPRETER_VERSION_BYTES = 4 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an external command without invoking a shell")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--output")
    parser.add_argument("--cwd")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--stdout-tail", type=int, default=4000)
    parser.add_argument("--stderr-tail", type=int, default=4000)
    parser.add_argument("--receipt-spec", help="Controlled external wall-time receipt specification JSON")
    parser.add_argument("--quality-contract", help="Schema-2 exact-output quality contract for receipt mode")
    parser.add_argument("--baseline-receipt", help="Compatible schema-2 baseline receipt for candidate receipt mode")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def system_command(args: list[str]) -> str | None:
    try:
        stdout, _stderr = _run_bounded_probe(
            args,
            env=dict(os.environ),
            timeout=3,
            stdout_limit=64 * 1024,
            stderr_limit=64 * 1024,
            label="system metadata probe",
        )
        value = stdout.decode("utf-8", errors="strict").strip()
        return value or None
    except (OSError, SkillError, UnicodeDecodeError):
        return None


def environment_metadata() -> dict[str, Any]:
    data: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        data["mac_model"] = system_command(["sysctl", "-n", "hw.model"])
        data["cpu_brand"] = system_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = system_command(["sysctl", "-n", "hw.memsize"])
        data["memory_bytes"] = int(memory) if memory and memory.isdigit() else None
        data["macos_version"] = system_command(["sw_vers", "-productVersion"])
    try:
        data["mlx_version"] = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        data["mlx_version"] = None
    return data


def parse_environment_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SkillError(f"--env must be KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        if not key:
            raise SkillError("--env key cannot be empty")
        if ENVIRONMENT_KEY_RE.fullmatch(key) is None:
            raise SkillError(f"--env key must be a portable environment name, got {key!r}")
        overrides[key] = val
    return overrides


def parse_env(values: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(parse_environment_overrides(values))
    return env


def redact_environment_values(value: str, overrides: dict[str, str]) -> str:
    """Remove explicit override values if a benchmarked command echoes them."""
    safe = redact_secret_text(value)
    for environment_value in sorted(set(overrides.values()), key=len, reverse=True):
        if environment_value:
            safe = safe.replace(environment_value, "[REDACTED]")
    return safe


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    proc.kill()


class _TailCapture:
    """Retain only the requested tail while draining a process stream."""

    def __init__(self, limit: int):
        self.limit = limit
        self.data = bytearray()
        self.total_bytes = 0

    def add(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if self.limit == 0:
            return
        self.data.extend(chunk)
        if len(self.data) > self.limit:
            del self.data[:-self.limit]

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


def _drain_stream(stream: BinaryIO, capture: _TailCapture) -> None:
    try:
        while chunk := stream.read(64 * 1024):
            capture.add(chunk)
    except (OSError, ValueError):
        pass
    finally:
        stream.close()


def _close_streams(streams: tuple[BinaryIO, BinaryIO]) -> None:
    for stream in streams:
        try:
            stream.close()
        except OSError:
            pass


def _run_bounded_probe(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    stdout_limit: int,
    stderr_limit: int,
    label: str,
) -> tuple[bytes, bytes]:
    """Run a metadata probe without allowing either captured stream to grow unbounded."""
    if stdout_limit < 0 or stderr_limit < 0:
        raise SkillError(f"{label} output limits must be non-negative")
    stdout_capture = _TailCapture(stdout_limit)
    stderr_capture = _TailCapture(stderr_limit)
    process: subprocess.Popen[bytes] | None = None
    streams: tuple[BinaryIO, BinaryIO] | None = None
    started_threads: list[threading.Thread] = []
    close_first = False
    termination_requested = threading.Event()
    termination_lock = threading.Lock()

    def request_termination() -> None:
        assert process is not None
        with termination_lock:
            if termination_requested.is_set():
                return
            termination_requested.set()
            kill_process_tree(process)

    def drain(stream: BinaryIO, capture: _TailCapture) -> None:
        try:
            while chunk := stream.read(64 * 1024):
                capture.add(chunk)
                if capture.total_bytes > capture.limit:
                    request_termination()
                    break
        except (OSError, ValueError):
            pass
        finally:
            stream.close()

    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    try:
        if process.stdout is None or process.stderr is None:
            raise SkillError(f"{label} did not expose capture pipes")
        streams = (process.stdout, process.stderr)
        threads = (
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout_capture),
                daemon=True,
                name="mlx-interpreter-probe-stdout",
            ),
            threading.Thread(
                target=drain,
                args=(process.stderr, stderr_capture),
                daemon=True,
                name="mlx-interpreter-probe-stderr",
            ),
        )
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        try:
            process.wait(timeout=timeout)
            # A probe must not leave descendants holding its capture pipes open.
            request_termination()
        except subprocess.TimeoutExpired as exc:
            request_termination()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            raise SkillError(f"{label} timed out after {timeout:g} seconds") from exc
    except BaseException:
        close_first = True
        request_termination()
        try:
            process.wait(timeout=1)
        except BaseException:
            pass
        raise
    finally:
        if streams is not None and close_first:
            _close_streams(streams)
        for thread in started_threads:
            thread.join(timeout=1)
        if streams is not None and any(thread.is_alive() for thread in started_threads):
            _close_streams(streams)
            for thread in started_threads:
                if thread.is_alive():
                    thread.join(timeout=1)

    if stdout_capture.total_bytes > stdout_limit:
        raise SkillError(f"{label} stdout exceeded {stdout_limit} bytes")
    if stderr_capture.total_bytes > stderr_limit:
        raise SkillError(f"{label} stderr exceeded {stderr_limit} bytes")
    if any(thread.is_alive() for thread in started_threads):
        raise SkillError(f"{label} output drain did not stop")
    if process.returncode != 0:
        raise SkillError(f"{label} failed with exit status {process.returncode}")
    return bytes(stdout_capture.data), bytes(stderr_capture.data)


def run_once(
    command: list[str],
    *,
    cwd: str | None,
    env: dict[str, str],
    timeout: float | None,
    stdout_tail: int,
    stderr_tail: int,
) -> dict[str, Any]:
    if stdout_tail < 0 or stderr_tail < 0:
        raise SkillError("stdout/stderr tail sizes must be non-negative")
    rss_peak: int | None = None
    stop = threading.Event()
    proc: subprocess.Popen[bytes] | None = None

    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None

    def monitor() -> None:
        nonlocal rss_peak
        assert proc is not None
        if psutil is None:
            return
        try:
            root = psutil.Process(proc.pid)
            while not stop.wait(0.01):
                processes = [root] + root.children(recursive=True)
                rss = 0
                for item in processes:
                    try:
                        rss += int(item.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                rss_peak = max(rss_peak or 0, rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    start = time.perf_counter()
    timed_out = False
    stdout_capture = _TailCapture(stdout_tail)
    stderr_capture = _TailCapture(stderr_tail)
    streams: tuple[BinaryIO, BinaryIO] | None = None
    started_threads: list[threading.Thread] = []
    close_first = False
    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    try:
        if proc.stdout is None or proc.stderr is None:
            raise SkillError("benchmark subprocess did not expose capture pipes")
        streams = (proc.stdout, proc.stderr)
        threads = (
            threading.Thread(
                target=_drain_stream,
                args=(proc.stdout, stdout_capture),
                daemon=True,
                name="mlx-benchmark-stdout",
            ),
            threading.Thread(
                target=_drain_stream,
                args=(proc.stderr, stderr_capture),
                daemon=True,
                name="mlx-benchmark-stderr",
            ),
            threading.Thread(target=monitor, daemon=True, name="mlx-benchmark-rss"),
        )
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        try:
            proc.wait(timeout=timeout)
            # Reap children that remain in the command's process group after
            # a short-lived group leader exits.
            kill_process_tree(proc)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(proc)
            proc.wait()
    except BaseException:
        close_first = True
        kill_process_tree(proc)
        try:
            proc.wait(timeout=1)
        except BaseException:
            pass
        raise
    finally:
        stop.set()
        if streams is not None and close_first:
            _close_streams(streams)
        for thread in started_threads:
            thread.join(timeout=1)
        if streams is not None and any(thread.is_alive() for thread in started_threads):
            _close_streams(streams)
            for thread in started_threads:
                if thread.is_alive():
                    thread.join(timeout=1)
    elapsed = time.perf_counter() - start
    return {
        "wall_seconds": elapsed,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "peak_rss_bytes": rss_peak,
        "stdout_tail": stdout_capture.text(),
        "stderr_tail": stderr_capture.text(),
    }


def load_receipt_spec(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillError(f"could not read --receipt-spec {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != RECEIPT_SPEC_FIELDS:
        raise SkillError("receipt spec has an invalid field set")
    if value.get("schema_version") != 1:
        raise SkillError("receipt spec must use schema_version 1")
    label = value.get("label")
    if not isinstance(label, str) or LABEL_RE.fullmatch(label) is None:
        raise SkillError("receipt spec label must be a portable slug")
    if value.get("comparison_role") not in {"baseline", "candidate", "observation"}:
        raise SkillError("receipt spec comparison_role must be baseline, candidate, or observation")
    models = value.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("target"), dict):
        raise SkillError("receipt spec must declare models.target")
    for role, model in models.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(model, dict)
            or set(model) != MODEL_IDENTITY_FIELDS
            or any(not isinstance(model.get(field), str) or not model[field] for field in MODEL_IDENTITY_FIELDS)
            or PINNED_REVISION_RE.fullmatch(model["revision"]) is None
            or PINNED_REVISION_RE.fullmatch(model["source_revision"]) is None
        ):
            raise SkillError(
                "receipt spec model identities require exact non-empty fields and pinned revisions"
            )
    if not isinstance(value.get("workload"), dict):
        raise SkillError("receipt spec must declare workload")
    if not isinstance(value.get("variant_config"), dict) or not value["variant_config"]:
        raise SkillError("receipt spec variant_config must be a non-empty object")
    enabled_methods = value.get("enabled_methods")
    if (
        not isinstance(enabled_methods, list)
        or any(not isinstance(method, str) or not method for method in enabled_methods)
        or len(enabled_methods) != len(set(enabled_methods))
    ):
        raise SkillError("receipt spec enabled_methods must contain unique non-empty strings")
    if value["comparison_role"] == "candidate" and not enabled_methods:
        raise SkillError("candidate receipt spec must declare enabled_methods")
    if value["comparison_role"] != "candidate" and enabled_methods:
        raise SkillError("baseline and observation receipt specs must not enable methods")
    rollback = value.get("rollback_condition")
    if not isinstance(rollback, str) or not rollback.strip():
        raise SkillError("receipt spec rollback_condition must be non-empty")
    return value


def validate_receipt_workload_artifacts(root: Path, spec: dict[str, Any]) -> None:
    import validate_benchmarks as validation

    artifacts = spec.get("workload", {}).get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SkillError("receipt spec workload must contain checked-in artifacts")
    runner_count = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("role"), str):
            raise SkillError("receipt spec workload artifacts require a role")
        valid, reason = validation.check_artifact(root, artifact)
        if not valid:
            raise SkillError(f"receipt spec workload artifact is invalid: {reason or 'invalid'}")
        runner_count += artifact.get("role") == "runner"
    if runner_count != 1:
        raise SkillError("receipt spec workload requires exactly one digest-pinned runner artifact")


def controlled_receipt_environment() -> dict[str, str]:
    """Return the small ambient environment allowed to affect receipt execution."""
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
    }
    if sys.platform == "win32":
        for key in ("SYSTEMROOT", "WINDIR"):
            if os.environ.get(key):
                environment[key] = os.environ[key]
    return environment


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_receipt_output_path(root: Path, relative_value: str) -> Path:
    """Create a receipt-owned parent path without following symlinks."""
    import validate_benchmarks as validation

    relative = Path(relative_value)
    if validation.resolve_artifact(root, relative_value) is None:
        raise SkillError(f"receipt output path is unsafe: {relative_value}")
    resolved_root = root.resolve()
    current = resolved_root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise SkillError(f"receipt output parent is unsafe: {relative_value}")
        current.mkdir(exist_ok=True)
    output = resolved_root / relative
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise SkillError(f"receipt output is unsafe: {relative_value}")
    return output


def resolved_python_environment(
    interpreter: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Bind the isolated interpreter's automatic import/package surface."""
    probe = """
import hashlib
import importlib.metadata
import json
import pathlib
import sys

def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None

distributions = []
for dist in importlib.metadata.distributions():
    try:
        metadata = dist.read_text("METADATA") or ""
        record = dist.read_text("RECORD") or ""
        direct_url = dist.read_text("direct_url.json") or ""
        name = dist.metadata.get("Name") or ""
        version = dist.version or ""
        location = str(pathlib.Path(dist.locate_file("")).resolve())
    except Exception:
        continue
    if not name or not version:
        continue
    distributions.append({
        "name": name,
        "version": version,
        "location_sha256": sha(location),
        "metadata_sha256": sha(metadata),
        "record_sha256": sha(record),
        "direct_url_sha256": sha(direct_url),
    })

startup_files = []
seen = set()
for entry in sys.path:
    base = pathlib.Path(entry)
    if not base.is_dir():
        continue
    for candidate in [*base.glob("*.pth"), base / "sitecustomize.py", base / "usercustomize.py"]:
        try:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_file() or resolved.is_symlink():
                continue
            raw = resolved.read_bytes()
        except OSError:
            continue
        seen.add(resolved)
        startup_files.append({
            "path_sha256": sha(str(resolved)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        })

print(json.dumps({
    "schema_version": 1,
    "sys_path_sha256": sha(json.dumps(list(sys.path), separators=(",", ":"))),
    "distributions": sorted(distributions, key=lambda row: (
        row["name"].lower(), row["version"], row["location_sha256"]
    )),
    "startup_files": sorted(startup_files, key=lambda row: row["path_sha256"]),
}, sort_keys=True, separators=(",", ":")))
"""
    try:
        stdout, _ = _run_bounded_probe(
            [str(interpreter), *RECEIPT_INTERPRETER_FLAGS, "-c", probe],
            env=environment,
            timeout=15,
            stdout_limit=MAX_INTERPRETER_ENVIRONMENT_BYTES,
            stderr_limit=MAX_INTERPRETER_PROBE_STDERR_BYTES,
            label="receipt interpreter environment probe",
        )
    except (OSError, subprocess.SubprocessError, SkillError) as exc:
        raise SkillError(f"could not inspect receipt interpreter environment: {exc}") from exc
    if not stdout:
        raise SkillError("receipt interpreter environment descriptor is empty")
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError("receipt interpreter environment descriptor is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "sys_path_sha256", "distributions", "startup_files"}
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("sys_path_sha256"), str)
        or not isinstance(payload.get("distributions"), list)
        or not isinstance(payload.get("startup_files"), list)
    ):
        raise SkillError("receipt interpreter environment descriptor has an invalid shape")
    return {**payload, "sha256": canonical_sha256(payload)}


def resolve_receipt_interpreter(
    command: list[str],
    environment: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    name = command[0] if command else ""
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or re.fullmatch(r"python(?:3(?:\.\d+)*)?", name) is None
    ):
        raise SkillError("receipt mode requires a bare allowlisted Python interpreter name")
    located = shutil.which(name, path=environment.get("PATH"))
    if not located:
        raise SkillError(f"could not resolve receipt interpreter {name!r}")
    resolved = Path(located).resolve()
    if not resolved.is_file():
        raise SkillError("resolved receipt interpreter is not a regular file")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    try:
        version_stdout, version_stderr = _run_bounded_probe(
            [str(resolved), *RECEIPT_INTERPRETER_FLAGS, "--version"],
            env=environment,
            timeout=3,
            stdout_limit=MAX_INTERPRETER_VERSION_BYTES,
            stderr_limit=MAX_INTERPRETER_VERSION_BYTES,
            label="receipt interpreter version probe",
        )
    except (OSError, subprocess.SubprocessError, SkillError) as exc:
        raise SkillError(f"could not identify receipt interpreter: {exc}") from exc
    try:
        version = (version_stdout or version_stderr).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SkillError("receipt interpreter reported a non-UTF-8 version") from exc
    if not version:
        raise SkillError("receipt interpreter did not report a version")
    return str(resolved), {
        "name": name,
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
        "version": version,
        "flags": list(RECEIPT_INTERPRETER_FLAGS),
        "environment": resolved_python_environment(resolved, environment),
    }


def attested_system_metadata(
    system: dict[str, Any],
    interpreter: dict[str, Any],
) -> dict[str, Any]:
    """Project software versions from the interpreter that executed the runner."""
    result = dict(system)
    version = interpreter.get("version")
    if isinstance(version, str) and version.startswith("Python "):
        result["python"] = version.removeprefix("Python ")
    environment = interpreter.get("environment")
    distributions = environment.get("distributions") if isinstance(environment, dict) else None
    mlx_versions = [
        distribution.get("version")
        for distribution in distributions
        if isinstance(distribution, dict)
        and str(distribution.get("name", "")).lower() == "mlx"
        and isinstance(distribution.get("version"), str)
        and distribution["version"]
    ] if isinstance(distributions, list) else []
    if len(mlx_versions) != 1:
        raise SkillError("attested interpreter must expose exactly one MLX distribution version")
    result["mlx_version"] = mlx_versions[0]
    return result


def prepare_receipt_quality_output(
    root: Path,
    spec: dict[str, Any],
    contract_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate and resolve the output the measured runner must recreate."""
    import validate_benchmarks as validation

    try:
        safe_contract = contract_path.resolve()
        if (
            contract_path.is_symlink()
            or not safe_contract.is_file()
        ):
            raise SkillError("quality contract must be a regular non-symlink file")
        contract_bytes = safe_contract.read_bytes()
        if len(contract_bytes) > validation.MAX_QUALITY_ARTIFACT_BYTES:
            raise SkillError("quality contract exceeds the controlled artifact limit")
        contract = json.loads(contract_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"could not read --quality-contract {contract_path}: {exc}") from exc
    required = {
        "schema_version",
        "validator",
        "metric",
        "reference_artifact",
        "candidate_artifact",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != required
        or contract.get("schema_version") != 2
        or not validation.exact_output_validator_descriptor_valid(contract.get("validator"))
        or contract.get("metric") != "exact-output-parity"
    ):
        raise SkillError("receipt mode requires a schema-2 exact-output quality contract")
    reference = contract.get("reference_artifact")
    candidate = contract.get("candidate_artifact")
    reference_valid, reference_reason = validation.check_artifact(root, reference)
    if not reference_valid:
        raise SkillError(
            f"controlled quality reference artifact is invalid: {reference_reason or 'invalid'}"
        )
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"path", "sha256", "size_bytes"}
        or not isinstance(candidate.get("sha256"), str)
        or validation.HEX_DIGEST_RE.fullmatch(candidate["sha256"]) is None
        or isinstance(candidate.get("size_bytes"), bool)
        or not isinstance(candidate.get("size_bytes"), int)
        or not 0 <= candidate["size_bytes"] <= validation.MAX_QUALITY_ARTIFACT_BYTES
    ):
        raise SkillError("controlled quality candidate artifact descriptor is invalid")
    declared_candidate_path = candidate.get("path")
    relative_candidate_path = Path(str(declared_candidate_path))
    expected_prefix = ("quality", "outputs", str(spec.get("label")))
    if (
        not isinstance(declared_candidate_path, str)
        or relative_candidate_path.is_absolute()
        or relative_candidate_path.parts[:3] != expected_prefix
        or len(relative_candidate_path.parts) != 4
        or any(part in {"", ".", ".."} for part in relative_candidate_path.parts)
    ):
        raise SkillError(
            "controlled quality output must be quality/outputs/<label>/<file>"
        )
    unresolved_candidate_path = root / relative_candidate_path
    for parent in [root / Path(*relative_candidate_path.parts[:index]) for index in range(1, 4)]:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise SkillError("controlled quality output path contains an unsafe parent")
    if unresolved_candidate_path.is_symlink():
        raise SkillError("controlled quality candidate artifact must not be a symlink")
    candidate_path = validation.resolve_artifact(root, declared_candidate_path)
    if candidate_path is None or declared_candidate_path == (
        reference.get("path") if isinstance(reference, dict) else None
    ):
        raise SkillError("controlled quality candidate artifact path is invalid")

    variant_config = spec.get("variant_config")
    output_source = ["variant_config", "quality_output_path"]
    if (
        not isinstance(variant_config, dict)
        or variant_config.get("quality_output_path") != declared_candidate_path
        or not any(
            isinstance(entry, dict) and entry.get("source") == output_source
            for entry in spec.get("argv_template", [])
        )
    ):
        raise SkillError(
            "receipt argv_template must bind variant_config.quality_output_path "
            "to the quality candidate artifact"
        )

    immutable_paths = {
        validation.resolve_artifact(root, artifact.get("path"))
        for artifact in spec.get("workload", {}).get("artifacts", [])
        if isinstance(artifact, dict)
    }
    reserved_paths = {
        root / "raw" / str(spec.get("label")) / "benchmark-command.json",
        root / "quality" / f"{spec.get('label')}.bound.json",
    }
    if candidate_path in immutable_paths or candidate_path in reserved_paths:
        raise SkillError("controlled quality candidate artifact overlaps immutable receipt evidence")
    if candidate_path.exists() and (candidate_path.is_symlink() or not candidate_path.is_file()):
        raise SkillError("controlled quality candidate artifact must be a regular file")
    candidate_path = prepare_receipt_output_path(root, declared_candidate_path)
    quality_control = {
        "path": safe_contract,
        "source_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_size_bytes": len(contract_bytes),
        "contract": contract,
    }
    return candidate_path, candidate, quality_control


def verify_quality_control_snapshot(control: dict[str, Any]) -> None:
    path = control.get("path")
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise SkillError("quality contract changed during measured execution")
    raw = path.read_bytes()
    if (
        len(raw) != control.get("source_size_bytes")
        or hashlib.sha256(raw).hexdigest() != control.get("source_sha256")
    ):
        raise SkillError("quality contract changed during measured execution")


def resolve_candidate_baseline(
    root: Path,
    spec: dict[str, Any],
    value: str | None,
) -> Path | None:
    import validate_benchmarks as validation

    role = spec.get("comparison_role")
    if role != "candidate":
        if value:
            raise SkillError("--baseline-receipt is valid only for a candidate receipt")
        return None
    if not value:
        raise SkillError("candidate receipt mode requires --baseline-receipt")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != ".json":
        raise SkillError("candidate baseline must be an exact root-level JSON receipt filename")
    baseline_path = validation.resolve_artifact(root, value)
    if baseline_path is None or not baseline_path.is_file() or baseline_path.is_symlink():
        raise SkillError("candidate baseline receipt is missing or unsafe")
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"candidate baseline receipt is unreadable: {exc}") from exc
    if (
        not isinstance(baseline, dict)
        or baseline.get("schema_version") != 2
        or baseline.get("label") != baseline_path.stem
        or baseline.get("comparison", {}).get("role") != "baseline"
    ):
        raise SkillError("candidate baseline must be a root-level schema-2 baseline receipt")
    blockers, _, _ = validation.schema2_gate_state(baseline, root)
    blockers = [
        blocker for blocker in blockers
        if blocker not in {"execution-semantics-unattested", "unstable-primary-metric"}
    ]
    if blockers:
        raise SkillError(
            "candidate baseline is not validation-complete: " + ", ".join(sorted(set(blockers)))
        )
    return baseline_path


def validate_receipt_output_layout(
    root: Path,
    receipt_path: Path,
    spec: dict[str, Any],
    quality_output_path: Path,
    quality_control: dict[str, Any],
    baseline_path: Path | None,
) -> None:
    import validate_benchmarks as validation

    expected_receipt_name = f"{spec.get('label')}.json"
    resolved_receipt_path = receipt_path.resolve()
    if (
        resolved_receipt_path.name != expected_receipt_name
        or resolved_receipt_path.parent != root.resolve()
    ):
        raise SkillError(f"receipt output must be the root-level file {expected_receipt_name}")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SkillError("receipt output already exists; refusing to overwrite reviewed evidence")

    contract = quality_control.get("contract")
    canonical_contract = json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    contract_digest = hashlib.sha256(canonical_contract.encode("utf-8")).hexdigest()
    planned_outputs = {
        "receipt": resolved_receipt_path,
        "raw report": root / "raw" / str(spec.get("label")) / "benchmark-command.json",
        "bound quality": root / "quality" / f"{spec.get('label')}.bound.json",
        "stored quality contract": root / "quality" / "inputs" / f"{contract_digest}.json",
        "candidate quality output": quality_output_path,
    }
    resolved_outputs = {label: path.resolve() for label, path in planned_outputs.items()}
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise SkillError("receipt output layout contains colliding control paths")

    immutable_paths: set[Path] = set()
    for artifact in spec.get("workload", {}).get("artifacts", []):
        if isinstance(artifact, dict):
            path = validation.resolve_artifact(root, artifact.get("path"))
            if path is not None:
                immutable_paths.add(path)
    reference = contract.get("reference_artifact") if isinstance(contract, dict) else None
    reference_path = validation.resolve_artifact(
        root,
        reference.get("path") if isinstance(reference, dict) else None,
    )
    if reference_path is not None:
        immutable_paths.add(reference_path)
    if baseline_path is not None:
        immutable_paths.add(baseline_path.resolve())
    control_path = quality_control.get("path")
    if isinstance(control_path, Path):
        resolved_control = control_path.resolve()
        stored_contract = resolved_outputs["stored quality contract"]
        if resolved_control != stored_contract:
            immutable_paths.add(resolved_control)
        elif control_path.read_text(encoding="utf-8") != canonical_contract:
            raise SkillError("quality control path collides with noncanonical stored contract")
    for label, output in resolved_outputs.items():
        if output in immutable_paths:
            raise SkillError(f"{label} collides with immutable benchmark evidence")


def capture_quality_output(
    root: Path,
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    import validate_benchmarks as validation

    if not path.is_file() or path.is_symlink():
        raise SkillError("measured command did not create a regular quality output artifact")
    if path.stat().st_size > validation.MAX_QUALITY_ARTIFACT_BYTES:
        raise SkillError("measured quality output exceeds the controlled artifact limit")
    observed = file_artifact(root, path)
    if observed != expected:
        raise SkillError("measured command quality output does not match the declared artifact")
    return observed


def file_artifact(root: Path, path: Path, *, truncated: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SkillError(f"receipt artifact must stay under {root}: {path}") from exc
    raw = path.read_bytes()
    result: dict[str, Any] = {
        "path": str(relative),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    if truncated is not None:
        result["truncated"] = truncated
    return result


def initialize_attestation_layout(root: Path, spec: dict[str, Any]) -> None:
    label = str(spec.get("label"))
    label_root = root / "attestations" / label
    if label_root.exists() or label_root.is_symlink():
        raise SkillError(
            f"attested receipt evidence already exists for {label}; refusing to overwrite"
        )
    label_root.mkdir(parents=True)


def prepare_attested_invocation(
    root: Path,
    spec: dict[str, Any],
    command: list[str],
    quality_control: dict[str, Any],
    *,
    phase: str,
    run_index: int,
) -> dict[str, Any]:
    import validate_benchmarks as validation

    label = str(spec["label"])
    invocation_prefix = f"attestations/{label}/runs/{phase}-{run_index:03d}"
    contract_source = quality_control.get("path")
    if not isinstance(contract_source, Path):
        raise SkillError("attested invocation is missing the quality-contract snapshot")
    contract_raw = contract_source.read_bytes()
    contract_path = prepare_receipt_output_path(
        root,
        f"{invocation_prefix}/quality-contract.json",
    )
    if contract_path.exists():
        raise SkillError("attested invocation evidence already exists; refusing to overwrite")
    try:
        contract_text = contract_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError("attested quality-contract snapshot is not UTF-8") from exc
    atomic_write_text(contract_path, contract_text)
    contract_artifact = file_artifact(root, contract_path)

    challenge = {
        "schema_version": 1,
        "nonce": secrets.token_hex(32),
        "receipt_label": label,
        "phase": phase,
        "run_index": run_index,
        "command": command,
        "command_sha256": canonical_sha256(command),
        "runner_argv_sha256": canonical_sha256(command[1:]),
        "quality_contract": contract_artifact,
    }
    challenge_snapshot = prepare_receipt_output_path(
        root,
        f"{invocation_prefix}/challenge.json",
    )
    if challenge_snapshot.exists():
        raise SkillError("attested invocation challenge already exists; refusing to overwrite")
    challenge_text = dump_json(challenge)
    atomic_write_text(challenge_snapshot, challenge_text)

    variant = spec.get("variant_config", {})
    current_challenge = prepare_receipt_output_path(
        root,
        str(variant.get("attestation_challenge_path")),
    )
    current_evidence = prepare_receipt_output_path(
        root,
        str(variant.get("attestation_output_path")),
    )
    if current_evidence.exists():
        current_evidence.unlink()
    atomic_write_text(current_challenge, challenge_text)
    challenge_artifact = file_artifact(root, challenge_snapshot)
    if challenge_artifact["size_bytes"] > validation.MAX_ATTESTATION_CHALLENGE_BYTES:
        raise SkillError("attestation challenge exceeds the bounded evidence limit")
    return {
        "prefix": invocation_prefix,
        "challenge": challenge_artifact,
        "current_challenge": current_challenge,
        "current_evidence": current_evidence,
    }


def capture_attested_invocation(
    root: Path,
    context: dict[str, Any],
    quality_output_path: Path,
) -> dict[str, Any]:
    import validate_benchmarks as validation

    evidence_path = context.get("current_evidence")
    if (
        not isinstance(evidence_path, Path)
        or evidence_path.is_symlink()
        or not evidence_path.is_file()
        or evidence_path.stat().st_size > validation.MAX_ATTESTATION_ARTIFACT_BYTES
    ):
        raise SkillError("attested runner did not create bounded regular evidence")
    try:
        evidence_text = evidence_path.read_text(encoding="utf-8")
        evidence_payload = json.loads(evidence_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"attested runner evidence is invalid JSON: {exc}") from exc
    current_challenge = context.get("current_challenge")
    if not isinstance(current_challenge, Path):
        raise SkillError("attested invocation challenge state is missing")
    challenge_descriptor = {
        "sha256": hashlib.sha256(current_challenge.read_bytes()).hexdigest(),
        "size_bytes": current_challenge.stat().st_size,
    }
    if evidence_payload.get("challenge") != challenge_descriptor:
        raise SkillError("attested runner evidence does not bind the parent challenge")

    prefix = str(context["prefix"])
    evidence_snapshot = prepare_receipt_output_path(root, f"{prefix}/evidence.json")
    output_snapshot = prepare_receipt_output_path(root, f"{prefix}/output.json")
    if evidence_snapshot.exists() or output_snapshot.exists():
        raise SkillError("attested invocation evidence already exists; refusing to overwrite")
    atomic_write_text(evidence_snapshot, evidence_text)
    try:
        output_text = quality_output_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillError("attested measured output is not bounded UTF-8") from exc
    if len(output_text.encode("utf-8")) > validation.MAX_ATTESTATION_ARTIFACT_BYTES:
        raise SkillError("attested output exceeds the bounded evidence limit")
    atomic_write_text(output_snapshot, output_text)
    output_artifact = file_artifact(root, output_snapshot)
    evidence_output = evidence_payload.get("output_artifact")
    if (
        not isinstance(evidence_output, dict)
        or evidence_output.get("sha256") != output_artifact["sha256"]
        or evidence_output.get("size_bytes") != output_artifact["size_bytes"]
    ):
        raise SkillError("attested runner evidence does not bind the measured output")
    return {
        "challenge": context["challenge"],
        "evidence": file_artifact(root, evidence_snapshot),
        "output": output_artifact,
    }


def remove_current_attestation_outputs(root: Path, spec: dict[str, Any]) -> None:
    variant = spec.get("variant_config", {})
    for key in ("attestation_challenge_path", "attestation_output_path"):
        path = prepare_receipt_output_path(root, str(variant.get(key)))
        if path.exists():
            path.unlink()


def build_controlled_quality(
    receipt: dict[str, Any],
    receipt_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    import validate_benchmarks as validation

    evaluation = validation.evaluate_controlled_exact_output_quality_contract(receipt_path.parent, contract)
    canonical = json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    stored_contract = prepare_receipt_output_path(
        receipt_path.parent,
        f"quality/inputs/{digest}.json",
    )
    if stored_contract.exists():
        if stored_contract.read_text(encoding="utf-8") != canonical:
            raise SkillError("canonical quality contract path collides with different content")
    else:
        atomic_write_text(stored_contract, canonical)
    input_artifact = file_artifact(receipt_path.parent, stored_contract)
    payload = validation.build_controlled_exact_output_quality_payload(
        receipt,
        input_artifact,
        contract,
        receipt_path.parent,
    )
    if payload.get("status") != evaluation.get("status"):
        raise SkillError("controlled quality evaluation produced an inconsistent payload")
    bound_path = prepare_receipt_output_path(
        receipt_path.parent,
        f"quality/{receipt['label']}.bound.json",
    )
    dump_json(payload, bound_path)
    return {"status": payload["status"], "artifact": file_artifact(receipt_path.parent, bound_path)}


def build_external_receipt(
    *,
    spec: dict[str, Any],
    report: dict[str, Any],
    report_artifact: dict[str, Any],
    receipt_path: Path,
    quality_contract: dict[str, Any],
    baseline_receipt: Path | None,
) -> dict[str, Any]:
    import validate_benchmarks as validation

    workload_descriptor = {
        key: spec["workload"].get(key)
        for key in ("id", "artifacts", "parameters")
    }
    system = report["system"]
    target_descriptor = {
        "hardware": {
            "chip": system.get("cpu_brand") or system.get("processor") or system.get("machine"),
            "model": system.get("mac_model"),
            "memory_bytes": system.get("memory_bytes"),
        },
        "software": {
            "python": system.get("python"),
            "mlx": system.get("mlx_version"),
        },
        "benchmark_system": system,
        "execution_environment_sha256": report.get("execution_environment_sha256"),
        "interpreter": report.get("interpreter"),
    }
    comparison: dict[str, Any] = {
        "role": spec["comparison_role"],
        "primary_metric": "wall_seconds",
    }
    if baseline_receipt is not None and spec["comparison_role"] != "candidate":
        raise SkillError("--baseline-receipt is valid only for a candidate receipt")
    if baseline_receipt is not None:
        try:
            baseline_relative = baseline_receipt.resolve().relative_to(receipt_path.parent.resolve())
        except ValueError as exc:
            raise SkillError("baseline receipt must stay under the receipt root") from exc
        comparison.update({
            "baseline_receipt": str(baseline_relative),
            "baseline_sha256": hashlib.sha256(baseline_receipt.read_bytes()).hexdigest(),
        })
    elif spec["comparison_role"] == "candidate":
        raise SkillError("candidate receipt mode requires --baseline-receipt")

    runs = [
        {
            "run": index,
            "wall_seconds": measured["wall_seconds"],
            "raw_output": report_artifact,
            **(
                {"execution_attestation": measured["execution_attestation"]}
                if "execution_attestation" in measured
                else {}
            ),
        }
        for index, measured in enumerate(report["runs"], start=1)
    ]
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "label": spec["label"],
        "timestamp": report["generated_at"],
        "environment": system,
        "versions": {"mlx": system.get("mlx_version")},
        "command": report["command"],
        "command_display": report["command_display"],
        "cwd": report["cwd"],
        "config_notes": spec["variant_config"],
        "variant_config": spec["variant_config"],
        "enabled_methods": spec["enabled_methods"],
        "models": spec["models"],
        "target": {
            "descriptor": target_descriptor,
            "sha256": validation.canonical_hash(target_descriptor),
        },
        "workload": {
            **workload_descriptor,
            "sha256": validation.canonical_hash(workload_descriptor),
        },
        "comparison": comparison,
        "warmup_runs": report["warmup_count"],
        "measured_runs": report["requested_runs"],
        "timeout_seconds": report["timeout_seconds"],
        "runs": runs,
        "aggregate": validation.recompute_aggregates(runs, validation.EXTERNAL_METRICS),
        "stability": {
            "primary_metric": "wall_seconds",
            "max_cv": 0.10,
            "min_runs": 5,
        },
        "quality": {},
        "rollback_condition": spec["rollback_condition"],
    }
    receipt["runner"] = validation.build_external_runner_descriptor(receipt, spec["argv_template"])
    if receipt["command"] != validation.resolve_external_argv_template(receipt, spec["argv_template"]):
        raise SkillError("resolved receipt command does not match the executed command")
    if not validation.external_command_is_safe(receipt["command"]):
        raise SkillError("external receipt command is unsafe")
    receipt["experiment"] = validation.build_experiment_contract(receipt)
    receipt["quality"] = build_controlled_quality(receipt, receipt_path, quality_contract)
    return receipt


def main() -> int:
    args = parse_args()
    try:
        receipt_spec: dict[str, Any] | None = None
        receipt_path: Path | None = None
        quality_contract_path: Path | None = None
        quality_control: dict[str, Any] | None = None
        quality_output_path: Path | None = None
        expected_quality_output: dict[str, Any] | None = None
        baseline_path: Path | None = None
        execution_environment_sha256: str | None = None
        interpreter_descriptor: dict[str, Any] | None = None
        resolved_interpreter: str | None = None
        attested_mode = False
        serialized_cwd = args.cwd or "."
        if args.receipt_spec:
            if args.command:
                raise SkillError("receipt mode resolves its command from argv_template; do not pass a command after --")
            if not args.output or not args.quality_contract:
                raise SkillError("receipt mode requires --output and --quality-contract")
            receipt_spec = load_receipt_spec(args.receipt_spec)
            requested_receipt_path = Path(args.output).expanduser()
            if requested_receipt_path.is_symlink():
                raise SkillError("receipt output must not be a symlink")
            receipt_path = requested_receipt_path.resolve()
            if args.env:
                raise SkillError(
                    "receipt mode forbids --env because hidden overrides cannot be compared; "
                    "bind controlled variants through argv_template"
                )
            if serialized_cwd != ".":
                raise SkillError("receipt mode requires --cwd . so bound artifact paths are exact")
            if args.runs > MAX_RECEIPT_RUNS or args.warmup > MAX_RECEIPT_WARMUPS:
                raise SkillError(
                    f"receipt mode caps runs at {MAX_RECEIPT_RUNS} and warmups at {MAX_RECEIPT_WARMUPS}"
                )
            if args.stdout_tail > MAX_RECEIPT_TAIL_BYTES or args.stderr_tail > MAX_RECEIPT_TAIL_BYTES:
                raise SkillError(f"receipt mode caps each output tail at {MAX_RECEIPT_TAIL_BYTES} bytes")
            if (
                args.timeout is None
                or not isinstance(args.timeout, (int, float))
                or not math.isfinite(float(args.timeout))
                or float(args.timeout) <= 0
                or float(args.timeout) > MAX_RECEIPT_TIMEOUT_SECONDS
            ):
                raise SkillError(
                    "receipt mode requires a finite positive --timeout no greater than "
                    f"{MAX_RECEIPT_TIMEOUT_SECONDS:g} seconds"
                )
            import validate_benchmarks as validation

            validate_receipt_workload_artifacts(receipt_path.parent, receipt_spec)
            quality_contract_path = Path(args.quality_contract).expanduser()
            quality_output_path, expected_quality_output, quality_control = prepare_receipt_quality_output(
                receipt_path.parent,
                receipt_spec,
                quality_contract_path,
            )
            if quality_output_path in {receipt_path, quality_control["path"]}:
                raise SkillError("controlled quality output overlaps receipt control files")
            baseline_path = resolve_candidate_baseline(
                receipt_path.parent,
                receipt_spec,
                args.baseline_receipt,
            )
            validate_receipt_output_layout(
                receipt_path.parent,
                receipt_path,
                receipt_spec,
                quality_output_path,
                quality_control,
                baseline_path,
            )

            template_receipt = {
                "label": receipt_spec["label"],
                "models": receipt_spec["models"],
                "workload": receipt_spec["workload"],
                "variant_config": receipt_spec["variant_config"],
            }
            args.command = validation.resolve_external_argv_template(
                template_receipt,
                receipt_spec["argv_template"],
            )
            runner_descriptor = validation.build_external_runner_descriptor(
                template_receipt,
                receipt_spec["argv_template"],
            )
            attested_mode = runner_descriptor.get("id") == validation.ATTESTED_RUNNER_ID
            if not validation.external_command_is_safe(args.command):
                raise SkillError("external receipt command is unsafe")
            cwd = str(receipt_path.parent)
            if attested_mode:
                initialize_attestation_layout(receipt_path.parent, receipt_spec)
        else:
            if args.quality_contract or args.baseline_receipt:
                raise SkillError("--quality-contract and --baseline-receipt require --receipt-spec")
            if not args.command:
                raise SkillError("Provide a command after --")
            cwd = str(Path(args.cwd).resolve()) if args.cwd else None
        if args.warmup < 0 or args.runs <= 0:
            raise SkillError("--warmup must be >= 0 and --runs must be > 0")
        if args.stdout_tail < 0 or args.stderr_tail < 0:
            raise SkillError("--stdout-tail and --stderr-tail sizes must be non-negative")
        environment_overrides = parse_environment_overrides(args.env)
        if receipt_spec is not None:
            env = controlled_receipt_environment()
            execution_environment_sha256 = canonical_sha256(env)
            resolved_interpreter, interpreter_descriptor = resolve_receipt_interpreter(
                args.command,
                env,
            )
            execution_command = [
                resolved_interpreter,
                *interpreter_descriptor["flags"],
                *args.command[1:],
            ]
        else:
            env = os.environ.copy()
            env.update(environment_overrides)
            execution_command = args.command
        warmups: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for run_index in range(1, args.warmup + 1):
            attestation_context = (
                prepare_attested_invocation(
                    receipt_path.parent,
                    receipt_spec,
                    args.command,
                    quality_control,
                    phase="warmup",
                    run_index=run_index,
                )
                if attested_mode
                and receipt_path is not None
                and receipt_spec is not None
                and quality_control is not None
                else None
            )
            result = run_once(execution_command, cwd=cwd, env=env, timeout=args.timeout, stdout_tail=args.stdout_tail, stderr_tail=args.stderr_tail)
            result["stdout_tail"] = redact_environment_values(result["stdout_tail"], environment_overrides)
            result["stderr_tail"] = redact_environment_values(result["stderr_tail"], environment_overrides)
            result["phase"] = "warmup"
            if (
                attestation_context is not None
                and quality_output_path is not None
                and result["returncode"] == 0
                and result["timed_out"] is False
            ):
                result["execution_attestation"] = capture_attested_invocation(
                    receipt_path.parent,
                    attestation_context,
                    quality_output_path,
                )
            warmups.append(result)
            if result["returncode"] != 0 or result["timed_out"] is True:
                break
        if not warmups or (
            len(warmups) == args.warmup
            and warmups[-1]["returncode"] == 0
            and warmups[-1]["timed_out"] is False
        ):
            for run_index in range(1, args.runs + 1):
                if quality_output_path is not None and quality_output_path.exists():
                    quality_output_path.unlink()
                attestation_context = (
                    prepare_attested_invocation(
                        receipt_path.parent,
                        receipt_spec,
                        args.command,
                        quality_control,
                        phase="measure",
                        run_index=run_index,
                    )
                    if attested_mode
                    and receipt_path is not None
                    and receipt_spec is not None
                    and quality_control is not None
                    else None
                )
                result = run_once(execution_command, cwd=cwd, env=env, timeout=args.timeout, stdout_tail=args.stdout_tail, stderr_tail=args.stderr_tail)
                result["stdout_tail"] = redact_environment_values(result["stdout_tail"], environment_overrides)
                result["stderr_tail"] = redact_environment_values(result["stderr_tail"], environment_overrides)
                result["phase"] = "measure"
                if (
                    quality_output_path is not None
                    and expected_quality_output is not None
                    and result["returncode"] == 0
                    and result["timed_out"] is False
                ):
                    result["quality_output"] = capture_quality_output(
                        receipt_path.parent,
                        quality_output_path,
                        expected_quality_output,
                    )
                    if attestation_context is not None:
                        result["execution_attestation"] = capture_attested_invocation(
                            receipt_path.parent,
                            attestation_context,
                            quality_output_path,
                        )
                runs.append(result)
                if result["returncode"] != 0 or result["timed_out"] is True:
                    break
        successful_runs = [
            result
            for result in runs
            if result["returncode"] == 0
            and result["timed_out"] is False
            and (
                receipt_spec is None
                or result.get("quality_output") == expected_quality_output
            )
            and (not attested_mode or isinstance(result.get("execution_attestation"), dict))
        ]
        times = [float(result["wall_seconds"]) for result in successful_runs]
        rss = [
            int(result["peak_rss_bytes"])
            for result in successful_runs
            if result.get("peak_rss_bytes") is not None
        ]
        warmups_ok = len(warmups) == args.warmup and all(
            result["returncode"] == 0 and result["timed_out"] is False
            and (not attested_mode or isinstance(result.get("execution_attestation"), dict))
            for result in warmups
        )
        ok = warmups_ok and len(successful_runs) == args.runs
        if receipt_spec is not None and interpreter_descriptor is not None:
            current_interpreter, current_descriptor = resolve_receipt_interpreter(
                args.command,
                env,
            )
            if (
                current_interpreter != resolved_interpreter
                or current_descriptor != interpreter_descriptor
            ):
                raise SkillError("receipt interpreter environment changed during measured execution")
        system = environment_metadata()
        if attested_mode:
            if interpreter_descriptor is None:
                raise SkillError("attested execution is missing its interpreter descriptor")
            system = attested_system_metadata(system, interpreter_descriptor)
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "command_display": shlex.join(args.command),
            "cwd": serialized_cwd,
            "environment_overrides": {key: "[REDACTED]" for key in environment_overrides},
            "system": system,
            "warmup_count": args.warmup,
            "requested_runs": args.runs,
            "timeout_seconds": args.timeout,
            "ok": ok,
            "summary": {
                "successful_runs": len(times),
                "wall_seconds_min": min(times) if times else None,
                "wall_seconds_median": statistics.median(times) if times else None,
                "wall_seconds_mean": statistics.mean(times) if times else None,
                "wall_seconds_p95": percentile(times, 0.95) if times else None,
                "wall_seconds_max": max(times) if times else None,
                "peak_rss_bytes_max": max(rss) if rss else None,
            },
            "warmups": warmups,
            "runs": runs,
            "notes": [
                "This harness measures process wall time; model-specific code must also emit TTFT, tokens/s, RTF, and quality metrics.",
                "Peak RSS is available only when psutil is installed and may not equal MLX active/cache memory.",
                "Cold compile/load and warm steady-state should be reported separately.",
            ],
        }
        if execution_environment_sha256 is not None:
            report["execution_environment_sha256"] = execution_environment_sha256
        if interpreter_descriptor is not None:
            report["interpreter"] = interpreter_descriptor
        if receipt_spec is not None and receipt_path is not None:
            if quality_control is None:
                raise SkillError("receipt mode quality control snapshot is missing")
            verify_quality_control_snapshot(quality_control)
            if attested_mode:
                remove_current_attestation_outputs(receipt_path.parent, receipt_spec)
            raw_report_path = prepare_receipt_output_path(
                receipt_path.parent,
                f"raw/{receipt_spec['label']}/benchmark-command.json",
            )
            dump_json(report, raw_report_path)
            if raw_report_path.stat().st_size > MAX_RECEIPT_REPORT_BYTES:
                raise SkillError(f"receipt mode raw report exceeds {MAX_RECEIPT_REPORT_BYTES} bytes")
            if not ok:
                return 1
            raw_artifact = file_artifact(receipt_path.parent, raw_report_path, truncated=False)
            receipt = build_external_receipt(
                spec=receipt_spec,
                report=report,
                report_artifact=raw_artifact,
                receipt_path=receipt_path,
                quality_contract=quality_control["contract"],
                baseline_receipt=baseline_path,
            )
            dump_json(receipt, receipt_path)
            print(f"wrote benchmark receipt: {receipt_path}")
        else:
            text = dump_json(report, args.output)
            if args.output is None:
                sys.stdout.write(text)
        return 0 if ok else 1
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
