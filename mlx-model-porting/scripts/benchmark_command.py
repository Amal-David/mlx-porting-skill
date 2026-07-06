#!/usr/bin/env python3
"""Run a command repeatedly and capture reproducible wall-time/RSS/environment data."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import signal
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json


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
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=3)
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
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


def parse_env(values: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    for value in values:
        if "=" not in value:
            raise SkillError(f"--env must be KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        if not key:
            raise SkillError("--env key cannot be empty")
        env[key] = val
    return env


def kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    proc.kill()


def run_once(command: list[str], *, cwd: str | None, env: dict[str, str], timeout: float | None, stdout_tail: int, stderr_tail: int) -> dict[str, Any]:
    rss_peak: int | None = None
    stop = threading.Event()
    proc: subprocess.Popen[str] | None = None

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
    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree(proc)
        stdout, stderr = proc.communicate()
    finally:
        stop.set()
        thread.join(timeout=1)
    elapsed = time.perf_counter() - start
    return {
        "wall_seconds": elapsed,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "peak_rss_bytes": rss_peak,
        "stdout_tail": stdout[-stdout_tail:] if stdout_tail else "",
        "stderr_tail": stderr[-stderr_tail:] if stderr_tail else "",
    }


def main() -> int:
    args = parse_args()
    try:
        if not args.command:
            raise SkillError("Provide a command after --")
        if args.warmup < 0 or args.runs <= 0:
            raise SkillError("--warmup must be >= 0 and --runs must be > 0")
        env = parse_env(args.env)
        cwd = str(Path(args.cwd).resolve()) if args.cwd else None
        warmups: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for _ in range(args.warmup):
            result = run_once(args.command, cwd=cwd, env=env, timeout=args.timeout, stdout_tail=args.stdout_tail, stderr_tail=args.stderr_tail)
            result["phase"] = "warmup"
            warmups.append(result)
            if result["returncode"] != 0:
                break
        if not warmups or warmups[-1]["returncode"] == 0:
            for _ in range(args.runs):
                result = run_once(args.command, cwd=cwd, env=env, timeout=args.timeout, stdout_tail=args.stdout_tail, stderr_tail=args.stderr_tail)
                result["phase"] = "measure"
                runs.append(result)
                if result["returncode"] != 0:
                    break
        times = [float(r["wall_seconds"]) for r in runs if r["returncode"] == 0]
        rss = [int(r["peak_rss_bytes"]) for r in runs if r.get("peak_rss_bytes") is not None]
        ok = len(times) == args.runs
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "command_display": shlex.join(args.command),
            "cwd": cwd or os.getcwd(),
            "environment_overrides": args.env,
            "system": environment_metadata(),
            "warmup_count": args.warmup,
            "requested_runs": args.runs,
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
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        return 0 if ok else 1
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
