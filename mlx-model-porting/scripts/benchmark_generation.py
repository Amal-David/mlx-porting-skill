#!/usr/bin/env python3
"""Benchmark mlx_lm generate-style commands and write honest receipt JSON."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_command import environment_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "mlx-model-porting" / "assets" / "benchmarks"
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PROMPT_RE = re.compile(r"Prompt:\s*(\d+)\s+tokens,\s*([0-9]+(?:\.[0-9]+)?)\s+tokens-per-sec")
GENERATION_RE = re.compile(r"Generation:\s*(\d+)\s+tokens,\s*([0-9]+(?:\.[0-9]+)?)\s+tokens-per-sec")
PEAK_MEMORY_RE = re.compile(r"Peak memory:\s*([0-9]+(?:\.[0-9]+)?)\s+GB")
EXPECTED_PATTERNS = [
    "Prompt: <tokens> tokens, <tokens-per-sec> tokens-per-sec",
    "Generation: <tokens> tokens, <tokens-per-sec> tokens-per-sec",
    "Peak memory: <GB> GB",
]


class ReceiptError(RuntimeError):
    """Raised for expected harness failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mlx_lm generate-style commands and write token-throughput benchmark receipts"
    )
    parser.add_argument("--label", required=True, help="Receipt id slug")
    parser.add_argument("--runs", type=int, default=5, help="Measured run count")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup run count discarded from the receipt")
    parser.add_argument("--baseline-receipt", help="Optional baseline receipt JSON for median speed ratios")
    parser.add_argument("--output", help="Receipt JSON path")
    parser.add_argument("--config-note", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def parse_config_notes(values: list[str]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ReceiptError(f"--config-note must be KEY=VALUE, got {value!r}")
        key, note_value = value.split("=", 1)
        if not key:
            raise ReceiptError("--config-note key cannot be empty")
        if key in notes:
            raise ReceiptError(f"duplicate --config-note key: {key}")
        notes[key] = note_value
    return notes


def package_version(module_name: str, distribution_name: str | None = None) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        return importlib.metadata.version(distribution_name or module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_metrics(output: str, run_index: int) -> dict[str, Any]:
    prompt = PROMPT_RE.search(output)
    generation = GENERATION_RE.search(output)
    peak_memory = PEAK_MEMORY_RE.search(output)
    missing: list[str] = []
    if prompt is None:
        missing.append(EXPECTED_PATTERNS[0])
    if generation is None:
        missing.append(EXPECTED_PATTERNS[1])
    if peak_memory is None:
        missing.append(EXPECTED_PATTERNS[2])
    if missing:
        tail = output[-4000:] if output else "<empty output>"
        raise ReceiptError(
            "missing benchmark output pattern(s) for measured run "
            f"{run_index}: expected {missing}; actual output tail:\n{tail}"
        )

    assert prompt is not None and generation is not None and peak_memory is not None
    prompt_tokens = int(prompt.group(1))
    prompt_tps = float(prompt.group(2))
    generation_tokens = int(generation.group(1))
    generation_tps = float(generation.group(2))
    peak_memory_gb = float(peak_memory.group(1))
    if prompt_tps <= 0:
        raise ReceiptError(f"prompt tokens-per-sec must be positive for measured run {run_index}")
    return {
        "run": run_index,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        "peak_memory_gb": peak_memory_gb,
        "ttft_proxy_s": prompt_tokens / prompt_tps,
    }


def run_command(command: list[str], phase: str, index: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{phase} run {index} failed with exit code {result.returncode}", file=sys.stderr)
        print("stdout:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("stderr:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return result


def summarize(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"median": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    metric_names = [
        "prompt_tokens",
        "prompt_tps",
        "generation_tokens",
        "generation_tps",
        "peak_memory_gb",
        "ttft_proxy_s",
    ]
    return {name: summarize([run[name] for run in runs]) for name in metric_names}


def aggregate_median(receipt: dict[str, Any], metric: str) -> float:
    try:
        value = receipt["aggregate"][metric]["median"]
    except (KeyError, TypeError) as exc:
        raise ReceiptError(f"baseline receipt lacks aggregate median for {metric}") from exc
    if not isinstance(value, (int, float)) or value <= 0:
        raise ReceiptError(f"baseline receipt median for {metric} must be a positive number")
    return float(value)


def compute_speedup(candidate: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"could not read baseline receipt {baseline_path}: {exc}") from exc

    candidate_aggregate = candidate["aggregate"]
    candidate_decode = float(candidate_aggregate["generation_tps"]["median"])
    candidate_prefill = float(candidate_aggregate["prompt_tps"]["median"])
    candidate_ttft = float(candidate_aggregate["ttft_proxy_s"]["median"])
    if candidate_decode <= 0 or candidate_prefill <= 0 or candidate_ttft <= 0:
        raise ReceiptError("candidate aggregate medians must be positive before comparing to baseline")

    baseline_decode = aggregate_median(baseline, "generation_tps")
    baseline_prefill = aggregate_median(baseline, "prompt_tps")
    baseline_ttft = aggregate_median(baseline, "ttft_proxy_s")
    return {
        "baseline_label": baseline.get("label"),
        "baseline_file": str(baseline_path),
        "basis": "Ratios of aggregate medians; ttft_proxy uses inverse ratio because lower is better.",
        "ratios": {
            "decode_tps": candidate_decode / baseline_decode,
            "prefill_tps": candidate_prefill / baseline_prefill,
            "ttft_proxy_inverse": baseline_ttft / candidate_ttft,
        },
    }


def receipt_path_for(label: str, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return DEFAULT_BENCHMARK_ROOT / f"{label}.json"


def relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def chip_label(environment: dict[str, Any]) -> str | None:
    for key in ("cpu_brand", "mac_model", "processor", "machine"):
        value = environment.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def update_index(receipt_path: Path, receipt: dict[str, Any]) -> None:
    index_path = receipt_path.parent / "receipts_index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReceiptError(f"could not parse benchmark receipt index {index_path}: {exc}") from exc
    else:
        index = {"receipts": []}
    receipts = index.get("receipts")
    if not isinstance(receipts, list):
        raise ReceiptError(f"benchmark receipt index {index_path} must contain a receipts array")

    entry = {
        "label": receipt["label"],
        "file": relative_or_absolute(index_path.parent, receipt_path),
        "date": receipt["timestamp"],
        "chip": chip_label(receipt["environment"]),
        "command_summary": shlex.join(receipt["command"]),
        "config_notes": receipt["config_notes"],
    }
    index["receipts"] = [item for item in receipts if not isinstance(item, dict) or item.get("label") != receipt["label"]]
    index["receipts"].append(entry)
    index["receipts"].sort(key=lambda item: str(item.get("label", "")))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_receipt(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if not LABEL_RE.fullmatch(args.label):
        raise ReceiptError("--label must be a slug containing only letters, numbers, dot, underscore, or dash")
    if args.runs <= 0 or args.warmup < 0:
        raise ReceiptError("--runs must be > 0 and --warmup must be >= 0")
    if not args.command:
        raise ReceiptError("Provide a command after --")

    config_notes = parse_config_notes(args.config_note)
    for index in range(1, args.warmup + 1):
        result = run_command(args.command, "warmup", index)
        if result.returncode != 0:
            raise SystemExit(1)

    measured_runs: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        result = run_command(args.command, "measured", index)
        if result.returncode != 0:
            raise SystemExit(1)
        measured_runs.append(parse_metrics(f"{result.stdout}\n{result.stderr}", index))

    aggregate = aggregate_metrics(measured_runs)
    timestamp = datetime.now(timezone.utc).isoformat()
    environment = environment_metadata()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "timestamp": timestamp,
        "environment": environment,
        "versions": {
            "mlx": package_version("mlx", "mlx"),
            "mlx_lm": package_version("mlx_lm", "mlx-lm"),
        },
        "command": args.command,
        "command_display": shlex.join(args.command),
        "config_notes": config_notes,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "runs": measured_runs,
        "aggregate": aggregate,
        "ttft_proxy": {
            "metric": "ttft_proxy_s",
            "median_s": aggregate["ttft_proxy_s"]["median"],
            "proxy_note": "Computed as prompt_tokens / prompt_tps from the mlx_lm Prompt line; this is not instrumented first-token latency.",
        },
    }
    if args.baseline_receipt:
        receipt["speedup_vs_baseline"] = compute_speedup(receipt, Path(args.baseline_receipt).expanduser().resolve())
    return receipt_path_for(args.label, args.output), receipt


def main() -> int:
    try:
        path, receipt = build_receipt(parse_args())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        update_index(path, receipt)
        print(f"wrote benchmark receipt: {path}")
        return 0
    except ReceiptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
