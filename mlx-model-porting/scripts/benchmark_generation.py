#!/usr/bin/env python3
"""Benchmark mlx_lm generate-style commands and write honest receipt JSON."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import atomic_write_text, dump_json, redact_secret_text, run_process_capture
from benchmark_command import environment_metadata
from validate_benchmarks import (
    DECLARATIVE_QUALITY_VALIDATOR,
    EXACT_OUTPUT_QUALITY_VALIDATOR,
    build_assessment_report,
    build_controlled_exact_output_quality_payload,
    build_declarative_quality_payload,
    build_experiment_contract,
    build_receipts_index,
    build_runner_descriptor,
    canonical_hash,
    evaluate_controlled_exact_output_quality_contract,
    evaluate_declarative_quality_contract,
    file_sha256,
    render_benchmark_report,
)


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
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-run timeout in seconds")
    parser.add_argument("--baseline-receipt", help="Optional baseline receipt JSON for median speed ratios")
    parser.add_argument("--output", help="Receipt JSON path")
    parser.add_argument("--config-note", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--target-model", help="Pinned target model id for the schema-2 experiment contract")
    parser.add_argument("--target-revision", help="Immutable 40-64 hex target revision")
    parser.add_argument("--lineage-id", help="Controlled model-lineage identifier shared with compatible baselines")
    parser.add_argument("--source-id", help="Pinned source-model identity; defaults to --target-model")
    parser.add_argument("--source-revision", help="Pinned source revision; defaults to --target-revision")
    parser.add_argument("--draft-model", help="Optional pinned draft model id")
    parser.add_argument("--draft-revision", help="Immutable 40-64 hex draft revision")
    parser.add_argument("--draft-lineage-id", help="Optional draft lineage identifier")
    parser.add_argument("--draft-source-id", help="Pinned draft source identity; defaults to --draft-model")
    parser.add_argument("--draft-source-revision", help="Pinned draft source revision; defaults to --draft-revision")
    parser.add_argument("--workload-id", help="Stable workload identifier")
    parser.add_argument(
        "--workload-artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Checked-in workload input and role; may be repeated",
    )
    parser.add_argument("--workload-params-json", help="Canonical JSON object of generation/workload parameters")
    parser.add_argument(
        "--quality-artifact",
        help="Legacy quality attestation artifact; recorded for observation but never promotion-ready",
    )
    parser.add_argument(
        "--quality-evaluator",
        help="Legacy Python evaluator recorded without execution; never promotion-ready",
    )
    parser.add_argument(
        "--quality-contract",
        help=(
            "Quality JSON evaluated by the harness and independently recomputed by validation. "
            "Schema 2 exact-output-parity contracts can promote; schema 1 declarative contracts "
            "are retained as observations only."
        ),
    )
    parser.add_argument("--rollback-condition", help="Explicit rollback condition for this experiment")
    parser.add_argument("--enabled-method", action="append", default=[], help="Ordered optimization method id; may be repeated")
    parser.add_argument("--comparison-role", choices=("baseline", "candidate", "observation"))
    parser.add_argument("--primary-metric", choices=("prompt_tps", "generation_tps", "peak_memory_gb", "ttft_proxy_s"), default="generation_tps")
    parser.add_argument("--max-cv", type=float, default=0.10, help="Maximum population coefficient of variation")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    quality_inputs = [args.quality_artifact, args.quality_evaluator, args.quality_contract]
    if sum(value is not None for value in quality_inputs) > 1:
        parser.error("--quality-artifact, --quality-evaluator, and --quality-contract are mutually exclusive")
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


def parse_workload_params(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"--workload-params-json must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ReceiptError("--workload-params-json must be a non-empty object")
    return parsed


def portable_artifact(receipt_root: Path, value: str, role: str | None = None) -> dict[str, Any]:
    path = Path(value).expanduser()
    resolved = path.resolve()
    rendered = relative_or_absolute(receipt_root, resolved)
    artifact: dict[str, Any] = {"path": rendered}
    if role:
        artifact["role"] = role
    if resolved.is_file():
        artifact["sha256"] = file_sha256(resolved)
        artifact["size_bytes"] = resolved.stat().st_size
    return artifact


def parse_workload_artifacts(values: list[str], receipt_root: Path) -> list[dict[str, Any]]:
    artifacts = []
    for value in values:
        if "=" not in value:
            raise ReceiptError(f"--workload-artifact must be ROLE=PATH, got {value!r}")
        role, path = value.split("=", 1)
        if not role or not path:
            raise ReceiptError("--workload-artifact requires non-empty ROLE and PATH")
        artifacts.append(portable_artifact(receipt_root, path, role))
    return artifacts


def record_legacy_quality_evaluator(
    receipt_path: Path,
    evaluator_source: Path,
) -> dict[str, Any]:
    try:
        evaluator_text = evaluator_source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReceiptError(f"could not read --quality-evaluator {evaluator_source}: {exc}") from exc
    if evaluator_source.suffix != ".py":
        raise ReceiptError("--quality-evaluator must be a Python source file")
    evaluator_digest = hashlib.sha256(evaluator_text.encode("utf-8")).hexdigest()
    evaluator_path = receipt_path.parent / "quality" / "legacy-evaluators" / f"{evaluator_digest}.py"
    atomic_write_text(evaluator_path, evaluator_text)
    return {
        "status": "unverified",
        "artifact": portable_artifact(receipt_path.parent, str(evaluator_path)),
        "attestation_only": True,
        "legacy_kind": "python-evaluator-not-executed",
    }


def build_declarative_quality(
    receipt_path: Path,
    receipt: dict[str, Any],
    contract_source: Path,
) -> dict[str, Any]:
    try:
        contract = json.loads(contract_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"could not read --quality-contract {contract_source}: {exc}") from exc
    controlled_exact_output = (
        isinstance(contract, dict)
        and contract.get("schema_version") == 2
        and contract.get("validator") == EXACT_OUTPUT_QUALITY_VALIDATOR
    )
    try:
        if controlled_exact_output:
            evaluation = evaluate_controlled_exact_output_quality_contract(
                receipt_path.parent,
                contract,
            )
        else:
            evaluation = evaluate_declarative_quality_contract(receipt, contract)
    except Exception as exc:  # SkillError is intentionally rendered as a CLI contract failure.
        raise ReceiptError(f"invalid --quality-contract {contract_source}: {exc}") from exc
    canonical_contract = json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    contract_digest = hashlib.sha256(canonical_contract.encode("utf-8")).hexdigest()
    contract_path = receipt_path.parent / "quality" / "inputs" / f"{contract_digest}.json"
    atomic_write_text(contract_path, canonical_contract)
    input_artifact = portable_artifact(receipt_path.parent, str(contract_path))
    if controlled_exact_output:
        bound_payload = build_controlled_exact_output_quality_payload(
            receipt,
            input_artifact,
            contract,
            receipt_path.parent,
        )
        expected_validator = EXACT_OUTPUT_QUALITY_VALIDATOR
    else:
        bound_payload = build_declarative_quality_payload(receipt, input_artifact, contract)
        expected_validator = DECLARATIVE_QUALITY_VALIDATOR
    if bound_payload.get("validator") != expected_validator or bound_payload.get("status") != evaluation.get("status"):
        raise ReceiptError("declarative quality evaluation produced an inconsistent payload")
    bound_path = receipt_path.parent / "quality" / f"{receipt['label']}.bound.json"
    dump_json(bound_payload, bound_path)
    return {
        "status": bound_payload["status"],
        "artifact": portable_artifact(receipt_path.parent, str(bound_path)),
    }


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


def run_command(
    command: list[str],
    phase: str,
    index: int,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    result, timed_out = run_process_capture(command, timeout=timeout)
    if result.returncode != 0:
        timeout_note = f" after timing out at {timeout:g}s" if timed_out else ""
        print(f"{phase} run {index} failed with exit code {result.returncode}{timeout_note}", file=sys.stderr)
        print("stdout:", file=sys.stderr)
        print(redact_secret_text(result.stdout), file=sys.stderr)
        print("stderr:", file=sys.stderr)
        print(redact_secret_text(result.stderr), file=sys.stderr)
    return result


def write_raw_output_artifact(
    receipt_path: Path,
    label: str,
    run_index: int,
    result: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    raw_path = receipt_path.parent / "raw" / label / f"run-{run_index:03d}.txt"
    text = redact_secret_text(
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )
    atomic_write_text(raw_path, text)
    encoded = text.encode("utf-8")
    return {
        "path": str(raw_path.relative_to(receipt_path.parent)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "truncated": "...[truncated " in text,
    }


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
    assessment_path = receipt_path.parent / "receipt_assessments.json"
    index_path = receipt_path.parent / "receipts_index.json"
    report_path = receipt_path.parent.parent / "BENCHMARK_REPORT.md"
    report = build_assessment_report(receipt_path.parent)
    index = build_receipts_index(report, receipt_path.parent)
    dump_json(report, assessment_path)
    dump_json(index, index_path)
    atomic_write_text(report_path, render_benchmark_report(report))


def build_receipt(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if not LABEL_RE.fullmatch(args.label):
        raise ReceiptError("--label must be a slug containing only letters, numbers, dot, underscore, or dash")
    if args.runs <= 0 or args.warmup < 0:
        raise ReceiptError("--runs must be > 0 and --warmup must be >= 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ReceiptError("--timeout must be a finite number > 0")
    if args.max_cv <= 0:
        raise ReceiptError("--max-cv must be > 0")
    if not args.command:
        raise ReceiptError("Provide a command after --")

    receipt_path = receipt_path_for(args.label, args.output)
    config_notes = parse_config_notes(args.config_note)
    workload_parameters = parse_workload_params(args.workload_params_json)
    workload_artifacts = parse_workload_artifacts(args.workload_artifact, receipt_path.parent)
    for index in range(1, args.warmup + 1):
        result = run_command(args.command, "warmup", index, args.timeout)
        if result.returncode != 0:
            raise SystemExit(1)

    measured_runs: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        result = run_command(args.command, "measured", index, args.timeout)
        if result.returncode != 0:
            raise SystemExit(1)
        metrics = parse_metrics(f"{result.stdout}\n{result.stderr}", index)
        metrics["raw_output"] = write_raw_output_artifact(receipt_path, args.label, index, result)
        measured_runs.append(metrics)

    aggregate = aggregate_metrics(measured_runs)
    timestamp = datetime.now(timezone.utc).isoformat()
    environment = environment_metadata()
    versions = {
        "mlx": package_version("mlx", "mlx"),
        "mlx_lm": package_version("mlx_lm", "mlx-lm"),
    }
    target_descriptor = {
        "hardware": {
            "chip": chip_label(environment),
            "model": environment.get("mac_model"),
            "memory_bytes": environment.get("memory_bytes"),
        },
        "software": {
            "python": environment.get("python"),
            "platform": environment.get("platform"),
            "macos": environment.get("macos_version"),
            **versions,
        },
    }
    models: dict[str, Any] = {
        "target": {
            "id": args.target_model,
            "revision": args.target_revision,
            "lineage_id": args.lineage_id,
            "source_id": args.source_id or args.target_model,
            "source_revision": args.source_revision or args.target_revision,
        }
    }
    if args.draft_model or args.draft_revision or args.draft_lineage_id or args.draft_source_id or args.draft_source_revision:
        models["draft"] = {
            "id": args.draft_model,
            "revision": args.draft_revision,
            "lineage_id": args.draft_lineage_id,
            "source_id": args.draft_source_id or args.draft_model,
            "source_revision": args.draft_source_revision or args.draft_revision,
        }
    workload: dict[str, Any] = {}
    if args.workload_id or workload_artifacts or workload_parameters:
        workload_descriptor = {
            "id": args.workload_id,
            "artifacts": workload_artifacts,
            "parameters": workload_parameters,
        }
        workload = {**workload_descriptor, "sha256": canonical_hash(workload_descriptor)}
    quality: dict[str, Any] = {}
    if args.quality_artifact:
        quality_path = Path(args.quality_artifact).expanduser().resolve()
        try:
            quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReceiptError(f"could not read --quality-artifact {quality_path}: {exc}") from exc
        if not isinstance(quality_payload, dict):
            raise ReceiptError("--quality-artifact must contain a JSON object")
        quality = {
            "status": quality_payload.get("status"),
            "artifact": portable_artifact(receipt_path.parent, str(quality_path)),
            "attestation_only": True,
        }
    comparison_role = args.comparison_role or ("candidate" if args.baseline_receipt else "baseline")
    comparison: dict[str, Any] = {
        "role": comparison_role,
        "primary_metric": args.primary_metric,
    }
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "label": args.label,
        "timestamp": timestamp,
        "environment": environment,
        "versions": versions,
        "command": args.command,
        "command_display": shlex.join(args.command),
        "config_notes": config_notes,
        "variant_config": config_notes,
        "enabled_methods": list(args.enabled_method),
        "models": models,
        "target": {"descriptor": target_descriptor, "sha256": canonical_hash(target_descriptor)},
        "workload": workload,
        "comparison": comparison,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "timeout_seconds": args.timeout,
        "runs": measured_runs,
        "aggregate": aggregate,
        "stability": {
            "primary_metric": args.primary_metric,
            "max_cv": args.max_cv,
            "min_runs": 5,
        },
        "quality": quality,
        "rollback_condition": args.rollback_condition,
        "ttft_proxy": {
            "metric": "ttft_proxy_s",
            "median_s": aggregate["ttft_proxy_s"]["median"],
            "proxy_note": "Computed as prompt_tokens / prompt_tps from the mlx_lm Prompt line; this is not instrumented first-token latency.",
        },
    }
    if args.baseline_receipt:
        baseline_path = Path(args.baseline_receipt).expanduser().resolve()
        receipt["speedup_vs_baseline"] = compute_speedup(receipt, baseline_path)
        receipt["speedup_vs_baseline"]["baseline_file"] = relative_or_absolute(receipt_path.parent, baseline_path)
        comparison.update({
            "baseline_receipt": relative_or_absolute(receipt_path.parent, baseline_path),
            "baseline_sha256": file_sha256(baseline_path),
        })
    receipt["runner"] = build_runner_descriptor(args.command)
    receipt["experiment"] = build_experiment_contract(receipt)
    if args.quality_contract:
        contract_path = Path(args.quality_contract).expanduser().resolve()
        receipt["quality"] = build_declarative_quality(receipt_path, receipt, contract_path)
    elif args.quality_evaluator:
        evaluator_path = Path(args.quality_evaluator).expanduser().resolve()
        receipt["quality"] = record_legacy_quality_evaluator(receipt_path, evaluator_path)
    return receipt_path, receipt


def main() -> int:
    try:
        path, receipt = build_receipt(parse_args())
        dump_json(receipt, path)
        update_index(path, receipt)
        print(f"wrote benchmark receipt: {path}")
        return 0
    except ReceiptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
