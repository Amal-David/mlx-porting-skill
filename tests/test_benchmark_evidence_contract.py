from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"
BENCHMARKS = SKILL / "assets" / "benchmarks"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_benchmarks  # noqa: E402
import benchmark_generation  # noqa: E402
import benchmark_command  # noqa: E402
import generate_claim_catalog  # noqa: E402
import _common  # noqa: E402


HISTORICAL_HASHES = {
    "kv-baseline-8k.json": "c6c3b60be9b195a933fbeb1d89499a24e968cdaa9601593f4e7dac6fa1a991c1",
    "kv-4bit-8k.json": "1dc4551ff2e27d1183a159fccfdb7be4e498f03c9e9d92a917cc8f171f34a2ca",
    "pcache-cold.json": "2edbbedc36a86dda78ab6dc18d5c973b7cd89db43e1e8c819877f6e72069eb3c",
    "pcache-warm.json": "47cfeba0fd66d45cfaa9c4343bad6573695304c00bc19ea5fc784202bf321c37",
    "quant-baseline-bf16.json": "d299da6bdadcd525bbd10713673c2e6bc2b9a6f6d66f7707ee54cb859b53ed6c",
    "quant-4bit.json": "ae6da36683617bff20cda370b1707c4b46b46874ad5ac41992d6046a8db77175",
    "spec-baseline.json": "81d21b1e1849418f97b59e622fdb91fc50f335f706612eac973dbceb62566a9f",
    "spec-draft-k2.json": "a16c7d6bba98fc0725ba78a8ef465d52cdec8a210eecc437a8f56dbd164220f3",
    "spec-draft-k3.json": "065cf9b998d60aca0b31a7403d8f96f3fde918fa79b86fca7427166049b1f21c",
    "spec-draft-k4.json": "a1ad234c72d15f97df7cc43920fd52441dd2141a126d984fa42d3e044452c6a4",
    "stack-measured-together.json": "e21e6eb137d7521918b13f76ccfd7a823013d4f3b3927a9586ca82f2c1a2fe70",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def artifact(root: Path, relative: str, content: str) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    raw = content.encode("utf-8")
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def complete_receipt(root: Path, label: str, *, role: str) -> dict[str, object]:
    prompt = artifact(root, "inputs/prompt.txt", "Write a stable answer.\n")
    target_descriptor = {
        "hardware": {"chip": "Apple Test", "model": "MacTest,1", "memory_bytes": 16_000_000_000},
        "software": {"python": "3.14.4", "mlx": "0.30.4", "mlx_lm": "0.31.3"},
    }
    workload_descriptor = {
        "id": "fixture-workload",
        "artifacts": [{"role": "prompt", **prompt}],
        "parameters": {"max_tokens": 64, "temperature": 0.0, "seed": 7},
    }
    quality_path = f"quality/{label}.json"
    runs = []
    generation_values = (60.0, 60.5, 59.5, 60.2, 59.8) if role == "candidate" else (50.0, 50.5, 49.5, 50.2, 49.8)
    for index, generation_tps in enumerate(generation_values, start=1):
        raw = artifact(
            root,
            f"raw/{label}/run-{index:03d}.txt",
            "Prompt: 32 tokens, 100.0 tokens-per-sec\n"
            f"Generation: 64 tokens, {generation_tps} tokens-per-sec\n"
            "Peak memory: 1.0 GB\n",
        )
        runs.append({
            "run": index,
            "prompt_tokens": 32,
            "prompt_tps": 100.0,
            "generation_tokens": 64,
            "generation_tps": generation_tps,
            "peak_memory_gb": 1.0,
            "ttft_proxy_s": 0.32,
            "raw_output": {**raw, "truncated": False},
        })
    receipt: dict[str, object] = {
        "schema_version": 2,
        "label": label,
        "timestamp": "2026-07-09T00:00:00+00:00",
        "environment": {"platform": "macOS-test", "macos_version": "26.0"},
        "command": [
            "python3",
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            "example/model",
            "--prompt",
            "Write a stable answer.\n",
            "--max-tokens",
            "64",
            "--temp",
            "0.0",
            "--seed",
            "7",
        ],
        "runner": {},
        "enabled_methods": ["fixture-optimization"] if role == "candidate" else [],
        "variant_config": {"fixture_mode": role},
        "models": {
            "target": {
                "id": "example/model",
                "revision": "a" * 40,
                "lineage_id": "example-controlled-lineage",
                "source_id": "example/source-model",
                "source_revision": "b" * 40,
            }
        },
        "target": {"descriptor": target_descriptor, "sha256": validate_benchmarks.canonical_hash(target_descriptor)},
        "workload": {**workload_descriptor, "sha256": validate_benchmarks.canonical_hash(workload_descriptor)},
        "comparison": {"role": role, "primary_metric": "generation_tps"},
        "warmup_runs": 1,
        "measured_runs": 5,
        "timeout_seconds": 600.0,
        "runs": runs,
        "aggregate": validate_benchmarks.recompute_aggregates(runs),
        "stability": {"primary_metric": "generation_tps", "max_cv": 0.05, "min_runs": 5},
        "quality": {"status": "pass", "artifact": {"path": quality_path}},
        "rollback_condition": "Rollback when quality fails or decode throughput is not above baseline noise.",
    }
    attach_schema_contracts(root, receipt)
    return receipt


def attach_schema_contracts(root: Path, receipt: dict[str, object]) -> None:
    command = receipt["command"]
    assert isinstance(command, list)
    runner = validate_benchmarks.build_runner_descriptor(command)
    receipt["runner"] = runner
    models = receipt["models"]
    target = receipt["target"]
    workload = receipt["workload"]
    comparison = receipt["comparison"]
    runs = receipt["runs"]
    assert isinstance(models, dict) and isinstance(models["target"], dict)
    assert isinstance(target, dict) and isinstance(workload, dict) and isinstance(comparison, dict)
    assert isinstance(runs, list)
    receipt["experiment"] = validate_benchmarks.build_experiment_contract(receipt)
    reference_artifact = artifact(
        root,
        "quality/artifacts/exact-output-reference.txt",
        "stable expected output\n",
    )
    candidate_artifact = artifact(
        root,
        f"quality/artifacts/{receipt['label']}-exact-output.txt",
        "stable expected output\n",
    )
    quality_contract = {
        "schema_version": 2,
        "validator": validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
        "metric": "exact-output-parity",
        "reference_artifact": reference_artifact,
        "candidate_artifact": candidate_artifact,
    }
    quality_contract_text = json.dumps(quality_contract, indent=2, sort_keys=True) + "\n"
    quality_contract_sha256 = hashlib.sha256(quality_contract_text.encode("utf-8")).hexdigest()
    quality_input = artifact(
        root,
        f"quality/inputs/{quality_contract_sha256}.json",
        quality_contract_text,
    )
    quality_value = validate_benchmarks.build_controlled_exact_output_quality_payload(
        receipt,
        quality_input,
        quality_contract,
        root,
    )
    quality = receipt["quality"]
    assert isinstance(quality, dict) and isinstance(quality["artifact"], dict)
    quality_path = root / str(quality["artifact"]["path"])
    write_json(quality_path, quality_value)
    quality["artifact"]["sha256"] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
    quality["artifact"]["size_bytes"] = quality_path.stat().st_size


def set_generation_values(root: Path, receipt: dict[str, object], values: tuple[float, ...]) -> None:
    runs = receipt["runs"]
    assert isinstance(runs, list) and len(runs) == len(values)
    for run, value in zip(runs, values):
        assert isinstance(run, dict)
        run["generation_tps"] = value
        raw_output = run["raw_output"]
        assert isinstance(raw_output, dict)
        raw_path = root / str(raw_output["path"])
        text = (
            "Prompt: 32 tokens, 100.0 tokens-per-sec\n"
            f"Generation: 64 tokens, {value} tokens-per-sec\n"
            "Peak memory: 1.0 GB\n"
        )
        raw_path.write_text(text, encoding="utf-8")
        raw_output["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw_output["size_bytes"] = len(text.encode("utf-8"))
    receipt["aggregate"] = validate_benchmarks.recompute_aggregates(runs)
    attach_schema_contracts(root, receipt)


def write_complete_pair(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    baseline = complete_receipt(root, "baseline", role="baseline")
    write_json(root / "baseline.json", baseline)
    candidate = complete_receipt(root, "candidate", role="candidate")
    candidate["models"]["target"]["revision"] = "c" * 40
    comparison = candidate["comparison"]
    assert isinstance(comparison, dict)
    comparison.update({
        "baseline_receipt": "baseline.json",
        "baseline_sha256": hashlib.sha256((root / "baseline.json").read_bytes()).hexdigest(),
    })
    attach_schema_contracts(root, candidate)
    return baseline, candidate


def external_argv_template(executable: str = "python3") -> list[dict[str, object]]:
    return [
        {"literal": executable},
        {"source": ["workload", "artifacts", 0, "path"]},
        {"literal": "--model"},
        {"source": ["models", "target", "id"]},
        {"literal": "--revision"},
        {"source": ["models", "target", "revision"]},
        {"literal": "--input"},
        {"source": ["workload", "artifacts", 1, "path"]},
        {"literal": "--steps"},
        {"source": ["workload", "parameters", "steps"]},
        {"literal": "--mode"},
        {"source": ["variant_config", "mode"]},
        {"literal": "--output"},
        {"source": ["variant_config", "quality_output_path"]},
    ]


def attach_external_quality(root: Path, receipt: dict[str, object]) -> None:
    reference = artifact(root, "quality/artifacts/external-reference.txt", "stable output\n")
    candidate = artifact(
        root,
        f"quality/outputs/{receipt['label']}/observed.txt",
        "stable output\n",
    )
    contract = {
        "schema_version": 2,
        "validator": validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
        "metric": "exact-output-parity",
        "reference_artifact": reference,
        "candidate_artifact": candidate,
    }
    contract_text = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(contract_text.encode("utf-8")).hexdigest()
    contract_artifact = artifact(root, f"quality/inputs/{digest}.json", contract_text)
    payload = validate_benchmarks.build_controlled_exact_output_quality_payload(
        receipt,
        contract_artifact,
        contract,
        root,
    )
    quality_path = root / "quality" / f"{receipt['label']}.json"
    write_json(quality_path, payload)
    receipt["quality"] = {
        "status": "pass",
        "artifact": {
            "path": str(quality_path.relative_to(root)),
            "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            "size_bytes": quality_path.stat().st_size,
        },
    }


def external_receipt(
    root: Path,
    label: str,
    *,
    role: str,
    wall_values: tuple[float, ...],
    baseline_path: Path | None = None,
    template: list[dict[str, object]] | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    runner = artifact(root, "runners/family-neutral-runner.py", "# digest-pinned fixture runner\n")
    workload_input = artifact(root, "inputs/family-neutral-input.bin", "controlled input\n")
    quality_output = artifact(
        root,
        f"quality/outputs/{label}/observed.txt",
        "stable output\n",
    )
    execution_environment_sha256 = "e" * 64
    interpreter_environment_payload = {
        "schema_version": 1,
        "sys_path_sha256": "9" * 64,
        "distributions": [],
        "startup_files": [],
    }
    interpreter = {
        "name": "python3",
        "sha256": "f" * 64,
        "size_bytes": 1024,
        "version": "Python 3.14.4",
        "flags": ["-I", "-B"],
        "environment": {
            **interpreter_environment_payload,
            "sha256": validate_benchmarks.canonical_hash(interpreter_environment_payload),
        },
    }
    system = {
        "platform": "macOS-test",
        "machine": "arm64",
        "processor": "arm",
        "python": "3.14.4",
        "cpu_count": 10,
        "mac_model": "MacTest,1",
        "cpu_brand": "Apple Test",
        "memory_bytes": 16_000_000_000,
        "macos_version": "26.0",
        "mlx_version": "0.32.0",
    }
    target_descriptor = {
        "hardware": {"chip": "Apple Test", "model": "MacTest,1", "memory_bytes": 16_000_000_000},
        "software": {"python": "3.14.4", "mlx": "0.32.0"},
        "benchmark_system": system,
        "execution_environment_sha256": execution_environment_sha256,
        "interpreter": interpreter,
    }
    workload_descriptor = {
        "id": "family-neutral-workload",
        "artifacts": [
            {"role": "runner", **runner},
            {"role": "input", **workload_input},
        ],
        "parameters": {"steps": 4},
    }
    receipt: dict[str, object] = {
        "schema_version": 2,
        "label": label,
        "timestamp": "2026-07-10T00:00:00+00:00",
        "environment": system,
        "versions": {"mlx": "0.32.0"},
        "cwd": ".",
        "config_notes": {
            "mode": mode or role,
            "quality_output_path": quality_output["path"],
        },
        "variant_config": {
            "mode": mode or role,
            "quality_output_path": quality_output["path"],
        },
        "enabled_methods": ["fixture-optimization"] if role == "candidate" else [],
        "models": {
            "target": {
                "id": "example/non-language-model",
                "revision": "a" * 40,
                "lineage_id": "example-family-neutral-lineage",
                "source_id": "example/source-model",
                "source_revision": "b" * 40,
            }
        },
        "target": {"descriptor": target_descriptor, "sha256": validate_benchmarks.canonical_hash(target_descriptor)},
        "workload": {**workload_descriptor, "sha256": validate_benchmarks.canonical_hash(workload_descriptor)},
        "comparison": {"role": role, "primary_metric": "wall_seconds"},
        "warmup_runs": 0,
        "measured_runs": len(wall_values),
        "timeout_seconds": 10.0,
        "rollback_condition": "Rollback when exact output changes or wall time does not beat noise.",
    }
    template = template or external_argv_template()
    receipt["command"] = validate_benchmarks.resolve_external_argv_template(receipt, template)
    receipt["command_display"] = shlex.join(receipt["command"])
    report_runs = [
        {
            "wall_seconds": value,
            "returncode": 0,
            "timed_out": False,
            "peak_rss_bytes": None,
            "stdout_tail": "self-reported wall_seconds=0.000001 is not trusted",
            "stderr_tail": "",
            "phase": "measure",
            "quality_output": quality_output,
        }
        for value in wall_values
    ]
    report = {
        "schema_version": 1,
        "generated_at": receipt["timestamp"],
        "command": receipt["command"],
        "command_display": receipt["command_display"],
        "cwd": ".",
        "environment_overrides": {},
        "execution_environment_sha256": execution_environment_sha256,
        "interpreter": interpreter,
        "system": system,
        "warmup_count": 0,
        "requested_runs": len(wall_values),
        "timeout_seconds": 10.0,
        "ok": True,
        "summary": {
            "successful_runs": len(wall_values),
            "wall_seconds_min": min(wall_values),
            "wall_seconds_median": statistics.median(wall_values),
            "wall_seconds_mean": statistics.mean(wall_values),
            "wall_seconds_p95": benchmark_command.percentile(list(wall_values), 0.95),
            "wall_seconds_max": max(wall_values),
            "peak_rss_bytes_max": None,
        },
        "warmups": [],
        "runs": report_runs,
        "notes": [],
    }
    report_path = root / "raw" / label / "benchmark-command.json"
    write_json(report_path, report)
    raw = {
        "path": str(report_path.relative_to(root)),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "size_bytes": report_path.stat().st_size,
        "truncated": False,
    }
    receipt["runs"] = [
        {"run": index, "wall_seconds": value, "raw_output": raw}
        for index, value in enumerate(wall_values, start=1)
    ]
    receipt["aggregate"] = validate_benchmarks.recompute_aggregates(
        receipt["runs"], validate_benchmarks.EXTERNAL_METRICS
    )
    receipt["stability"] = {"primary_metric": "wall_seconds", "max_cv": 0.05, "min_runs": 5}
    if baseline_path is not None:
        receipt["comparison"].update({
            "baseline_receipt": str(baseline_path.relative_to(root)),
            "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        })
    receipt["runner"] = validate_benchmarks.build_external_runner_descriptor(receipt, template)
    receipt["experiment"] = validate_benchmarks.build_experiment_contract(receipt)
    attach_external_quality(root, receipt)
    return receipt


def write_external_pair(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    baseline = external_receipt(
        root,
        "external-baseline",
        role="baseline",
        wall_values=(1.00, 1.01, 0.99, 1.00, 1.00),
    )
    baseline_path = root / "external-baseline.json"
    write_json(baseline_path, baseline)
    candidate = external_receipt(
        root,
        "external-candidate",
        role="candidate",
        wall_values=(0.70, 0.71, 0.69, 0.70, 0.70),
        baseline_path=baseline_path,
    )
    return baseline, candidate


def attested_argv_template() -> list[dict[str, object]]:
    return [
        {"literal": "python3.13"},
        {"source": ["workload", "artifacts", 0, "path"]},
        {"literal": "--model"},
        {"source": ["models", "target", "id"]},
        {"literal": "--revision"},
        {"source": ["models", "target", "revision"]},
        {"literal": "--input"},
        {"source": ["workload", "artifacts", 1, "path"]},
        {"literal": "--steps"},
        {"source": ["workload", "parameters", "generate_steps"]},
        {"literal": "--mode"},
        {"source": ["variant_config", "mode"]},
        {"literal": "--output"},
        {"source": ["variant_config", "quality_output_path"]},
        {"literal": "--attestation-challenge"},
        {"source": ["variant_config", "attestation_challenge_path"]},
        {"literal": "--attestation-output"},
        {"source": ["variant_config", "attestation_output_path"]},
    ]


def _attested_dependency(
    root: Path,
    *,
    scope: str,
    module_names: list[str],
    loaded_path: str,
    content: str,
) -> dict[str, object]:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "scope": scope,
        "module_names": module_names,
        "loaded_path": loaded_path,
        "artifact": artifact(root, f"attestations/dependencies/{digest}.bin", content),
    }


def attested_receipt(
    root: Path,
    label: str,
    *,
    role: str,
    wall_values: tuple[float, ...],
    baseline_path: Path | None = None,
) -> dict[str, object]:
    mode = "f32" if role == "baseline" else "bf16"
    rule = validate_benchmarks.ATTESTED_VARIANTS[mode]
    runner_text = (BENCHMARKS / validate_benchmarks.ATTESTED_RUNNER_PATH).read_text(
        encoding="utf-8"
    )
    runner_artifact = artifact(
        root,
        validate_benchmarks.ATTESTED_RUNNER_PATH,
        runner_text,
    )
    input_artifact = artifact(
        root,
        "qwen2.5-0.5b/input.json",
        (BENCHMARKS / "qwen2.5-0.5b/input.json").read_text(encoding="utf-8"),
    )
    expected_text = (BENCHMARKS / "qwen2.5-0.5b/expected.json").read_text(
        encoding="utf-8"
    )
    reference = artifact(root, "qwen2.5-0.5b/expected.json", expected_text)
    quality_output = artifact(
        root,
        f"quality/outputs/{label}/result.json",
        expected_text,
    )
    quality_contract = {
        "schema_version": 2,
        "validator": validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
        "metric": "exact-output-parity",
        "reference_artifact": reference,
        "candidate_artifact": quality_output,
    }
    interpreter_payload = {
        "schema_version": 1,
        "sys_path_sha256": "9" * 64,
        "distributions": [],
        "startup_files": [],
    }
    interpreter = {
        "name": "python3.13",
        "sha256": "f" * 64,
        "size_bytes": 1024,
        "version": "Python 3.13.5",
        "flags": ["-I", "-B"],
        "environment": {
            **interpreter_payload,
            "sha256": validate_benchmarks.canonical_hash(interpreter_payload),
        },
    }
    system = {
        "platform": "macOS-test",
        "machine": "arm64",
        "processor": "arm",
        "python": "3.13.5",
        "cpu_count": 10,
        "mac_model": "Mac16,8",
        "cpu_brand": "Apple M4 Pro",
        "memory_bytes": 48_000_000_000,
        "macos_version": "26.0",
        "mlx_version": "0.27.1",
    }
    target_descriptor = {
        "hardware": {
            "chip": "Apple M4 Pro",
            "model": "Mac16,8",
            "memory_bytes": 48_000_000_000,
        },
        "software": {"python": "3.13.5", "mlx": "0.27.1"},
        "benchmark_system": system,
        "execution_environment_sha256": "e" * 64,
        "interpreter": interpreter,
    }
    workload_descriptor = {
        "id": validate_benchmarks.ATTESTED_WORKLOAD_ID,
        "artifacts": [
            {"role": "runner", **runner_artifact},
            {"role": "input", **input_artifact},
        ],
        "parameters": validate_benchmarks.ATTESTED_WORKLOAD_PARAMETERS,
    }
    variant = {
        "dtype_policy": rule["dtype_policy"],
        "mode": mode,
        "quality_output_path": quality_output["path"],
        "weights_bytes": rule["weights_bytes"],
        "weights_sha256": rule["revision"],
        "attestation_challenge_path": f"attestations/{label}/current-challenge.json",
        "attestation_output_path": f"attestations/{label}/current-evidence.json",
    }
    receipt: dict[str, object] = {
        "schema_version": 2,
        "label": label,
        "timestamp": "2026-07-10T00:00:00+00:00",
        "environment": system,
        "versions": {"mlx": "0.27.1"},
        "cwd": ".",
        "config_notes": variant,
        "variant_config": variant,
        "enabled_methods": ["bf16-weight-cast"] if role == "candidate" else [],
        "models": {
            "target": {
                "id": validate_benchmarks.ATTESTED_MODEL_ID,
                "revision": rule["revision"],
                "lineage_id": "qwen2.5-0.5b-instruct-worked-port",
                "source_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "source_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            }
        },
        "target": {
            "descriptor": target_descriptor,
            "sha256": validate_benchmarks.canonical_hash(target_descriptor),
        },
        "workload": {
            **workload_descriptor,
            "sha256": validate_benchmarks.canonical_hash(workload_descriptor),
        },
        "comparison": {"role": role, "primary_metric": "wall_seconds"},
        "warmup_runs": 0,
        "measured_runs": len(wall_values),
        "timeout_seconds": 10.0,
        "rollback_condition": "Rollback on output mismatch or a wall-time gain within noise.",
    }
    if baseline_path is not None:
        receipt["comparison"].update({
            "baseline_receipt": baseline_path.name,
            "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        })
    template = attested_argv_template()
    receipt["command"] = validate_benchmarks.resolve_external_argv_template(receipt, template)
    receipt["command_display"] = shlex.join(receipt["command"])

    dependencies = [
        _attested_dependency(
            root,
            scope="mlx-runtime",
            module_names=["mlx", "mlx.core"],
            loaded_path="runtime/core.py",
            content="# mlx runtime fixture\n",
        ),
        _attested_dependency(
            root,
            scope="port-package",
            module_names=["model"],
            loaded_path="port/model.py",
            content="# generated model fixture\n",
        ),
        _attested_dependency(
            root,
            scope="port-package",
            module_names=["config"],
            loaded_path="port/config.py",
            content="# generated config fixture\n",
        ),
        _attested_dependency(
            root,
            scope="port-config",
            module_names=[],
            loaded_path="port/config.json",
            content="{\"model_type\":\"qwen2\"}\n",
        ),
    ]
    expected_workload = validate_benchmarks.expected_attested_workload(receipt)
    assert expected_workload is not None
    report_runs: list[dict[str, object]] = []
    receipt_attestations: list[dict[str, object]] = []
    quality_contract_text = json.dumps(quality_contract, indent=2) + "\n"
    for run_index, wall_seconds in enumerate(wall_values, start=1):
        prefix = f"attestations/{label}/runs/measure-{run_index:03d}"
        contract_snapshot = artifact(
            root,
            f"{prefix}/quality-contract.json",
            quality_contract_text,
        )
        challenge = {
            "schema_version": 1,
            "nonce": f"{run_index:064x}",
            "receipt_label": label,
            "phase": "measure",
            "run_index": run_index,
            "command": receipt["command"],
            "command_sha256": validate_benchmarks.canonical_hash(receipt["command"]),
            "runner_argv_sha256": validate_benchmarks.canonical_hash(receipt["command"][1:]),
            "quality_contract": contract_snapshot,
        }
        challenge_artifact = artifact(
            root,
            f"{prefix}/challenge.json",
            json.dumps(challenge, indent=2) + "\n",
        )
        output_snapshot = artifact(root, f"{prefix}/output.json", expected_text)
        evidence_payload = {
            "schema_version": 1,
            "adapter": {
                "id": validate_benchmarks.ATTESTED_RUNNER_ID,
                "version": 1,
                "implementation": runner_artifact,
            },
            "challenge": {
                "sha256": challenge_artifact["sha256"],
                "size_bytes": challenge_artifact["size_bytes"],
            },
            "runner_argv_sha256": validate_benchmarks.canonical_hash(receipt["command"][1:]),
            "workload": {
                "descriptor": expected_workload,
                "sha256": validate_benchmarks.canonical_hash(expected_workload),
            },
            "model_artifact": {
                "path": rule["weights_path"],
                "sha256": rule["revision"],
                "size_bytes": rule["weights_bytes"],
            },
            "dependencies": dependencies,
            "dependency_set_sha256": validate_benchmarks.canonical_hash(dependencies),
            "output_artifact": quality_output,
        }
        evidence = {
            **evidence_payload,
            "evidence_sha256": validate_benchmarks.canonical_hash(evidence_payload),
        }
        evidence_artifact = artifact(
            root,
            f"{prefix}/evidence.json",
            json.dumps(evidence, indent=2) + "\n",
        )
        execution_attestation = {
            "challenge": challenge_artifact,
            "evidence": evidence_artifact,
            "output": output_snapshot,
        }
        receipt_attestations.append(execution_attestation)
        report_runs.append({
            "wall_seconds": wall_seconds,
            "returncode": 0,
            "timed_out": False,
            "peak_rss_bytes": None,
            "stdout_tail": "execution_attested=true is self-report only",
            "stderr_tail": "",
            "phase": "measure",
            "quality_output": quality_output,
            "execution_attestation": execution_attestation,
        })
    report = {
        "schema_version": 1,
        "generated_at": receipt["timestamp"],
        "command": receipt["command"],
        "command_display": receipt["command_display"],
        "cwd": ".",
        "environment_overrides": {},
        "system": system,
        "warmup_count": 0,
        "requested_runs": len(wall_values),
        "timeout_seconds": 10.0,
        "ok": True,
        "summary": {
            "successful_runs": len(wall_values),
            "wall_seconds_min": min(wall_values),
            "wall_seconds_median": statistics.median(wall_values),
            "wall_seconds_mean": statistics.mean(wall_values),
            "wall_seconds_p95": benchmark_command.percentile(list(wall_values), 0.95),
            "wall_seconds_max": max(wall_values),
            "peak_rss_bytes_max": None,
        },
        "warmups": [],
        "runs": report_runs,
        "notes": [],
        "execution_environment_sha256": "e" * 64,
        "interpreter": interpreter,
    }
    report_path = root / "raw" / label / "benchmark-command.json"
    write_json(report_path, report)
    report_artifact = {
        "path": str(report_path.relative_to(root)),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "size_bytes": report_path.stat().st_size,
        "truncated": False,
    }
    receipt["runs"] = [
        {
            "run": run_index,
            "wall_seconds": wall_seconds,
            "raw_output": report_artifact,
            "execution_attestation": execution_attestation,
        }
        for run_index, (wall_seconds, execution_attestation) in enumerate(
            zip(wall_values, receipt_attestations),
            start=1,
        )
    ]
    receipt["aggregate"] = validate_benchmarks.recompute_aggregates(
        receipt["runs"],
        validate_benchmarks.EXTERNAL_METRICS,
    )
    receipt["stability"] = {
        "primary_metric": "wall_seconds",
        "max_cv": 0.10,
        "min_runs": 5,
    }
    receipt["runner"] = validate_benchmarks.build_external_runner_descriptor(receipt, template)
    receipt["experiment"] = validate_benchmarks.build_experiment_contract(receipt)
    canonical_contract = json.dumps(
        quality_contract,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    contract_digest = hashlib.sha256(canonical_contract.encode("utf-8")).hexdigest()
    contract_artifact = artifact(
        root,
        f"quality/inputs/{contract_digest}.json",
        canonical_contract,
    )
    quality_payload = validate_benchmarks.build_controlled_exact_output_quality_payload(
        receipt,
        contract_artifact,
        quality_contract,
        root,
    )
    bound_path = root / "quality" / f"{label}.bound.json"
    write_json(bound_path, quality_payload)
    receipt["quality"] = {
        "status": "pass",
        "artifact": {
            "path": str(bound_path.relative_to(root)),
            "sha256": hashlib.sha256(bound_path.read_bytes()).hexdigest(),
            "size_bytes": bound_path.stat().st_size,
        },
    }
    return receipt


def write_attested_pair(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    baseline = attested_receipt(
        root,
        "attested-baseline",
        role="baseline",
        wall_values=(1.00, 1.01, 0.99, 1.00, 1.00),
    )
    baseline_path = root / "attested-baseline.json"
    write_json(baseline_path, baseline)
    candidate = attested_receipt(
        root,
        "attested-candidate",
        role="candidate",
        wall_values=(0.70, 0.71, 0.69, 0.70, 0.70),
        baseline_path=baseline_path,
    )
    write_json(root / "attested-candidate.json", candidate)
    return baseline, candidate


def rewrite_attested_report(root: Path, receipt: dict[str, object]) -> None:
    runs = receipt["runs"]
    assert isinstance(runs, list) and runs
    first_raw = runs[0]["raw_output"]
    assert isinstance(first_raw, dict)
    report_path = root / str(first_raw["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for report_run, receipt_run in zip(report["runs"], runs):
        if "execution_attestation" in receipt_run:
            report_run["execution_attestation"] = receipt_run["execution_attestation"]
        else:
            report_run.pop("execution_attestation", None)
    write_json(report_path, report)
    raw_artifact = {
        "path": str(report_path.relative_to(root)),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "size_bytes": report_path.stat().st_size,
        "truncated": False,
    }
    for run in runs:
        run["raw_output"] = raw_artifact


class BenchmarkEvidenceContractTests(unittest.TestCase):
    def test_historical_receipts_remain_immutable_observations(self) -> None:
        for name, expected in HISTORICAL_HASHES.items():
            self.assertEqual(hashlib.sha256((BENCHMARKS / name).read_bytes()).hexdigest(), expected, name)

        first = validate_benchmarks.build_assessment_report(BENCHMARKS)
        second = validate_benchmarks.build_assessment_report(BENCHMARKS)
        self.assertEqual(first, second)
        summary = validate_benchmarks.assessment_summary(first)
        self.assertEqual(summary["receipt_count"], 13)
        self.assertEqual(summary["promotion_ready_count"], 1)
        self.assertEqual(summary["rejected_count"], 1)
        by_label = {row["label"]: row for row in first["assessments"]}
        self.assertEqual(
            [
                label for label, row in by_label.items()
                if row["classification"] == "promotion_ready"
            ],
            ["qwen2.5-0.5b-port-bf16"],
        )
        self.assertTrue(all(row["gates"]["aggregates_recomputed"] for row in by_label.values()))
        self.assertIn("incompatible-quant-baseline", by_label["quant-4bit"]["reasons"])
        self.assertIn("missing-checked-in-input", by_label["pcache-warm"]["reasons"])
        self.assertIn("baseline-workload-incompatible", by_label["pcache-warm"]["reasons"])
        self.assertIn("mlx-lm-0.31.1-speculative-correctness-fix", by_label["spec-draft-k2"]["reasons"])
        self.assertEqual(by_label["stack-measured-together"]["classification"], "rejected")
        self.assertEqual(
            by_label["stack-measured-together"]["enabled_methods"],
            ["uniform-kv-quantization", "prompt-prefix-cache", "draft-model-speculation"],
        )
        self.assertIn("partial-invalid-stack-configuration", by_label["stack-measured-together"]["reasons"])

    def test_schema2_candidate_stays_observation_until_execution_is_attested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            write_json(root / "candidate.json", candidate)

            report = validate_benchmarks.build_assessment_report(root)
            by_label = {row["label"]: row for row in report["assessments"]}
            self.assertEqual(by_label["baseline"]["classification"], "performance_observation")
            self.assertEqual(by_label["candidate"]["classification"], "performance_observation")
            self.assertEqual(by_label["candidate"]["reasons"], ["execution-semantics-unattested"])
            self.assertTrue(by_label["candidate"]["gates"]["baseline_compatible"])
            self.assertTrue(by_label["candidate"]["gates"]["improvement_beyond_noise"])
            self.assertTrue(by_label["candidate"]["gates"]["runner_valid"])
            self.assertTrue(by_label["candidate"]["gates"]["quality_binding_valid"])
            self.assertTrue(by_label["candidate"]["gates"]["experiment_compatible"])
            self.assertFalse(by_label["candidate"]["gates"]["execution_attested"])
            self.assertIsNone(by_label["candidate"]["experiment_fingerprint"])
            direct_fingerprint = validate_benchmarks.build_experiment_fingerprint(
                candidate,
                receipt_sha256=hashlib.sha256((root / "candidate.json").read_bytes()).hexdigest(),
                root=root,
            )
            self.assertTrue(_common._experiment_fingerprint_valid(direct_fingerprint))
            self.assertIsNone(by_label["baseline"]["experiment_fingerprint"])
            self.assertGreater(
                by_label["candidate"]["improvement"]["observed_ratio"],
                by_label["candidate"]["improvement"]["required_ratio"],
            )
            self.assertEqual(
                set(by_label["candidate"]["recomputed_median_ratios"]),
                {"decode_tps", "prefill_tps", "ttft_proxy_inverse", "peak_memory_inverse"},
            )

    def test_mlx_baseline_compatibility_binds_protocol_and_operating_system(self) -> None:
        mutations = (
            ("warmup count", lambda receipt: receipt.__setitem__("warmup_runs", 0)),
            ("declared run count", lambda receipt: receipt.__setitem__("measured_runs", 99)),
            ("timeout", lambda receipt: receipt.__setitem__("timeout_seconds", 120.0)),
            (
                "operating system",
                lambda receipt: receipt["environment"].__setitem__("macos_version", "27.0"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, candidate = write_complete_pair(root)
                mutate(candidate)
                attach_schema_contracts(root, candidate)
                write_json(root / "candidate.json", candidate)

                row = next(
                    item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                    if item["label"] == "candidate"
                )
                self.assertFalse(row["gates"]["experiment_compatible"])
                self.assertFalse(row["gates"]["baseline_compatible"])
                self.assertIn("experiment-invariant-incompatible", row["reasons"])

    def test_family_neutral_external_wall_time_pair_stays_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_external_pair(root)
            candidate_path = root / "external-candidate.json"
            write_json(candidate_path, candidate)

            report = validate_benchmarks.build_assessment_report(root)
            by_label = {row["label"]: row for row in report["assessments"]}
            row = by_label["external-candidate"]
            self.assertEqual(row["classification"], "performance_observation")
            self.assertEqual(row["primary_metric"], "wall_seconds")
            self.assertEqual(
                set(row["recomputed_median_ratios"]),
                {"wall_seconds_inverse"},
            )
            self.assertGreater(row["recomputed_median_ratios"]["wall_seconds_inverse"], 1.4)
            self.assertTrue(row["gates"]["runner_valid"])
            self.assertTrue(row["gates"]["raw_outputs_valid"])
            self.assertIn("execution-semantics-unattested", row["reasons"])
            self.assertFalse(row["gates"]["execution_attested"])
            self.assertIsNone(row["experiment_fingerprint"])
            fingerprint = validate_benchmarks.build_experiment_fingerprint(
                candidate,
                receipt_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                root=root,
            )
            self.assertTrue(generate_claim_catalog.experiment_fingerprint_valid(fingerprint))
            self.assertTrue(_common._experiment_fingerprint_valid(fingerprint))
            self.assertEqual(
                fingerprint["payload"]["measured_runs"][0]["metrics"],
                {"wall_seconds": 0.70},
            )

    def test_repository_owned_attested_lane_can_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_attested_pair(root)
            report = validate_benchmarks.build_assessment_report(root)
            by_label = {row["label"]: row for row in report["assessments"]}
            self.assertTrue(by_label["attested-baseline"]["gates"]["execution_attested"])
            row = by_label["attested-candidate"]
            self.assertEqual(row["classification"], "promotion_ready")
            self.assertTrue(row["promotion_ready"])
            self.assertTrue(row["gates"]["execution_attested"])
            self.assertEqual(row["reasons"], [])
            self.assertGreater(
                row["recomputed_median_ratios"]["wall_seconds_inverse"],
                row["improvement"]["required_ratio"],
            )
            self.assertIsNotNone(row["experiment_fingerprint"])

    def test_attested_system_metadata_uses_executed_interpreter_versions(self) -> None:
        system = {"python": "3.14-parent", "mlx_version": "0.30.4"}
        interpreter = {
            "version": "Python 3.13.11",
            "environment": {
                "distributions": [
                    {"name": "mlx", "version": "0.24.0"},
                ]
            },
        }
        self.assertEqual(
            benchmark_command.attested_system_metadata(system, interpreter),
            {"python": "3.13.11", "mlx_version": "0.24.0"},
        )

    def test_attested_lane_rejects_forged_runner_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_attested_pair(root)
            runner_path = root / validate_benchmarks.ATTESTED_RUNNER_PATH
            runner_path.write_text("# forged runner bytes\n", encoding="utf-8")

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == candidate["label"]
            )
            self.assertFalse(row["gates"]["execution_attested"])
            self.assertIn("attestation-runner-digest-mismatch", row["reasons"])

    def test_attested_lane_rejects_swapped_output_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_attested_pair(root)
            output = candidate["runs"][0]["execution_attestation"]["output"]
            (root / output["path"]).write_text("swapped output\n", encoding="utf-8")

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == candidate["label"]
            )
            self.assertFalse(row["gates"]["execution_attested"])
            self.assertIn("attestation-output-digest-mismatch", row["reasons"])

    def test_attested_lane_rejects_edited_dependency_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_attested_pair(root)
            run = candidate["runs"][0]
            evidence_artifact = run["execution_attestation"]["evidence"]
            evidence_path = root / evidence_artifact["path"]
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["dependencies"][0]["artifact"]["sha256"] = "0" * 64
            evidence["dependency_set_sha256"] = validate_benchmarks.canonical_hash(
                evidence["dependencies"]
            )
            evidence_payload = {
                key: value for key, value in evidence.items() if key != "evidence_sha256"
            }
            evidence["evidence_sha256"] = validate_benchmarks.canonical_hash(evidence_payload)
            write_json(evidence_path, evidence)
            evidence_artifact.update({
                "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                "size_bytes": evidence_path.stat().st_size,
            })
            rewrite_attested_report(root, candidate)
            write_json(root / "attested-candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == candidate["label"]
            )
            self.assertFalse(row["gates"]["execution_attested"])
            self.assertIn("attestation-dependency-digest-mismatch", row["reasons"])

    def test_attested_lane_rejects_copied_evidence_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_attested_pair(root)
            source = candidate["runs"][0]["execution_attestation"]["evidence"]
            copied = candidate["runs"][1]["execution_attestation"]["evidence"]
            copied_path = root / copied["path"]
            copied_path.write_bytes((root / source["path"]).read_bytes())
            copied.update({
                "sha256": hashlib.sha256(copied_path.read_bytes()).hexdigest(),
                "size_bytes": copied_path.stat().st_size,
            })
            rewrite_attested_report(root, candidate)
            write_json(root / "attested-candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == candidate["label"]
            )
            self.assertFalse(row["gates"]["execution_attested"])
            self.assertIn("attestation-challenge-mismatch", row["reasons"])

    def test_attested_lane_rejects_self_reported_only_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_attested_pair(root)
            for run in candidate["runs"]:
                run.pop("execution_attestation")
            rewrite_attested_report(root, candidate)
            write_json(root / "attested-candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == candidate["label"]
            )
            self.assertFalse(row["gates"]["execution_attested"])
            self.assertIn("attestation-evidence-missing", row["reasons"])

    def test_external_repetition_identity_ignores_only_label_owned_quality_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = external_receipt(
                root,
                "identity-baseline",
                role="baseline",
                wall_values=(1.00, 1.01, 0.99, 1.00, 1.00),
            )
            baseline_path = root / "identity-baseline.json"
            write_json(baseline_path, baseline)

            fingerprints: list[dict[str, object]] = []
            compatible_receipt: dict[str, object] | None = None
            for label, wall_values in (
                ("identity-candidate-a", (0.70, 0.71, 0.69, 0.70, 0.70)),
                ("identity-candidate-b", (0.72, 0.71, 0.73, 0.72, 0.72)),
            ):
                receipt = external_receipt(
                    root,
                    label,
                    role="candidate",
                    wall_values=wall_values,
                    baseline_path=baseline_path,
                )
                receipt_path = root / f"{label}.json"
                write_json(receipt_path, receipt)
                fingerprints.append(validate_benchmarks.build_experiment_fingerprint(
                    receipt,
                    receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    root=root,
                ))
                compatible_receipt = receipt

            identities = [
                _common.experiment_identity_from_fingerprint(fingerprint)
                for fingerprint in fingerprints
            ]
            self.assertNotEqual(fingerprints[0]["sha256"], fingerprints[1]["sha256"])
            self.assertEqual(identities[0], identities[1])

            changed = external_receipt(
                root,
                "identity-candidate-mode-change",
                role="candidate",
                mode="different-semantic-mode",
                wall_values=(0.70, 0.71, 0.69, 0.70, 0.70),
                baseline_path=baseline_path,
            )
            changed_path = root / "identity-candidate-mode-change.json"
            write_json(changed_path, changed)
            changed_fingerprint = validate_benchmarks.build_experiment_fingerprint(
                changed,
                receipt_sha256=hashlib.sha256(changed_path.read_bytes()).hexdigest(),
                root=root,
            )
            self.assertNotEqual(
                identities[0],
                _common.experiment_identity_from_fingerprint(changed_fingerprint),
            )

            assert compatible_receipt is not None
            for field, value in (("warmup_runs", 1), ("timeout_seconds", 20.0)):
                protocol_changed = copy.deepcopy(compatible_receipt)
                protocol_changed[field] = value
                protocol_changed["experiment"] = validate_benchmarks.build_experiment_contract(
                    protocol_changed
                )
                protocol_path = root / f"identity-protocol-{field}.json"
                write_json(protocol_path, protocol_changed)
                protocol_fingerprint = validate_benchmarks.build_experiment_fingerprint(
                    protocol_changed,
                    receipt_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    root=root,
                )
                self.assertNotEqual(
                    identities[0],
                    _common.experiment_identity_from_fingerprint(protocol_fingerprint),
                    field,
                )

    def test_benchmark_command_receipt_mode_measures_external_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            root = temp_root / "benchmarks"
            root.mkdir()
            runner_path = root / "runner.py"
            runner_path.write_text(
                "import argparse,time\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--revision'); "
                "p.add_argument('--input'); p.add_argument('--steps'); p.add_argument('--mode'); "
                "p.add_argument('--output'); a=p.parse_args()\n"
                "time.sleep(0.30 if a.mode == 'baseline' else 0.05)\n"
                "Path(a.output).write_text('stable output\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            input_path = root / "input.bin"
            input_path.write_text("controlled input\n", encoding="utf-8")
            runner_artifact = artifact(root, "runner.py", runner_path.read_text(encoding="utf-8"))
            input_artifact = artifact(root, "input.bin", input_path.read_text(encoding="utf-8"))
            reference = artifact(root, "quality/reference.txt", "stable output\n")
            fake_system = {
                "platform": "macOS-test",
                "machine": "arm64",
                "processor": "arm",
                "python": "3.14.4",
                "cpu_count": 10,
                "mac_model": "MacTest,1",
                "cpu_brand": "Apple Test",
                "memory_bytes": 16_000_000_000,
                "macos_version": "26.0",
                "mlx_version": "0.32.0",
            }

            def run(label: str, role: str, baseline: str | None = None) -> Path:
                observed = artifact(
                    root,
                    f"quality/outputs/{label}/observed.txt",
                    "stable output\n",
                )
                quality_contract = temp_root / f"{label}.quality-contract.json"
                write_json(quality_contract, {
                    "schema_version": 2,
                    "validator": validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
                    "metric": "exact-output-parity",
                    "reference_artifact": reference,
                    "candidate_artifact": observed,
                })
                spec_path = temp_root / f"{label}.spec.json"
                write_json(spec_path, {
                    "schema_version": 1,
                    "label": label,
                    "argv_template": external_argv_template(),
                    "models": {
                        "target": {
                            "id": "example/non-language-model",
                            "revision": "a" * 40,
                            "lineage_id": "example-family-neutral-lineage",
                            "source_id": "example/source-model",
                            "source_revision": "b" * 40,
                        }
                    },
                    "workload": {
                        "id": "external-cli-workload",
                        "artifacts": [
                            {"role": "runner", **runner_artifact},
                            {"role": "input", **input_artifact},
                        ],
                        "parameters": {"steps": 4},
                    },
                    "variant_config": {
                        "mode": role,
                        "quality_output_path": observed["path"],
                    },
                    "enabled_methods": ["fixture-optimization"] if role == "candidate" else [],
                    "comparison_role": role,
                    "rollback_condition": "Rollback on output or wall-time regression.",
                })
                output = root / f"{label}.json"
                argv = [
                    "benchmark_command.py",
                    "--receipt-spec", str(spec_path),
                    "--quality-contract", str(quality_contract),
                    "--warmup", "0",
                    "--runs", "5",
                    "--timeout", "10",
                    "--output", str(output),
                ]
                if baseline is not None:
                    argv.extend(["--baseline-receipt", baseline])
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(benchmark_command, "environment_metadata", return_value=fake_system),
                ):
                    self.assertEqual(benchmark_command.main(), 0)
                return output

            baseline = run("cli-baseline", "baseline")
            candidate = run("cli-candidate", "candidate", baseline.name)
            receipt = json.loads(candidate.read_text(encoding="utf-8"))
            self.assertEqual(receipt["runner"]["id"], validate_benchmarks.EXTERNAL_RUNNER_ID)
            self.assertEqual(receipt["runner"]["interpreter"]["flags"], ["-I", "-B"])
            self.assertEqual(receipt["cwd"], ".")
            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "cli-candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertIn("execution-semantics-unattested", row["reasons"])
            self.assertGreater(row["recomputed_median_ratios"]["wall_seconds_inverse"], 1.2)

    def test_candidate_cannot_alias_a_nested_baseline_to_a_root_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, candidate = write_external_pair(root)
            nested_path = root / "nested" / "external-baseline.json"
            nested = copy.deepcopy(baseline)
            nested["timestamp"] = "2026-07-10T00:00:01+00:00"
            write_json(nested_path, nested)
            candidate["comparison"].update({
                "baseline_receipt": "nested/external-baseline.json",
                "baseline_sha256": hashlib.sha256(nested_path.read_bytes()).hexdigest(),
            })
            attach_external_quality(root, candidate)
            write_json(root / "external-candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "external-candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertIn("missing-baseline-receipt", row["reasons"])
            self.assertFalse(row["promotion_ready"])

    def test_external_wall_time_rejects_forged_or_copied_parent_report(self) -> None:
        for case in (
            "forged-wall-time",
            "copied-baseline-report",
            "template-mismatch",
            "failed-warmup",
            "hidden-environment",
            "missing-measured-quality",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline, candidate = write_external_pair(root)
                if case == "forged-wall-time":
                    candidate["runs"][0]["wall_seconds"] = 0.000001
                    candidate["aggregate"] = validate_benchmarks.recompute_aggregates(
                        candidate["runs"], validate_benchmarks.EXTERNAL_METRICS
                    )
                elif case == "copied-baseline-report":
                    copied = baseline["runs"][0]["raw_output"]
                    for run in candidate["runs"]:
                        run["raw_output"] = copy.deepcopy(copied)
                    attach_external_quality(root, candidate)
                elif case == "template-mismatch":
                    mismatched = external_argv_template()
                    mismatched[-2] = {"literal": "--different-mode"}
                    candidate["runner"] = validate_benchmarks.build_external_runner_descriptor(candidate, mismatched)
                    candidate["experiment"] = validate_benchmarks.build_experiment_contract(candidate)
                else:
                    raw_output = candidate["runs"][0]["raw_output"]
                    report_path = root / raw_output["path"]
                    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
                    if case == "failed-warmup":
                        candidate["warmup_runs"] = 1
                        raw_report["warmup_count"] = 1
                        raw_report["warmups"] = [{
                            "wall_seconds": 0.5,
                            "returncode": 1,
                            "timed_out": False,
                            "peak_rss_bytes": None,
                            "stdout_tail": "",
                            "stderr_tail": "failed",
                            "phase": "warmup",
                        }]
                    elif case == "hidden-environment":
                        raw_report["environment_overrides"] = {
                            "BENCHMARK_MODE": "[REDACTED]"
                        }
                    else:
                        for measured in raw_report["runs"]:
                            measured.pop("quality_output", None)
                    write_json(report_path, raw_report)
                    raw_output.update({
                        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        "size_bytes": report_path.stat().st_size,
                    })
                    for run in candidate["runs"]:
                        run["raw_output"] = copy.deepcopy(raw_output)
                    attach_external_quality(root, candidate)
                write_json(root / "external-candidate.json", candidate)

                row = next(
                    item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                    if item["label"] == "external-candidate"
                )
                self.assertEqual(row["classification"], "performance_observation")
                if case == "template-mismatch":
                    self.assertFalse(row["gates"]["runner_valid"])
                    self.assertIn("uncontrolled-benchmark-runner", row["reasons"])
                else:
                    self.assertFalse(row["gates"]["raw_outputs_valid"])
                    self.assertIn("raw-output-digest-mismatch", row["reasons"])

    def test_external_runner_rejects_shell_dynamic_code_and_unpinned_implementation(self) -> None:
        self.assertFalse(validate_benchmarks.external_command_is_safe(["sh", "-c", "true"]))
        self.assertFalse(validate_benchmarks.external_command_is_safe(["python3", "-c", "print(1)"]))
        self.assertFalse(validate_benchmarks.external_command_is_safe(["env", "python3", "runner.py"]))
        self.assertFalse(validate_benchmarks.external_command_is_safe(["python3", "runner.py", "--trust-remote-code"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_external_pair(root)
            unsafe_template = external_argv_template("sh")
            with self.assertRaisesRegex(
                validate_benchmarks.SkillError,
                "digest-pinned Python runner",
            ):
                validate_benchmarks.build_external_runner_descriptor(candidate, unsafe_template)
            with self.assertRaisesRegex(
                validate_benchmarks.SkillError,
                "digest-pinned Python runner",
            ):
                validate_benchmarks.build_external_runner_descriptor(
                    candidate,
                    external_argv_template("./python3"),
                )

            unpinned = copy.deepcopy(candidate)
            artifacts = unpinned["workload"]["artifacts"]
            artifacts[0]["role"] = "helper"
            with self.assertRaisesRegex(validate_benchmarks.SkillError, "role runner"):
                validate_benchmarks.build_external_runner_descriptor(
                    unpinned,
                    external_argv_template(),
                )

            revision_only = external_argv_template()
            revision_only[3] = {"source": ["models", "target", "revision"]}
            with self.assertRaisesRegex(validate_benchmarks.SkillError, "missing required bindings: model"):
                validate_benchmarks.resolve_external_argv_template(candidate, revision_only)

    def test_experiment_fingerprint_binds_receipt_measurements_raw_outputs_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            candidate_path = root / "candidate.json"
            write_json(candidate_path, candidate)
            original_receipt_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            original = validate_benchmarks.build_experiment_fingerprint(
                candidate,
                receipt_sha256=original_receipt_sha256,
                root=root,
            )
            payload = original["payload"]
            self.assertEqual(payload["candidate_receipt_sha256"], original_receipt_sha256)
            self.assertEqual(payload["aggregate"], candidate["aggregate"])
            self.assertEqual(
                [row["raw_output"] for row in payload["measured_runs"]],
                [row["raw_output"] for row in candidate["runs"]],
            )
            self.assertEqual(
                payload["quality"]["result_sha256"],
                candidate["quality"]["artifact"]["sha256"],
            )

            performance_changed = copy.deepcopy(candidate)
            set_generation_values(root, performance_changed, (70.0, 70.5, 69.5, 70.2, 69.8))
            performance_path = root / "candidate-performance.json"
            write_json(performance_path, performance_changed)
            performance_fingerprint = validate_benchmarks.build_experiment_fingerprint(
                performance_changed,
                receipt_sha256=hashlib.sha256(performance_path.read_bytes()).hexdigest(),
                root=root,
            )
            self.assertNotEqual(performance_fingerprint["sha256"], original["sha256"])

            quality_changed = copy.deepcopy(candidate)
            quality_changed["label"] = "candidate-quality"
            quality_changed["quality"]["artifact"]["path"] = "quality/candidate-quality.json"
            attach_schema_contracts(root, quality_changed)
            quality_path = root / "candidate-quality.json"
            write_json(quality_path, quality_changed)
            quality_fingerprint = validate_benchmarks.build_experiment_fingerprint(
                quality_changed,
                receipt_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                root=root,
            )
            self.assertNotEqual(
                quality_fingerprint["payload"]["quality"]["result_sha256"],
                original["payload"]["quality"]["result_sha256"],
            )
            self.assertNotEqual(quality_fingerprint["sha256"], original["sha256"])

    def test_controlled_runner_is_bound_to_declared_model_and_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            candidate["command"] = [
                "python3",
                "-m",
                "mlx_lm",
                "generate",
                "--model",
                "unrelated/model",
                "--prompt",
                "different workload",
                "--max-tokens",
                "64",
                "--temp",
                "0.0",
                "--seed",
                "7",
            ]
            candidate["runner"] = validate_benchmarks.build_runner_descriptor(candidate["command"])
            candidate["experiment"] = validate_benchmarks.build_experiment_contract(candidate)
            attach_schema_contracts(root, candidate)
            write_json(root / "candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["runner_valid"])
            self.assertIn("uncontrolled-benchmark-runner", row["reasons"])

    def test_runner_rejects_remote_code_and_path_spoofing(self) -> None:
        for name, mutate in (
            (
                "remote-code",
                lambda command: [*command, "--trust-remote-code"],
            ),
            (
                "path-spoofed-python",
                lambda command: ["/tmp/python3", *command[1:]],
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, candidate = write_complete_pair(root)
                candidate["command"] = mutate(candidate["command"])
                candidate["runner"] = validate_benchmarks.build_runner_descriptor(candidate["command"])
                candidate["experiment"] = validate_benchmarks.build_experiment_contract(candidate)
                attach_schema_contracts(root, candidate)
                write_json(root / "candidate.json", candidate)
                row = next(
                    item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                    if item["label"] == "candidate"
                )
                self.assertEqual(row["classification"], "performance_observation")
                self.assertFalse(row["gates"]["runner_valid"])
                self.assertIn("uncontrolled-benchmark-runner", row["reasons"])

    def test_candidate_cannot_claim_a_method_with_the_baseline_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, candidate = write_complete_pair(root)
            candidate["models"] = copy.deepcopy(baseline["models"])
            attach_schema_contracts(root, candidate)
            write_json(root / "candidate.json", candidate)
            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["enabled_methods_valid"])
            self.assertIn("candidate-invocation-not-distinct", row["reasons"])

    def test_declarative_quality_requires_external_quality_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = complete_receipt(root, "candidate", role="candidate")
            receipt_only = {
                "schema_version": 1,
                "validator": validate_benchmarks.DECLARATIVE_QUALITY_VALIDATOR,
                "evidence": {},
                "checks": [{
                    "id": "run-count-is-not-model-quality",
                    "source": {"kind": "receipt", "path": ["runs"], "reducer": "length"},
                    "comparator": "gte",
                    "threshold": 5,
                }],
            }
            with self.assertRaisesRegex(Exception, "external-json quality check"):
                validate_benchmarks.evaluate_declarative_quality_contract(candidate, receipt_only)

    def test_arbitrary_external_json_quality_value_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            quality_contract = {
                "schema_version": 1,
                "validator": validate_benchmarks.DECLARATIVE_QUALITY_VALIDATOR,
                "evidence": {"arbitrary": 1},
                "checks": [{
                    "id": "self-attested-arbitrary-value",
                    "source": {
                        "kind": "external-json",
                        "path": ["arbitrary"],
                        "reducer": "value",
                    },
                    "comparator": "gte",
                    "threshold": 1,
                }],
            }
            contract_text = json.dumps(quality_contract, indent=2, sort_keys=True) + "\n"
            contract_sha256 = hashlib.sha256(contract_text.encode("utf-8")).hexdigest()
            quality_input = artifact(
                root,
                f"quality/inputs/{contract_sha256}.json",
                contract_text,
            )
            quality_payload = validate_benchmarks.build_declarative_quality_payload(
                candidate,
                quality_input,
                quality_contract,
            )
            quality_path = root / "quality" / "candidate.self-attested.json"
            write_json(quality_path, quality_payload)
            candidate["quality"] = {
                "status": "pass",
                "artifact": {
                    "path": str(quality_path.relative_to(root)),
                    "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                    "size_bytes": quality_path.stat().st_size,
                },
            }
            write_json(root / "candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["quality_valid"])
            self.assertFalse(row["gates"]["quality_binding_valid"])
            self.assertIsNone(row["experiment_fingerprint"])

    def test_stored_python_evaluator_provenance_is_not_proof_of_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            evaluator_text = "# deterministic fixture evaluator\n"
            evaluator_sha256 = hashlib.sha256(evaluator_text.encode("utf-8")).hexdigest()
            evaluator = artifact(
                root,
                f"quality/evaluators/{evaluator_sha256}.py",
                evaluator_text,
            )
            legacy_result = {
                "schema_version": 2,
                "validator": {"id": "mlx-benchmark-quality", "version": 1},
                "status": "pass",
                "checks": [{
                    "id": "self-attested",
                    "comparator": "gte",
                    "observed": 1.0,
                    "threshold": 1.0,
                    "passed": True,
                }],
            }
            input_payload = validate_benchmarks.build_quality_evaluator_input(candidate)
            input_artifact = artifact(
                root,
                "quality/candidate.evaluator-input.json",
                json.dumps(input_payload, indent=2) + "\n",
            )
            output_artifact = artifact(
                root,
                "quality/candidate.evaluator-output.json",
                json.dumps(legacy_result, indent=2) + "\n",
            )
            stderr_artifact = artifact(root, "quality/candidate.evaluator-stderr.txt", "")
            legacy_payload = {
                **legacy_result,
                "bindings": validate_benchmarks.expected_quality_bindings(candidate),
                "provenance": {
                    "mode": "checked-in-evaluator-v1",
                    "evaluator": evaluator,
                    "input": input_artifact,
                    "output": output_artifact,
                    "stderr": stderr_artifact,
                    "invocation": validate_benchmarks.build_quality_evaluator_invocation(
                        evaluator,
                        input_artifact,
                    ),
                },
            }
            legacy_path = root / "quality" / "candidate.legacy-bound.json"
            write_json(legacy_path, legacy_payload)
            candidate["quality"] = {
                "status": "pass",
                "artifact": {
                    "path": str(legacy_path.relative_to(root)),
                    "sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
                    "size_bytes": legacy_path.stat().st_size,
                },
            }
            write_json(root / "candidate.json", candidate)
            evaluator_path = root / evaluator["path"]
            self.assertEqual(evaluator_path.read_text(encoding="utf-8"), "# deterministic fixture evaluator\n")

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["quality_binding_valid"])
            self.assertIn("quality-binding-invalid", row["reasons"])

    def test_legacy_python_evaluator_is_recorded_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "evaluator-ran"
            evaluator = root / "legacy.py"
            evaluator.write_text(
                f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('unsafe')\n",
                encoding="utf-8",
            )
            candidate = complete_receipt(root, "candidate", role="candidate")
            candidate["quality"] = benchmark_generation.record_legacy_quality_evaluator(
                root / "candidate.json",
                evaluator,
            )
            write_json(root / "candidate.json", candidate)

            self.assertFalse(sentinel.exists())
            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["quality_valid"])
            self.assertFalse(row["gates"]["quality_binding_valid"])

    def test_schema2_candidate_rejects_regression_and_noise_level_gain(self) -> None:
        scenarios = {
            "large-regression": (10.0, 10.1, 9.9, 10.0, 10.0),
            "marginal-noisy": (49.0, 53.0, 51.0, 54.0, 50.0),
        }
        for name, values in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, candidate = write_complete_pair(root)
                set_generation_values(root, candidate, values)
                write_json(root / "candidate.json", candidate)
                row = next(
                    item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                    if item["label"] == "candidate"
                )
                self.assertEqual(row["classification"], "rejected")
                self.assertFalse(row["promotion_ready"])
                self.assertFalse(row["gates"]["improvement_beyond_noise"])
                self.assertIn("no-improvement-beyond-noise", row["reasons"])
                self.assertLessEqual(row["improvement"]["observed_ratio"], row["improvement"]["required_ratio"])

    def test_baseline_compatibility_requires_exact_pinned_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            target = candidate["models"]["target"]
            target["revision"] = "c" * 40
            attach_schema_contracts(root, candidate)
            write_json(root / "candidate.json", candidate)
            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertIn("execution-semantics-unattested", row["reasons"])
            self.assertTrue(row["gates"]["baseline_source_identity_match"])
            self.assertTrue(row["gates"]["experiment_compatible"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            target = candidate["models"]["target"]
            target["revision"] = "c" * 40
            target["source_revision"] = "d" * 40
            attach_schema_contracts(root, candidate)
            write_json(root / "candidate.json", candidate)
            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["baseline_source_identity_match"])
            self.assertFalse(row["gates"]["experiment_compatible"])
            self.assertIn("baseline-source-lineage-incompatible", row["reasons"])

    def test_arbitrary_runner_and_loose_quality_attestation_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            candidate["command"] = ["python3", "-c", "print('self-attested benchmark')"]
            attach_schema_contracts(root, candidate)
            quality = candidate["quality"]
            quality_path = root / quality["artifact"]["path"]
            write_json(quality_path, {"status": "pass", "checks": [{"passed": True}]})
            quality["artifact"]["sha256"] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
            quality["artifact"]["size_bytes"] = quality_path.stat().st_size
            write_json(root / "candidate.json", candidate)
            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["runner_valid"])
            self.assertFalse(row["gates"]["quality_binding_valid"])
            self.assertIn("uncontrolled-benchmark-runner", row["reasons"])
            self.assertIn("quality-binding-invalid", row["reasons"])

    def test_handwritten_bound_quality_pass_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, candidate = write_complete_pair(root)
            quality = candidate["quality"]
            quality_path = root / quality["artifact"]["path"]
            write_json(quality_path, {
                "schema_version": 2,
                "validator": {"id": "mlx-benchmark-quality", "version": 1},
                "status": "pass",
                "bindings": validate_benchmarks.expected_quality_bindings(candidate),
                "checks": [{
                    "id": "handwritten-pass",
                    "comparator": "gte",
                    "observed": 1.0,
                    "threshold": 1.0,
                    "passed": True,
                }],
            })
            quality["artifact"]["sha256"] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
            quality["artifact"]["size_bytes"] = quality_path.stat().st_size
            write_json(root / "candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertFalse(row["gates"]["quality_valid"])
            self.assertFalse(row["gates"]["quality_binding_valid"])
            self.assertIn("quality-binding-invalid", row["reasons"])

    def test_tampering_or_missing_evidence_blocks_schema2_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = complete_receipt(root, "baseline", role="baseline")
            write_json(root / "baseline.json", baseline)
            candidate = complete_receipt(root, "candidate", role="candidate")
            candidate["comparison"].update({
                "baseline_receipt": "baseline.json",
                "baseline_sha256": "0" * 64,
            })
            candidate["aggregate"]["generation_tps"]["median"] = 999
            weak_quality_path = root / candidate["quality"]["artifact"]["path"]
            write_json(weak_quality_path, {"schema_version": 1, "status": "pass", "checks": []})
            candidate["quality"]["artifact"]["sha256"] = hashlib.sha256(weak_quality_path.read_bytes()).hexdigest()
            candidate["quality"]["artifact"]["size_bytes"] = weak_quality_path.stat().st_size
            candidate["command"] = ["python3", "--api-key", "secret-value", "/tmp/private-input"]
            write_json(root / "candidate.json", candidate)

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertIn("aggregate-mismatch", row["reasons"])
            self.assertIn("baseline-digest-mismatch", row["reasons"])
            self.assertIn("quality-not-passing", row["reasons"])
            self.assertIn("secret-bearing-command", row["reasons"])
            self.assertIn("nonportable-ephemeral-path", row["reasons"])

    def test_generator_without_complete_experiment_spec_is_schema2_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            root.mkdir()
            output = root / "fixture.json"
            stub = (
                "print('Prompt: 32 tokens, 100.0 tokens-per-sec'); "
                "print('Generation: 64 tokens, 50.0 tokens-per-sec'); "
                "print('Peak memory: 1.0 GB')"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "benchmark_generation.py"),
                    "--label", "fixture",
                    "--warmup", "0",
                    "--runs", "1",
                    "--output", str(output),
                    "--", sys.executable, "-c", stub,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], 2)
            row = validate_benchmarks.build_assessment_report(root)["assessments"][0]
            self.assertEqual(row["classification"], "performance_observation")
            self.assertEqual(receipt["runner"]["id"], "uncontrolled")
            self.assertFalse(row["gates"]["runner_valid"])
            self.assertIn("uncontrolled-benchmark-runner", row["reasons"])
            self.assertIn("missing-model-lineage", row["reasons"])
            self.assertIn("missing-quality-artifact", row["reasons"])
            self.assertTrue(all("raw_output" in run for run in receipt["runs"]))
            generated_report = validate_benchmarks.build_assessment_report(root)
            self.assertEqual(
                json.loads((root / "receipt_assessments.json").read_text(encoding="utf-8")),
                generated_report,
            )
            self.assertEqual(
                json.loads((root / "receipts_index.json").read_text(encoding="utf-8")),
                validate_benchmarks.build_receipts_index(generated_report, root),
            )
            self.assertEqual(
                (root.parent / "BENCHMARK_REPORT.md").read_text(encoding="utf-8"),
                validate_benchmarks.render_benchmark_report(generated_report),
            )

    def test_generator_binds_controlled_exact_output_quality_and_promotes_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            root = temp_root / "benchmarks"
            root.mkdir()
            prompt = root / "prompt.txt"
            prompt.write_text("fixture prompt\n", encoding="utf-8")
            quality_contract = temp_root / "quality-contract.json"
            reference_output = artifact(
                root,
                "quality/artifacts/expected-model-output.txt",
                "fixture exact model output\n",
            )
            candidate_output = artifact(
                root,
                "quality/artifacts/observed-model-output.txt",
                "fixture exact model output\n",
            )
            write_json(quality_contract, {
                "schema_version": 2,
                "validator": validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
                "metric": "exact-output-parity",
                "reference_artifact": reference_output,
                "candidate_artifact": candidate_output,
            })

            fake_modules = temp_root / "fake-modules"
            package = fake_modules / "mlx_lm"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "__main__.py").write_text(
                "import os\n"
                "print('Prompt: 32 tokens, 100.0 tokens-per-sec')\n"
                "print(f\"Generation: 64 tokens, {os.environ['FIXTURE_TPS']} tokens-per-sec\")\n"
                "print('Peak memory: 1.0 GB')\n",
                encoding="utf-8",
            )
            dist_info = fake_modules / "mlx_lm-0.31.3.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: mlx-lm\nVersion: 0.31.3\n",
                encoding="utf-8",
            )
            mlx_package = fake_modules / "mlx"
            mlx_package.mkdir()
            (mlx_package / "__init__.py").write_text("", encoding="utf-8")
            mlx_dist_info = fake_modules / "mlx-0.32.0.dist-info"
            mlx_dist_info.mkdir()
            (mlx_dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: mlx\nVersion: 0.32.0\n",
                encoding="utf-8",
            )
            base_env = os.environ.copy()
            base_env["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(fake_modules), base_env.get("PYTHONPATH")) if value
            )

            def generate(label: str, target_revision: str, tps: str, baseline: Path | None = None) -> Path:
                output = root / f"{label}.json"
                command = [
                    sys.executable,
                    str(SCRIPTS / "benchmark_generation.py"),
                    "--label", label,
                    "--warmup", "0",
                    "--runs", "5",
                    "--output", str(output),
                    "--target-model", "example/converted-model",
                    "--target-revision", target_revision,
                    "--lineage-id", "example-lineage",
                    "--source-id", "example/source-model",
                    "--source-revision", "b" * 40,
                    "--workload-id", "fixture-workload",
                    "--workload-artifact", f"prompt={prompt}",
                    "--workload-params-json", '{"max_tokens":64,"temperature":0.0,"seed":7}',
                    "--quality-contract", str(quality_contract),
                    "--rollback-condition", "Rollback on quality or throughput regression.",
                ]
                if baseline is not None:
                    command.extend(["--baseline-receipt", str(baseline), "--enabled-method", "fixture-optimization"])
                command.extend([
                    "--",
                    "python3",
                    "-m",
                    "mlx_lm",
                    "generate",
                    "--model",
                    "example/converted-model",
                    "--prompt",
                    "fixture prompt\n",
                    "--max-tokens",
                    "64",
                    "--temp",
                    "0.0",
                    "--seed",
                    "7",
                ])
                env = {**base_env, "FIXTURE_TPS": tps}
                completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                return output

            baseline_path = generate("baseline", "a" * 40, "50.0")
            candidate_path = generate("candidate", "c" * 40, "60.0", baseline_path)
            receipt = json.loads(candidate_path.read_text(encoding="utf-8"))
            bound_quality_path = root / receipt["quality"]["artifact"]["path"]
            bound_quality = json.loads(bound_quality_path.read_text(encoding="utf-8"))
            self.assertNotEqual(bound_quality_path.resolve(), quality_contract.resolve())
            self.assertEqual(bound_quality["bindings"], validate_benchmarks.expected_quality_bindings(receipt))
            self.assertEqual(bound_quality["provenance"]["mode"], "controlled-exact-output-v1")
            self.assertEqual(
                bound_quality["validator"],
                validate_benchmarks.EXACT_OUTPUT_QUALITY_VALIDATOR,
            )
            self.assertEqual(receipt["runner"]["id"], "mlx-lm-generate")
            self.assertEqual(receipt["experiment"], validate_benchmarks.build_experiment_contract(receipt))

            row = next(
                item for item in validate_benchmarks.build_assessment_report(root)["assessments"]
                if item["label"] == "candidate"
            )
            self.assertEqual(row["classification"], "performance_observation")
            self.assertIn("execution-semantics-unattested", row["reasons"])
            self.assertTrue(row["gates"]["quality_binding_valid"])
            self.assertTrue(row["gates"]["experiment_compatible"])
            self.assertTrue(row["gates"]["improvement_beyond_noise"])
            self.assertIsNone(row["experiment_fingerprint"])

    def test_committed_assessment_and_index_are_deterministically_generated(self) -> None:
        report = validate_benchmarks.build_assessment_report(BENCHMARKS)
        index = validate_benchmarks.build_receipts_index(report, BENCHMARKS)
        self.assertEqual(
            json.loads((BENCHMARKS / "receipt_assessments.json").read_text(encoding="utf-8")),
            report,
        )
        self.assertEqual(
            json.loads((BENCHMARKS / "receipts_index.json").read_text(encoding="utf-8")),
            index,
        )
        self.assertEqual(
            (SKILL / "assets" / "BENCHMARK_REPORT.md").read_text(encoding="utf-8"),
            validate_benchmarks.render_benchmark_report(report),
        )

    def test_check_detects_human_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_root = Path(tmp) / "assets" / "benchmarks"
            benchmark_root.mkdir(parents=True)
            write_json(benchmark_root / "baseline.json", complete_receipt(benchmark_root, "baseline", role="baseline"))
            command = [
                sys.executable,
                str(SCRIPTS / "validate_benchmarks.py"),
                "generate",
                "--root",
                str(benchmark_root),
            ]
            generated = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            report_path = benchmark_root.parent / "BENCHMARK_REPORT.md"
            report_path.write_text("stale\n", encoding="utf-8")
            checked = subprocess.run(
                [*command[:2], "check", *command[3:]],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(checked.returncode, 1)
            self.assertIn("stale BENCHMARK_REPORT.md", checked.stdout)


if __name__ == "__main__":
    unittest.main()
