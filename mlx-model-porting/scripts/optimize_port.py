#!/usr/bin/env python3
"""Run a measured, quality-gated MLX-LM optimization sweep."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, stable_mean, stable_median, stable_pstdev
import quant_quality_gate as quality_gate


SCHEMA_VERSION = 1
DIAGNOSTIC_KIND = "mlx-port-optimization-loop"
DEFAULT_CONFIG_SWEEP = "8bit,4bit-g64,4bit-g32"
DEFAULT_BENCHMARK_PROMPT = (
    "Explain in one concise sentence why measured optimization is safer than guessing."
)
MAX_SOURCE_CONFIG_BYTES = 2 * 1024 * 1024
BOUNDARY = {
    "classification": "user_diagnostic_only",
    "promotable_claim": False,
    "writes_sealed_evidence": False,
    "note": (
        "These are local observations for this model, Mac, and workload. The report is not a "
        "benchmark receipt, receipt assessment, effective claim, or promotion input."
    ),
}


@dataclass(frozen=True)
class OptimizationConfig:
    config_id: str
    bits: int
    group_size: int
    mode: str = "affine"
    naive_default: bool = False


@dataclass(frozen=True)
class Runtime:
    mx: Any
    load: Any
    convert: Any
    generate_step: Any


CONFIGS = (
    OptimizationConfig("8bit", bits=8, group_size=64),
    OptimizationConfig("4bit-g64", bits=4, group_size=64, naive_default=True),
    OptimizationConfig("4bit-g32", bits=4, group_size=32),
)
CONFIG_BY_ID = {config.config_id: config for config in CONFIGS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert and measure a structured MLX-LM quantization sweep. Strict JSON is written "
            "to stdout and a human-readable table to stderr. Results are user diagnostics only."
        )
    )
    parser.add_argument("--model", required=True, help="Local unquantized Hugging Face/MLX model")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory for bf16 and quantized conversion artifacts",
    )
    parser.add_argument(
        "--configs",
        default=DEFAULT_CONFIG_SWEEP,
        help=(
            "Comma-separated sweep drawn from 8bit, 4bit-g64, and 4bit-g32 "
            f"(default: {DEFAULT_CONFIG_SWEEP})"
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed decode warmups")
    parser.add_argument("--runs", type=int, default=5, help="Timed decode repetitions")
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Greedy tokens per benchmark repetition; the first token is excluded from decode timing",
    )
    parser.add_argument(
        "--benchmark-prompt",
        default=DEFAULT_BENCHMARK_PROMPT,
        help="Fixed prompt used for decode measurement",
    )
    parser.add_argument(
        "--prompts-file",
        help="Optional quant_quality_gate strict-JSON prompt workload",
    )
    parser.add_argument(
        "--quality-max-tokens",
        type=int,
        default=quality_gate.DEFAULT_MAX_TOKENS,
        help="Greedy tokens per quality prompt",
    )
    parser.add_argument(
        "--max-perplexity-ratio",
        type=float,
        default=1.10,
        help="Maximum candidate/reference perplexity ratio (default: 1.10)",
    )
    parser.add_argument(
        "--min-firsttoken-agreement",
        type=float,
        default=0.0,
        help="Optional minimum first-token agreement; exact divergence is always reported",
    )
    parser.add_argument(
        "--max-degenerate-output-rate",
        type=float,
        default=0.0,
        help="Maximum candidate-only degenerate-output rate (default: 0.0)",
    )
    parser.add_argument("--seed", type=int, default=7, help="MLX random seed (default: 7)")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse matching conversion directories instead of refusing to overwrite",
    )
    return parser.parse_args(argv)


def parse_config_sweep(value: str) -> list[OptimizationConfig]:
    if not isinstance(value, str):
        raise SkillError("--configs must be a comma-separated string")
    names = [part.strip() for part in value.split(",")]
    if not names or any(not name for name in names):
        raise SkillError("--configs must contain one or more non-empty config IDs")
    unknown = sorted(set(names) - set(CONFIG_BY_ID))
    if unknown:
        raise SkillError(
            "unknown --configs value(s): "
            f"{', '.join(unknown)}; choose from {', '.join(CONFIG_BY_ID)}"
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SkillError(f"--configs contains duplicate value(s): {', '.join(duplicates)}")
    return [CONFIG_BY_ID[name] for name in names]


def _finite(value: float, flag: str) -> float:
    if not math.isfinite(value):
        raise SkillError(f"{flag} must be finite")
    return value


def validate_args(args: argparse.Namespace) -> list[OptimizationConfig]:
    configs = parse_config_sweep(args.configs)
    if isinstance(args.warmup_runs, bool) or not 0 <= args.warmup_runs <= 10:
        raise SkillError("--warmup-runs must be between 0 and 10")
    if isinstance(args.runs, bool) or not 1 <= args.runs <= 50:
        raise SkillError("--runs must be between 1 and 50")
    if isinstance(args.decode_tokens, bool) or not 2 <= args.decode_tokens <= 512:
        raise SkillError("--decode-tokens must be between 2 and 512")
    if isinstance(args.quality_max_tokens, bool) or not 1 <= args.quality_max_tokens <= 256:
        raise SkillError("--quality-max-tokens must be between 1 and 256")
    if not isinstance(args.benchmark_prompt, str) or not args.benchmark_prompt.strip():
        raise SkillError("--benchmark-prompt must be a non-empty string")
    if len(args.benchmark_prompt) > quality_gate.MAX_PROMPT_CHARS:
        raise SkillError(
            f"--benchmark-prompt must be at most {quality_gate.MAX_PROMPT_CHARS} characters"
        )
    if not isinstance(args.seed, int) or isinstance(args.seed, bool) or args.seed < 0:
        raise SkillError("--seed must be a non-negative integer")
    quality_args = argparse.Namespace(
        max_perplexity_ratio=_finite(
            args.max_perplexity_ratio,
            "--max-perplexity-ratio",
        ),
        min_firsttoken_agreement=_finite(
            args.min_firsttoken_agreement,
            "--min-firsttoken-agreement",
        ),
        max_degenerate_output_rate=_finite(
            args.max_degenerate_output_rate,
            "--max-degenerate-output-rate",
        ),
        max_tokens=args.quality_max_tokens,
    )
    quality_gate.validate_thresholds(quality_args)
    return configs


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise SkillError(f"{label} does not exist: {path}") from exc
    if not path.is_file() or metadata.st_size > MAX_SOURCE_CONFIG_BYTES:
        raise SkillError(f"{label} must be a file no larger than {MAX_SOURCE_CONFIG_BYTES} bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"could not read {label} as strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SkillError(f"{label} must contain a JSON object")
    return value


def validate_source_model(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    try:
        if not path.is_dir():
            raise SkillError(f"--model must be an existing local directory: {path}")
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"could not resolve --model: {path}: {exc}") from exc
    config = _strict_object(resolved / "config.json", "--model config.json")
    if config.get("model_file") is not None or config.get("auto_map") is not None:
        raise SkillError(
            "--model requests custom model code; only built-in mlx_lm models are supported"
        )
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise SkillError("--model config.json is missing model_type")
    if quality_gate._quantization_config(config) is not None:
        raise SkillError("--model must be an unquantized source model")
    weights = sorted(resolved.glob("model*.safetensors"))
    if not weights or any(not weight.is_file() for weight in weights):
        raise SkillError("--model must contain model*.safetensors weights")
    return resolved, config


def prepare_work_dir(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise SkillError(f"--work-dir must be a local non-symlink directory: {path}")
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"could not prepare --work-dir {path}: {exc}") from exc
    if resolved == source:
        raise SkillError("--work-dir must be different from --model")
    return resolved


def load_runtime() -> Runtime:
    try:
        import mlx.core as mx
        from mlx_lm import convert, load
        from mlx_lm.generate import generate_step
    except (ImportError, ModuleNotFoundError) as exc:
        raise SkillError(
            "optimize_port.py requires optional packages 'mlx' and 'mlx-lm' on Apple Silicon; "
            "install them with: python3 -m pip install mlx mlx-lm"
        ) from exc
    try:
        metal_available = bool(mx.metal.is_available())
    except (AttributeError, RuntimeError) as exc:
        raise SkillError("the installed MLX runtime does not expose a usable Metal backend") from exc
    if not metal_available:
        raise SkillError("optimize_port.py requires a real MLX Metal backend on Apple Silicon")
    return Runtime(mx=mx, load=load, convert=convert, generate_step=generate_step)


def _memory_call(mx: Any, name: str) -> int:
    function = getattr(mx, name, None)
    if function is None:
        function = getattr(getattr(mx, "metal", None), name, None)
    if function is None:
        raise SkillError(f"installed MLX does not expose {name}()")
    result = function()
    return int(result) if result is not None else 0


def _reset_measurement_state(runtime: Runtime) -> None:
    gc.collect()
    _memory_call(runtime.mx, "clear_cache")
    _memory_call(runtime.mx, "reset_peak_memory")


def _one_decode_run(
    runtime: Runtime,
    model: Any,
    prompt_tokens: list[int],
    decode_tokens: int,
) -> tuple[float, list[int]]:
    generator = runtime.generate_step(
        runtime.mx.array(prompt_tokens, dtype=runtime.mx.uint32),
        model,
        max_tokens=decode_tokens,
    )
    try:
        first_token, _ = next(generator)
    except StopIteration as exc:
        raise SkillError("mlx_lm.generate_step produced no benchmark tokens") from exc
    runtime.mx.synchronize()
    started = time.perf_counter()
    output = [int(first_token)]
    output.extend(int(token) for token, _ in generator)
    runtime.mx.synchronize()
    elapsed = time.perf_counter() - started
    expected_decode_tokens = decode_tokens - 1
    if len(output) != decode_tokens:
        raise SkillError(
            f"decode benchmark expected {decode_tokens} tokens but generated {len(output)}"
        )
    if elapsed <= 0:
        raise SkillError("decode benchmark measured a non-positive duration")
    return expected_decode_tokens / elapsed, output


def measure_decode(
    runtime: Runtime,
    model_path: Path,
    *,
    prompt: str,
    warmup_runs: int,
    timed_runs: int,
    decode_tokens: int,
    seed: int,
) -> dict[str, Any]:
    _reset_measurement_state(runtime)
    runtime.mx.random.seed(seed)
    try:
        model, tokenizer = runtime.load(str(model_path))
    except Exception as exc:
        raise SkillError(f"mlx_lm.load could not load local model {model_path}: {exc}") from exc
    prompt_tokens = quality_gate.tokenize_with_bos(tokenizer, prompt)
    for _ in range(warmup_runs):
        _one_decode_run(runtime, model, prompt_tokens, decode_tokens)
        gc.collect()
        _memory_call(runtime.mx, "clear_cache")

    throughputs: list[float] = []
    observed_outputs: list[list[int]] = []
    for _ in range(timed_runs):
        throughput, output = _one_decode_run(runtime, model, prompt_tokens, decode_tokens)
        throughputs.append(throughput)
        observed_outputs.append(output)
        gc.collect()
        _memory_call(runtime.mx, "clear_cache")
    if any(output != observed_outputs[0] for output in observed_outputs[1:]):
        raise SkillError("greedy benchmark outputs changed across timed runs")
    median = float(stable_median(throughputs))
    mean = stable_mean(throughputs)
    peak_memory = _memory_call(runtime.mx, "get_peak_memory")
    active_memory = _memory_call(runtime.mx, "get_active_memory")
    result = {
        "decode_tokens_per_second_median": median,
        "decode_tokens_per_second_mean": mean,
        "decode_tokens_per_second_cv": stable_pstdev(throughputs) / mean if mean else None,
        "decode_tokens_per_second_runs": throughputs,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "generated_tokens_per_run": decode_tokens,
        "timed_decode_tokens_per_run": decode_tokens - 1,
        "prompt_tokens_with_bos": len(prompt_tokens),
        "peak_memory_bytes": peak_memory,
        "peak_memory_gib": peak_memory / (1024**3),
        "active_memory_bytes_after_runs": active_memory,
        "greedy_output_token_ids": observed_outputs[0],
    }
    del model, tokenizer
    gc.collect()
    _memory_call(runtime.mx, "clear_cache")
    return result


def _artifact_quantization(path: Path) -> Any:
    return quality_gate._quantization_config(
        _strict_object(path / "config.json", f"artifact config.json at {path}")
    )


def _validate_reused_artifact(
    path: Path,
    source_model_type: str,
    config: OptimizationConfig | None,
) -> None:
    _, artifact_config = quality_gate.validate_model_directory(
        str(path),
        f"existing artifact {path.name}",
    )
    if artifact_config["model_type"] != source_model_type:
        raise SkillError(f"existing artifact {path} has a different model_type")
    quantization = quality_gate._quantization_config(artifact_config)
    if config is None:
        if quantization is not None:
            raise SkillError(f"existing bf16 artifact {path} is quantized")
        return
    if not isinstance(quantization, dict):
        raise SkillError(f"existing candidate {path} has no inspectable quantization config")
    expected = {
        "bits": config.bits,
        "group_size": config.group_size,
        "mode": config.mode,
    }
    observed = {key: quantization.get(key) for key in expected}
    if observed != expected:
        raise SkillError(
            f"existing candidate {path} quantization mismatch: expected {expected}, observed {observed}"
        )


def convert_artifact(
    runtime: Runtime,
    source: Path,
    destination: Path,
    source_model_type: str,
    config: OptimizationConfig | None,
    *,
    reuse_existing: bool,
) -> dict[str, Any]:
    if destination.exists():
        if not reuse_existing:
            raise SkillError(
                f"conversion destination already exists: {destination}; remove it or pass --reuse-existing"
            )
        _validate_reused_artifact(destination, source_model_type, config)
        return {"path": str(destination), "reused": True, "conversion_seconds": None}
    print(f"Converting {destination.name} -> {destination}", file=sys.stderr)
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime.convert(
                hf_path=str(source),
                mlx_path=str(destination),
                quantize=config is not None,
                q_group_size=config.group_size if config else None,
                q_bits=config.bits if config else None,
                q_mode=config.mode if config else "affine",
                dtype="bfloat16",
                trust_remote_code=False,
            )
    except Exception as exc:
        raise SkillError(f"mlx_lm.convert failed for {destination.name}: {exc}") from exc
    _validate_reused_artifact(destination, source_model_type, config)
    return {
        "path": str(destination),
        "reused": False,
        "conversion_seconds": time.perf_counter() - started,
    }


def capture_reference_quality(
    reference_path: Path,
    *,
    prompts_file: str | None,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    workload = quality_gate.load_workload(prompts_file)
    runtime = quality_gate.load_runtime()
    runtime.mx.random.seed(seed)
    measured, held_tokens, _ = quality_gate._measure_model(
        runtime,
        reference_path,
        None,
        None,
        workload,
        max_tokens,
    )
    perplexity = measured["perplexity"]["perplexity"]
    if (
        perplexity is None
        or perplexity < 1.0
        or perplexity > quality_gate.MAX_SANE_REFERENCE_PERPLEXITY
    ):
        raise SkillError(
            "reference perplexity sanity check failed before the sweep: expected a finite value "
            f"between 1 and {quality_gate.MAX_SANE_REFERENCE_PERPLEXITY:g}, observed {perplexity!r}"
        )
    outputs = []
    for index, generation in enumerate(measured["generations"]):
        outputs.append({
            "prompt_index": index,
            "prompt_sha256": _sha256_text(workload["prompts"][index]),
            **generation,
        })
    return {
        "workload_source": workload["source"],
        "held_text_sha256": _sha256_text(workload["held_text"]),
        "held_text_tokens_with_bos": len(held_tokens),
        "perplexity": measured["perplexity"],
        "greedy_outputs": outputs,
    }


def run_quality_gate(
    reference_path: Path,
    candidate_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gate_args = argparse.Namespace(
        reference=str(reference_path),
        candidate=str(candidate_path),
        prompts_file=args.prompts_file,
        max_perplexity_ratio=args.max_perplexity_ratio,
        min_firsttoken_agreement=args.min_firsttoken_agreement,
        max_degenerate_output_rate=args.max_degenerate_output_rate,
        max_tokens=args.quality_max_tokens,
    )
    return quality_gate.run_gate(gate_args)


def summarize_quality(gate_report: dict[str, Any]) -> dict[str, Any]:
    if gate_report.get("verdict") not in {"pass", "fail"}:
        raise SkillError("quant_quality_gate returned an invalid verdict")
    perplexity = gate_report["metrics"]["perplexity"]
    generation = gate_report["metrics"]["greedy_generation"]
    degeneration = gate_report["metrics"]["degenerate_output"]
    exact_output_divergence = [
        {
            "prompt_index": comparison["prompt_index"],
            "prompt_sha256": comparison["prompt_sha256"],
            "exact_match": comparison["exact_match"],
            "firsttoken_agreement": comparison["firsttoken_agreement"],
            "first_divergence_token_index": comparison["first_divergence_token_index"],
        }
        for comparison in generation["comparisons"]
    ]
    return {
        "verdict": gate_report["verdict"],
        "passed": bool(gate_report["passed"]),
        "perplexity_reference": perplexity["reference"],
        "perplexity_candidate": perplexity["candidate"],
        "perplexity_ratio": perplexity["ratio"],
        "exact_match_count": generation["exact_match_count"],
        "exact_match_rate": generation["exact_match_rate"],
        "firsttoken_agreement_rate": generation["firsttoken_agreement_rate"],
        "prompt_count": generation["prompt_count"],
        "candidate_degenerate_count": degeneration["candidate_count"],
        "candidate_only_degenerate_count": degeneration["candidate_only_count"],
        "candidate_only_degenerate_rate": degeneration["candidate_only_rate"],
        "exact_output_divergence": exact_output_divergence,
        "checks": gate_report["checks"],
        "quality_gate_run_id": gate_report["run_id"],
    }


def _selection_score(candidate: dict[str, Any]) -> float:
    performance = candidate["performance"]
    peak = performance["peak_memory_bytes"]
    if not isinstance(peak, int) or peak <= 0:
        raise SkillError("candidate peak memory must be a positive integer")
    return performance["decode_tokens_per_second_median"] / peak


def recommend_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise SkillError("recommendation requires at least one candidate")
    passing = [candidate for candidate in candidates if candidate["quality"]["passed"]]
    if passing:
        chosen = max(
            passing,
            key=lambda candidate: (
                _selection_score(candidate),
                candidate["performance"]["decode_tokens_per_second_median"],
                -candidate["performance"]["peak_memory_bytes"],
                candidate["config"]["id"],
            ),
        )
        return {
            "recommended_config_id": chosen["config"]["id"],
            "quality_held": True,
            "objective": "maximum measured decode tokens per second per peak Metal byte",
            "selection_score_tokens_per_second_per_byte": _selection_score(chosen),
            "fallback_config_id": None,
            "warning": None,
            "reason": (
                "Selected the highest throughput-per-peak-memory candidate among configurations "
                "that passed every configured quality check."
            ),
        }

    def degradation_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
        quality = candidate["quality"]
        ratio = quality["perplexity_ratio"]
        finite_ratio = float(ratio) if isinstance(ratio, (int, float)) and math.isfinite(ratio) else math.inf
        return (
            float(quality["candidate_only_degenerate_rate"]),
            finite_ratio,
            -float(quality["exact_match_rate"]),
            -_selection_score(candidate),
        )

    fallback = min(candidates, key=degradation_key)
    return {
        "recommended_config_id": None,
        "quality_held": False,
        "objective": "maximum measured decode tokens per second per peak Metal byte",
        "selection_score_tokens_per_second_per_byte": None,
        "fallback_config_id": fallback["config"]["id"],
        "warning": (
            "No quantized configuration passed the quality bar. The fallback is the "
            "least-degrading measured candidate and is not a quality-held recommendation."
        ),
        "reason": "No candidate passed every configured quality check.",
    }


def _performance_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_tps = baseline["decode_tokens_per_second_median"]
    candidate_tps = candidate["decode_tokens_per_second_median"]
    baseline_memory = baseline["peak_memory_bytes"]
    candidate_memory = candidate["peak_memory_bytes"]
    return {
        "speedup_vs_bf16": candidate_tps / baseline_tps,
        "memory_reduction_bytes_vs_bf16": baseline_memory - candidate_memory,
        "memory_reduction_fraction_vs_bf16": 1.0 - (candidate_memory / baseline_memory),
        "peak_memory_ratio_vs_bf16": candidate_memory / baseline_memory,
    }


def _hardware_metadata(runtime: Runtime) -> dict[str, Any]:
    try:
        device_info = runtime.mx.device_info()
    except (AttributeError, RuntimeError):
        device_info = runtime.mx.metal.device_info()
    allowed = {
        "device_name",
        "architecture",
        "memory_size",
        "max_recommended_working_set_size",
        "max_buffer_length",
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "metal_available": bool(runtime.mx.metal.is_available()),
        "mlx_default_device": str(runtime.mx.default_device()),
        "device": {key: value for key, value in device_info.items() if key in allowed},
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_descriptor(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "model_type": config["model_type"],
        "name_or_path": config.get("_name_or_path"),
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "weight_files": [
            {"name": weight.name, "size_bytes": weight.stat().st_size}
            for weight in sorted(path.glob("model*.safetensors"))
        ],
    }


def _contrast(
    candidates: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    by_id = {candidate["config"]["id"]: candidate for candidate in candidates}
    naive = by_id.get("4bit-g64")
    optimal_id = recommendation["recommended_config_id"]
    optimal = by_id.get(optimal_id) if optimal_id else None
    result: dict[str, Any] = {
        "naive_default_config_id": "4bit-g64" if naive else None,
        "naive_default_definition": "mlx_lm.convert affine 4-bit group-size 64 default",
        "structured_optimal_config_id": optimal_id,
        "quality_held": optimal is not None,
    }
    if naive is not None:
        result["naive_default"] = {
            "quality_verdict": naive["quality"]["verdict"],
            "decode_tokens_per_second_median": naive["performance"][
                "decode_tokens_per_second_median"
            ],
            "peak_memory_bytes": naive["performance"]["peak_memory_bytes"],
        }
    if optimal is not None:
        result["structured_optimal"] = {
            "quality_verdict": optimal["quality"]["verdict"],
            "decode_tokens_per_second_median": optimal["performance"][
                "decode_tokens_per_second_median"
            ],
            "peak_memory_bytes": optimal["performance"]["peak_memory_bytes"],
        }
    return result


def run_optimization(
    args: argparse.Namespace,
    *,
    runtime: Runtime | None = None,
) -> dict[str, Any]:
    configs = validate_args(args)
    source, source_config = validate_source_model(args.model)
    runtime = runtime or load_runtime()
    work_dir = prepare_work_dir(args.work_dir, source)
    runtime.mx.random.seed(args.seed)

    baseline_path = work_dir / "bf16"
    baseline_conversion = convert_artifact(
        runtime,
        source,
        baseline_path,
        source_config["model_type"],
        None,
        reuse_existing=args.reuse_existing,
    )
    print("Measuring bf16 baseline", file=sys.stderr)
    baseline_performance = measure_decode(
        runtime,
        baseline_path,
        prompt=args.benchmark_prompt,
        warmup_runs=args.warmup_runs,
        timed_runs=args.runs,
        decode_tokens=args.decode_tokens,
        seed=args.seed,
    )
    reference_quality = capture_reference_quality(
        baseline_path,
        prompts_file=args.prompts_file,
        max_tokens=args.quality_max_tokens,
        seed=args.seed,
    )

    candidates: list[dict[str, Any]] = []
    for config in configs:
        candidate_path = work_dir / config.config_id
        conversion = convert_artifact(
            runtime,
            source,
            candidate_path,
            source_config["model_type"],
            config,
            reuse_existing=args.reuse_existing,
        )
        print(f"Measuring {config.config_id}", file=sys.stderr)
        performance = measure_decode(
            runtime,
            candidate_path,
            prompt=args.benchmark_prompt,
            warmup_runs=args.warmup_runs,
            timed_runs=args.runs,
            decode_tokens=args.decode_tokens,
            seed=args.seed,
        )
        print(f"Running quality gate for {config.config_id}", file=sys.stderr)
        quality = summarize_quality(run_quality_gate(baseline_path, candidate_path, args))
        candidate = {
            "config": {
                "id": config.config_id,
                "bits": config.bits,
                "group_size": config.group_size,
                "mode": config.mode,
                "naive_default": config.naive_default,
            },
            "artifact": {
                **conversion,
                "quantization": _artifact_quantization(candidate_path),
            },
            "performance": performance,
            "comparison_vs_bf16": _performance_comparison(
                baseline_performance,
                performance,
            ),
            "quality": quality,
        }
        candidate["selection_score_tokens_per_second_per_byte"] = _selection_score(candidate)
        candidates.append(candidate)

    recommendation = recommend_candidate(candidates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "boundary": dict(BOUNDARY),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": _source_descriptor(source, source_config),
        "work_dir": str(work_dir),
        "environment": {
            "hardware": _hardware_metadata(runtime),
            "software": {
                "python": sys.version.split()[0],
                "mlx": _package_version("mlx"),
                "mlx_lm": _package_version("mlx-lm"),
            },
        },
        "methodology": {
            "config_sweep": [config.config_id for config in configs],
            "seed": args.seed,
            "benchmark": {
                "prompt_sha256": _sha256_text(args.benchmark_prompt),
                "warmup_runs": args.warmup_runs,
                "timed_runs": args.runs,
                "generated_tokens_per_run": args.decode_tokens,
                "timing_boundary": (
                    "generation after the first yielded token; excludes prefill and reports the "
                    "median of timed runs"
                ),
                "variance_note": (
                    "Decode throughput is a local observation with run-to-run variance; inspect CV "
                    "and raw timed-run values."
                ),
            },
            "quality": {
                "implementation": "quant_quality_gate.py",
                "max_perplexity_ratio": args.max_perplexity_ratio,
                "min_firsttoken_agreement": args.min_firsttoken_agreement,
                "max_candidate_only_degenerate_output_rate": args.max_degenerate_output_rate,
                "max_tokens_per_prompt": args.quality_max_tokens,
            },
            "recommendation_objective": (
                "maximum measured decode tokens per second per peak Metal byte among quality passes"
            ),
        },
        "baseline": {
            "config": {"id": "bf16", "dtype": "bfloat16", "quantized": False},
            "artifact": baseline_conversion,
            "performance": baseline_performance,
            "quality_reference": reference_quality,
        },
        "candidates": candidates,
        "recommendation": recommendation,
        "naive_default_vs_structured_optimal": _contrast(candidates, recommendation),
    }
    strict_json(report)
    return report


def _format_number(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def human_table(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    baseline_performance = baseline["performance"]
    reference_outputs = baseline["quality_reference"]["greedy_outputs"]
    rows = [{
        "config": "bf16",
        "tps": baseline_performance["decode_tokens_per_second_median"],
        "cv": baseline_performance["decode_tokens_per_second_cv"],
        "speedup": 1.0,
        "memory": baseline_performance["peak_memory_gib"],
        "reduction": 0.0,
        "ppl": 1.0,
        "exact": f"{len(reference_outputs)}/{len(reference_outputs)}",
        "verdict": "reference",
    }]
    for candidate in report["candidates"]:
        performance = candidate["performance"]
        comparison = candidate["comparison_vs_bf16"]
        quality = candidate["quality"]
        rows.append({
            "config": candidate["config"]["id"],
            "tps": performance["decode_tokens_per_second_median"],
            "cv": performance["decode_tokens_per_second_cv"],
            "speedup": comparison["speedup_vs_bf16"],
            "memory": performance["peak_memory_gib"],
            "reduction": comparison["memory_reduction_fraction_vs_bf16"],
            "ppl": quality["perplexity_ratio"],
            "exact": f"{quality['exact_match_count']}/{quality['prompt_count']}",
            "verdict": quality["verdict"],
        })
    lines = [
        "",
        "Local optimization observations (not promotable claims)",
        "Config       tok/s    CV      speedup  peak GiB  mem red.  ppl ratio  exact  quality",
        "-----------  -------  ------  -------  --------  --------  ---------  -----  -------",
    ]
    for row in rows:
        lines.append(
            f"{row['config']:<11}  {_format_number(row['tps']):>7}  "
            f"{_format_number(row['cv'], 3):>6}  {_format_number(row['speedup']):>7}  "
            f"{_format_number(row['memory'], 3):>8}  "
            f"{_format_number(100 * row['reduction'], 1):>7}%  "
            f"{_format_number(row['ppl'], 4):>9}  {row['exact']:>5}  {row['verdict']}"
        )
    recommendation = report["recommendation"]
    if recommendation["recommended_config_id"]:
        lines.append(
            f"Recommendation: {recommendation['recommended_config_id']} (quality held; "
            "best measured throughput per peak Metal byte)."
        )
    else:
        lines.append("Recommendation: none passed the configured quality bar.")
        lines.append(
            f"Warning fallback: {recommendation['fallback_config_id']} is least-degrading, "
            "not quality-held."
        )
    lines.append(BOUNDARY["note"])
    return "\n".join(lines) + "\n"


def strict_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise SkillError(f"optimization report is not strict JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    try:
        report = run_optimization(parse_args(argv))
        sys.stderr.write(human_table(report))
        sys.stdout.write(strict_json(report))
        return 0 if report["recommendation"]["recommended_config_id"] else 1
    except (SkillError, OSError, RuntimeError, ValueError) as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "kind": DIAGNOSTIC_KIND,
            "boundary": dict(BOUNDARY),
            "status": "error",
            "error": {"type": "SkillError", "message": str(exc)},
        }
        sys.stdout.write(strict_json(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
