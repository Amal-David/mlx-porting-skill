#!/usr/bin/env python3
"""Measure quantized MLX-LM quality without creating promotable evidence."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError


SCHEMA_VERSION = 1
DIAGNOSTIC_KIND = "mlx-quantization-quality-gate"
MAX_PROMPTS_BYTES = 1024 * 1024
MAX_PROMPTS = 32
MAX_PROMPT_CHARS = 4096
MAX_HELD_TEXT_CHARS = 32_768
MAX_MODEL_CONFIG_BYTES = 2 * 1024 * 1024
MAX_SANE_REFERENCE_PERPLEXITY = 512.0
DEFAULT_MAX_TOKENS = 32

BUILTIN_HELD_TEXT = (
    "Reliable software is built by checking assumptions against evidence. "
    "A useful measurement states what was tested, keeps the workload fixed, and records enough "
    "detail for another person to repeat it. When a result looks surprising, the first step is "
    "to verify the inputs and the scoring method before drawing a conclusion. Careful diagnostics "
    "make failures easier to understand and successful changes safer to keep."
)
BUILTIN_PROMPTS = (
    "Complete this sentence with one concise clause: Careful measurements matter because",
    "What is 17 multiplied by 6? Answer with only the number.",
    "Write one sentence describing a quiet library.",
    "The capital of France is",
)


@dataclass(frozen=True)
class Runtime:
    mx: Any
    nn: Any
    load: Any
    generate_step: Any
    make_prompt_cache: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a locally converted quantized MLX-LM model with its bf16 reference. "
            "This is a user diagnostic only and never writes benchmark receipts or claims."
        )
    )
    parser.add_argument("--reference", required=True, help="Local bf16 MLX-LM model directory")
    parser.add_argument("--candidate", required=True, help="Local quantized MLX-LM model directory")
    parser.add_argument(
        "--prompts-file",
        help=(
            "Optional strict-JSON prompt array, or object with prompts and held_text; "
            "the built-in workload is used otherwise"
        ),
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
        default=0.75,
        help="Minimum fraction of prompts with the same first greedy token (default: 0.75)",
    )
    parser.add_argument(
        "--max-degenerate-output-rate",
        type=float,
        default=0.0,
        help=(
            "Maximum fraction of prompts degenerate only for the candidate "
            "(default: 0.0)"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Greedy tokens generated per prompt (default: {DEFAULT_MAX_TOKENS})",
    )
    return parser.parse_args(argv)


def _finite_number(value: float, flag: str) -> float:
    if not math.isfinite(value):
        raise SkillError(f"{flag} must be finite")
    return value


def validate_thresholds(args: argparse.Namespace) -> None:
    max_ratio = _finite_number(args.max_perplexity_ratio, "--max-perplexity-ratio")
    min_first = _finite_number(args.min_firsttoken_agreement, "--min-firsttoken-agreement")
    max_degenerate = _finite_number(
        args.max_degenerate_output_rate,
        "--max-degenerate-output-rate",
    )
    if max_ratio <= 0:
        raise SkillError("--max-perplexity-ratio must be greater than zero")
    if not 0.0 <= min_first <= 1.0:
        raise SkillError("--min-firsttoken-agreement must be between 0 and 1")
    if not 0.0 <= max_degenerate <= 1.0:
        raise SkillError("--max-degenerate-output-rate must be between 0 and 1")
    if isinstance(args.max_tokens, bool) or not 1 <= args.max_tokens <= 256:
        raise SkillError("--max-tokens must be between 1 and 256")


def _read_model_config(path: Path, label: str) -> dict[str, Any]:
    config_path = path / "config.json"
    try:
        metadata = config_path.lstat()
    except OSError as exc:
        raise SkillError(f"{label} is missing config.json: {config_path}") from exc
    if config_path.is_symlink() or not config_path.is_file():
        raise SkillError(f"{label} config.json must be a regular non-symlink file")
    if metadata.st_size > MAX_MODEL_CONFIG_BYTES:
        raise SkillError(f"{label} config.json exceeds {MAX_MODEL_CONFIG_BYTES} bytes")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"could not read {label} config.json as strict JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise SkillError(f"{label} config.json must contain a JSON object")
    if config.get("model_file") is not None or config.get("auto_map") is not None:
        raise SkillError(
            f"{label} requests custom model code; this diagnostic supports built-in mlx_lm models only"
        )
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise SkillError(f"{label} config.json is missing model_type")
    return config


def validate_model_directory(value: str, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    try:
        if path.is_symlink() or not path.is_dir():
            raise SkillError(f"{label} must be an existing local non-symlink directory: {path}")
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SkillError(f"could not resolve {label}: {path}: {exc}") from exc
    config = _read_model_config(resolved, label)
    weights = sorted(resolved.glob("model*.safetensors"))
    if not weights or any(weight.is_symlink() or not weight.is_file() for weight in weights):
        raise SkillError(f"{label} must contain regular non-symlink model*.safetensors weights")
    return resolved, config


def load_workload(value: str | None) -> dict[str, Any]:
    if value is None:
        return {
            "source": "builtin-v1",
            "held_text": BUILTIN_HELD_TEXT,
            "prompts": list(BUILTIN_PROMPTS),
        }
    path = Path(value).expanduser()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SkillError(f"--prompts-file does not exist: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise SkillError("--prompts-file must be a regular non-symlink file")
    if metadata.st_size > MAX_PROMPTS_BYTES:
        raise SkillError(f"--prompts-file exceeds {MAX_PROMPTS_BYTES} bytes")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"could not read --prompts-file as strict JSON: {exc}") from exc
    held_text = BUILTIN_HELD_TEXT
    if isinstance(payload, list):
        prompts = payload
    elif isinstance(payload, dict):
        unknown = sorted(set(payload) - {"prompts", "held_text"})
        if unknown:
            raise SkillError(f"--prompts-file has unknown fields: {', '.join(unknown)}")
        prompts = payload.get("prompts")
        held_text = payload.get("held_text", held_text)
    else:
        raise SkillError("--prompts-file must contain a JSON array or object")
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS:
        raise SkillError(f"prompts must be a JSON array with 1 to {MAX_PROMPTS} entries")
    if any(
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt) > MAX_PROMPT_CHARS
        for prompt in prompts
    ):
        raise SkillError(
            f"each prompt must be a non-empty string no longer than {MAX_PROMPT_CHARS} characters"
        )
    if (
        not isinstance(held_text, str)
        or len(held_text.strip()) < 20
        or len(held_text) > MAX_HELD_TEXT_CHARS
    ):
        raise SkillError(
            "held_text must be a string between 20 and "
            f"{MAX_HELD_TEXT_CHARS} characters"
        )
    return {
        "source": str(path.resolve()),
        "held_text": held_text,
        "prompts": prompts,
    }


def load_runtime() -> Runtime:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
    except (ImportError, ModuleNotFoundError) as exc:
        raise SkillError(
            "quant_quality_gate.py requires optional packages 'mlx' and 'mlx-lm' on Apple "
            "Silicon; install them with: python3 -m pip install mlx mlx-lm"
        ) from exc
    return Runtime(
        mx=mx,
        nn=nn,
        load=load,
        generate_step=generate_step,
        make_prompt_cache=make_prompt_cache,
    )


def tokenize_with_bos(tokenizer: Any, text: str) -> list[int]:
    try:
        bos_token = getattr(tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not text.startswith(bos_token)
        token_ids = [
            int(token)
            for token in tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ]
    except Exception as exc:
        raise SkillError(f"tokenizer could not encode the diagnostic workload: {exc}") from exc
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        bos_token_id = int(bos_token_id)
        if not token_ids or token_ids[0] != bos_token_id:
            token_ids.insert(0, bos_token_id)
    if len(token_ids) < 2:
        raise SkillError("tokenizer produced fewer than two tokens for the held text")
    return token_ids


def token_level_perplexity(
    runtime: Runtime,
    model: Any,
    token_ids: list[int],
) -> dict[str, Any]:
    """Score actual next tokens through the same cached model path as generation."""
    mx = runtime.mx
    inputs = mx.array(token_ids[:-1], dtype=mx.uint32)[None]
    targets = mx.array(token_ids[1:], dtype=mx.uint32)[None]
    cache = runtime.make_prompt_cache(model)
    logits = model(inputs, cache=cache)
    if len(logits.shape) != 3 or logits.shape[:2] != inputs.shape:
        raise SkillError(
            "mlx_lm model forward returned unexpected logits shape "
            f"{tuple(logits.shape)} for inputs {tuple(inputs.shape)}"
        )
    if logits.shape[-1] <= max(token_ids):
        raise SkillError("tokenizer produced a token id outside the model vocabulary")
    log_probs = runtime.nn.log_softmax(logits.astype(mx.float32), axis=-1)
    selected = mx.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    nll = -mx.mean(selected)
    nll_value = float(nll.item())
    finite = math.isfinite(nll_value) and nll_value >= 0
    perplexity = math.exp(nll_value) if finite and nll_value < math.log(sys.float_info.max) else None
    if perplexity is not None and not math.isfinite(perplexity):
        perplexity = None
    del cache, logits, log_probs, selected, nll
    mx.clear_cache()
    return {
        "nll": nll_value if math.isfinite(nll_value) else None,
        "perplexity": perplexity,
        "scored_tokens": len(token_ids) - 1,
        "finite": perplexity is not None,
    }


def greedy_tokens(
    runtime: Runtime,
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
) -> list[int]:
    mx = runtime.mx
    eos_token_ids = {
        int(token)
        for token in getattr(tokenizer, "eos_token_ids", set())
        if token is not None
    }
    generated: list[int] = []
    for token, _ in runtime.generate_step(
        mx.array(prompt_tokens, dtype=mx.uint32),
        model,
        max_tokens=max_tokens,
    ):
        token_id = int(token)
        generated.append(token_id)
        if token_id in eos_token_ids:
            break
    mx.clear_cache()
    return generated


def _repeated_ngram_fraction(tokens: list[int], width: int) -> float:
    if len(tokens) < width:
        return 0.0
    ngrams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def _longest_token_run(tokens: list[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def _character_cycle(text: str) -> dict[str, int] | None:
    compact = "".join(character.lower() for character in text if character.isalnum())
    best: dict[str, int] | None = None
    for start in range(len(compact)):
        for period in range(2, min(32, (len(compact) - start) // 2) + 1):
            pattern = compact[start : start + period]
            repeats = 1
            while compact[
                start + repeats * period : start + (repeats + 1) * period
            ] == pattern:
                repeats += 1
            coverage = repeats * period
            if repeats >= 2 and coverage >= 12 and (
                best is None or coverage > best["coverage_chars"]
            ):
                best = {
                    "start_char": start,
                    "period_chars": period,
                    "repeats": repeats,
                    "coverage_chars": coverage,
                }
    return best


def detect_degenerate_output(tokens: list[int], text: str) -> dict[str, Any]:
    bigram_fraction = _repeated_ngram_fraction(tokens, 2)
    trigram_fraction = _repeated_ngram_fraction(tokens, 3)
    longest_run = _longest_token_run(tokens)
    character_cycle = _character_cycle(text)
    reasons: list[str] = []
    if len(tokens) >= 8 and longest_run >= 6:
        reasons.append("single-token-run")
    if len(tokens) >= 10 and bigram_fraction >= 0.60:
        reasons.append("repeated-bigrams")
    if len(tokens) >= 10 and trigram_fraction >= 0.50:
        reasons.append("repeated-trigrams")
    if character_cycle is not None:
        reasons.append("repeated-character-cycle")
    return {
        "detected": bool(reasons),
        "reasons": reasons,
        "longest_token_run": longest_run,
        "repeated_bigram_fraction": bigram_fraction,
        "repeated_trigram_fraction": trigram_fraction,
        "character_cycle": character_cycle,
    }


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    for index in range(max(len(reference), len(candidate))):
        if index >= len(reference) or index >= len(candidate) or reference[index] != candidate[index]:
            return index
    return None


def evaluate_thresholds(
    *,
    reference_perplexity: float,
    candidate_perplexity: float | None,
    firsttoken_agreement_count: int,
    prompt_count: int,
    candidate_only_degenerate_count: int,
    max_perplexity_ratio: float,
    min_firsttoken_agreement: float,
    max_degenerate_output_rate: float,
) -> dict[str, Any]:
    if prompt_count <= 0:
        raise SkillError("threshold evaluation requires at least one prompt")
    firsttoken_agreement = firsttoken_agreement_count / prompt_count
    candidate_only_degenerate_rate = candidate_only_degenerate_count / prompt_count
    ratio = (
        candidate_perplexity / reference_perplexity
        if candidate_perplexity is not None
        and math.isfinite(candidate_perplexity)
        and reference_perplexity > 0
        else None
    )
    checks = {
        "perplexity_ratio": {
            "passed": ratio is not None and ratio <= max_perplexity_ratio,
            "value": ratio,
            "maximum": max_perplexity_ratio,
        },
        "firsttoken_agreement": {
            "passed": firsttoken_agreement >= min_firsttoken_agreement,
            "value": firsttoken_agreement,
            "minimum": min_firsttoken_agreement,
        },
        "candidate_only_degenerate_output_rate": {
            "passed": candidate_only_degenerate_rate <= max_degenerate_output_rate,
            "value": candidate_only_degenerate_rate,
            "maximum": max_degenerate_output_rate,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "perplexity_ratio": ratio,
        "firsttoken_agreement": firsttoken_agreement,
        "candidate_only_degenerate_output_rate": candidate_only_degenerate_rate,
        "checks": checks,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_descriptor(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    weight_files = sorted(path.glob("model*.safetensors"))
    descriptor = {
        "path": str(path),
        "model_type": config["model_type"],
        "quantization": _quantization_config(config),
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "weight_files": [
            {"name": item.name, "size_bytes": item.stat().st_size}
            for item in weight_files
        ],
    }
    return descriptor


def _quantization_config(config: dict[str, Any]) -> Any:
    for container in (config, config.get("text_config")):
        if not isinstance(container, dict):
            continue
        for key in ("quantization", "quantization_config"):
            value = container.get(key)
            if value:
                return value
    return None


def _measure_model(
    runtime: Runtime,
    path: Path,
    held_tokens: list[int] | None,
    prompt_tokens: list[list[int]] | None,
    workload: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], list[int], list[list[int]]]:
    try:
        model, tokenizer = runtime.load(str(path))
    except Exception as exc:
        raise SkillError(f"mlx_lm.load could not load local model {path}: {exc}") from exc
    encoded_held = tokenize_with_bos(tokenizer, workload["held_text"])
    encoded_prompts = [tokenize_with_bos(tokenizer, prompt) for prompt in workload["prompts"]]
    if held_tokens is not None and encoded_held != held_tokens:
        raise SkillError("candidate tokenizer does not match the reference tokenizer on held_text")
    if prompt_tokens is not None and encoded_prompts != prompt_tokens:
        raise SkillError("candidate tokenizer does not match the reference tokenizer on prompts")
    perplexity = token_level_perplexity(runtime, model, encoded_held)
    generations: list[dict[str, Any]] = []
    for index, tokens in enumerate(encoded_prompts):
        output_tokens = greedy_tokens(runtime, model, tokenizer, tokens, max_tokens)
        try:
            text = tokenizer.decode(output_tokens, skip_special_tokens=False)
        except Exception as exc:
            raise SkillError(f"tokenizer could not decode generated prompt {index}: {exc}") from exc
        generations.append({
            "token_ids": output_tokens,
            "text": text,
            "degenerate": detect_degenerate_output(output_tokens, text),
        })
    del model, tokenizer
    gc.collect()
    runtime.mx.clear_cache()
    return {"perplexity": perplexity, "generations": generations}, encoded_held, encoded_prompts


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    validate_thresholds(args)
    reference_path, reference_config = validate_model_directory(args.reference, "--reference")
    candidate_path, candidate_config = validate_model_directory(args.candidate, "--candidate")
    if reference_path == candidate_path:
        raise SkillError("--reference and --candidate must be different directories")
    if reference_config["model_type"] != candidate_config["model_type"]:
        raise SkillError("--reference and --candidate must declare the same model_type")
    if _quantization_config(reference_config) is not None:
        raise SkillError("--reference must be an unquantized bf16/float MLX-LM model")
    if _quantization_config(candidate_config) is None:
        raise SkillError("--candidate must declare an mlx_lm quantization configuration")
    workload = load_workload(args.prompts_file)
    runtime = load_runtime()

    reference, held_tokens, prompt_tokens = _measure_model(
        runtime,
        reference_path,
        None,
        None,
        workload,
        args.max_tokens,
    )
    reference_ppl = reference["perplexity"]["perplexity"]
    if (
        reference_ppl is None
        or reference_ppl < 1.0
        or reference_ppl > MAX_SANE_REFERENCE_PERPLEXITY
    ):
        raise SkillError(
            "reference perplexity sanity check failed: expected a finite value between 1 and "
            f"{MAX_SANE_REFERENCE_PERPLEXITY:g} on fixed plain English, observed "
            f"{reference_ppl!r}; refusing to measure or report a candidate ratio"
        )

    candidate, _, _ = _measure_model(
        runtime,
        candidate_path,
        held_tokens,
        prompt_tokens,
        workload,
        args.max_tokens,
    )
    comparisons: list[dict[str, Any]] = []
    exact_match_count = 0
    firsttoken_agreement_count = 0
    reference_degenerate_count = 0
    candidate_degenerate_count = 0
    candidate_only_degenerate_count = 0
    for index, (reference_output, candidate_output) in enumerate(
        zip(reference["generations"], candidate["generations"])
    ):
        reference_ids = reference_output["token_ids"]
        candidate_ids = candidate_output["token_ids"]
        exact_match = reference_ids == candidate_ids
        first_agreement = bool(
            reference_ids and candidate_ids and reference_ids[0] == candidate_ids[0]
        )
        reference_degenerate = reference_output["degenerate"]["detected"]
        candidate_degenerate = candidate_output["degenerate"]["detected"]
        candidate_only_degenerate = candidate_degenerate and not reference_degenerate
        exact_match_count += int(exact_match)
        firsttoken_agreement_count += int(first_agreement)
        reference_degenerate_count += int(reference_degenerate)
        candidate_degenerate_count += int(candidate_degenerate)
        candidate_only_degenerate_count += int(candidate_only_degenerate)
        comparisons.append({
            "prompt_index": index,
            "prompt_sha256": _sha256_text(workload["prompts"][index]),
            "exact_match": exact_match,
            "firsttoken_agreement": first_agreement,
            "first_divergence_token_index": first_divergence(reference_ids, candidate_ids),
            "reference": reference_output,
            "candidate": candidate_output,
        })

    prompt_count = len(comparisons)
    threshold_result = evaluate_thresholds(
        reference_perplexity=reference_ppl,
        candidate_perplexity=candidate["perplexity"]["perplexity"],
        firsttoken_agreement_count=firsttoken_agreement_count,
        prompt_count=prompt_count,
        candidate_only_degenerate_count=candidate_only_degenerate_count,
        max_perplexity_ratio=args.max_perplexity_ratio,
        min_firsttoken_agreement=args.min_firsttoken_agreement,
        max_degenerate_output_rate=args.max_degenerate_output_rate,
    )
    versions = {
        "python": sys.version.split()[0],
        "mlx": _package_version("mlx"),
        "mlx_lm": _package_version("mlx-lm"),
    }
    run_descriptor = {
        "reference": _model_descriptor(reference_path, reference_config),
        "candidate": _model_descriptor(candidate_path, candidate_config),
        "workload": {
            "source": workload["source"],
            "held_text_sha256": _sha256_text(workload["held_text"]),
            "prompt_sha256": [_sha256_text(prompt) for prompt in workload["prompts"]],
            "max_tokens": args.max_tokens,
        },
        "thresholds": {
            "max_perplexity_ratio": args.max_perplexity_ratio,
            "min_firsttoken_agreement": args.min_firsttoken_agreement,
            "max_degenerate_output_rate": args.max_degenerate_output_rate,
        },
        "versions": versions,
    }
    run_id = _sha256_text(
        json.dumps(run_descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "boundary": {
            "classification": "user_diagnostic_only",
            "promotable_claim": False,
            "writes_sealed_evidence": False,
            "note": (
                "This measurement is not a benchmark receipt, receipt assessment, "
                "effective claim, or promotion input."
            ),
        },
        "run_id": run_id,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "pass" if threshold_result["passed"] else "fail",
        "passed": threshold_result["passed"],
        "models": {
            "reference": run_descriptor["reference"],
            "candidate": run_descriptor["candidate"],
        },
        "workload": {
            **run_descriptor["workload"],
            "held_text_tokens_with_bos": len(held_tokens),
            "prompt_count": prompt_count,
        },
        "thresholds": run_descriptor["thresholds"],
        "metrics": {
            "perplexity": {
                "reference": reference["perplexity"]["perplexity"],
                "candidate": candidate["perplexity"]["perplexity"],
                "ratio": threshold_result["perplexity_ratio"],
                "reference_nll": reference["perplexity"]["nll"],
                "candidate_nll": candidate["perplexity"]["nll"],
                "scored_tokens": reference["perplexity"]["scored_tokens"],
                "reference_sanity_ceiling": MAX_SANE_REFERENCE_PERPLEXITY,
            },
            "greedy_generation": {
                "max_tokens": args.max_tokens,
                "prompt_count": prompt_count,
                "exact_match_count": exact_match_count,
                "exact_match_rate": exact_match_count / prompt_count,
                "firsttoken_agreement_count": firsttoken_agreement_count,
                "firsttoken_agreement_rate": threshold_result["firsttoken_agreement"],
                "comparisons": comparisons,
            },
            "degenerate_output": {
                "reference_count": reference_degenerate_count,
                "candidate_count": candidate_degenerate_count,
                "candidate_only_count": candidate_only_degenerate_count,
                "candidate_only_rate": threshold_result["candidate_only_degenerate_output_rate"],
            },
        },
        "checks": threshold_result["checks"],
        "versions": versions,
    }


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
        raise SkillError(f"diagnostic result is not strict JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    try:
        report = run_gate(parse_args(argv))
        sys.stdout.write(strict_json(report))
        return 0 if report["passed"] else 1
    except (SkillError, OSError, RuntimeError, ValueError) as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "kind": DIAGNOSTIC_KIND,
            "verdict": "error",
            "error": {
                "type": "SkillError",
                "message": str(exc),
            },
        }
        sys.stdout.write(strict_json(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
