#!/usr/bin/env python3
"""Run source capture, MLX capture, and first-divergence parity in one command."""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from _capture_common import (
    DEFAULT_GENERATE_STEPS,
    DEFAULT_MAX_OUTPUT_MB,
    MAX_MANIFEST_JSON_BYTES,
    read_bounded_json,
    strict_json_bytes,
    validate_capture_limits,
    validate_capture_manifest,
    validate_input_mode,
    write_strict_json,
)
from _common import SkillError, run_process_capture, validate_comparison_tolerances


SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_TIMEOUT_SECONDS = 3600.0
LAYER_KEY_RE = re.compile(r"layer\.([0-9]+)\.hidden\Z")
ENCODER_LAYER_KEY_RE = re.compile(r"encoder\.layer\.([0-9]+)\.hidden\Z")
DECODER_LAYER_KEY_RE = re.compile(r"decoder\.layer\.([0-9]+)\.hidden\Z")
REPORT_FIELDS = {"schema_version", "ok", "inputs", "tolerances", "rungs", "summary"}
RUNG_FIELDS = {
    "position",
    "name",
    "source_key",
    "target_key",
    "exact",
    "pass",
    "max_abs",
    "max_rel",
    "cosine",
}
SUMMARY_FIELDS = {
    "status",
    "evaluated_rungs",
    "total_rungs",
    "stopped_at",
    "debug_target",
    "message",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the executable source-to-MLX first-divergence parity ladder",
    )
    parser.add_argument("--source-model", help="Pinned local Hugging Face model directory")
    parser.add_argument(
        "--mode",
        choices=("dense-decoder", "encoder", "encoder-decoder", "ssm", "asr"),
        default="dense-decoder",
        help="Parity contract (default: dense-decoder)",
    )
    parser.add_argument("--package", help="Directory produced by scaffold_port.py")
    parser.add_argument("--weights", help="Directory produced by convert_checkpoint.py")
    parser.add_argument("--output", help="Strict-JSON parity report; default: stdout")
    parser.add_argument("--prompt", action="append", help="Prompt to tokenize locally; repeatable")
    parser.add_argument("--prompts-file", action="append", help="One prompt per non-empty line")
    parser.add_argument("--token-ids", nargs="+", help="Tokenizer-free token IDs")
    parser.add_argument(
        "--attention-mask",
        nargs="+",
        help="Encoder modes only: one 0/1 value per tokenizer-free token ID",
    )
    parser.add_argument(
        "--waveform-samples",
        type=int,
        default=16000,
        help="ASR waveform fixture length (default: 16000)",
    )
    parser.add_argument("--generate-steps", type=int, default=DEFAULT_GENERATE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-dtype", action="store_true")
    parser.add_argument("--max-output-mb", type=float, default=DEFAULT_MAX_OUTPUT_MB)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--cosine-min", type=float, default=-1.0)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--allow-modified",
        action="store_true",
        help="Acknowledge drift in the user-owned scaffold package",
    )
    parser.add_argument(
        "--fault-inject-target",
        help="Validation-only: inject a deterministic error into one floating target capture",
    )
    return parser.parse_args(argv)


def _strict_fields(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SkillError(f"{label} has an invalid field set")
    return value


def build_parity_ladder(
    source_keys: list[str] | set[str],
    target_keys: list[str] | set[str],
    mode: str = "dense-decoder",
) -> list[dict[str, Any]]:
    """Map the stable same-name tensor contract into runbook ladder order."""
    source = set(source_keys)
    target = set(target_keys)
    if mode == "dense-decoder" and "encoder.embed" in source:
        mode = "encoder-decoder"

    if mode == "encoder":
        required = {"input_ids", "attention_mask", "embed", "final_hidden", "pooled"}
    elif mode == "encoder-decoder":
        required = {
            "input_ids",
            "attention_mask",
            "encoder.embed",
            "encoder.final_norm",
            "decoder_input_ids",
            "decoder.embed",
            "decoder.final_norm",
            "logits",
            "generated_token_ids",
        }
    elif mode == "asr":
        required = {"input_features", "embed", "final_hidden"}
    elif mode in {"dense-decoder", "ssm"}:
        required = {"input_ids", "embed", "final_norm", "logits", "generated_token_ids"}
    else:
        raise SkillError(f"unsupported parity mode {mode!r}")

    missing_source = sorted(required - source)
    if missing_source:
        raise SkillError("source capture is missing parity keys: " + ", ".join(missing_source))

    if mode == "encoder-decoder":
        encoder_indices = sorted(
            int(match.group(1))
            for key in source
            if (match := ENCODER_LAYER_KEY_RE.fullmatch(key)) is not None
        )
        decoder_indices = sorted(
            int(match.group(1))
            for key in source
            if (match := DECODER_LAYER_KEY_RE.fullmatch(key)) is not None
        )
        if (
            not encoder_indices
            or encoder_indices != list(range(encoder_indices[-1] + 1))
            or decoder_indices != encoder_indices
        ):
            raise SkillError(
                "source encoder/decoder layer hidden keys must be matching and contiguous"
            )
        ordered = ["input_ids", "attention_mask", "encoder.embed"]
        ordered.extend(f"encoder.layer.{index}.hidden" for index in encoder_indices)
        ordered.extend(("encoder.final_norm", "decoder_input_ids", "decoder.embed"))
        for index in decoder_indices:
            cross = f"decoder.layer.{index}.cross_attention"
            if cross not in source:
                raise SkillError(f"source capture is missing parity key: {cross}")
            ordered.extend((cross, f"decoder.layer.{index}.hidden"))
        ordered.extend(("decoder.final_norm", "logits", "generated_token_ids"))
    else:
        layer_indices = sorted(
            int(match.group(1))
            for key in source
            if (match := LAYER_KEY_RE.fullmatch(key)) is not None
        )
        if not layer_indices or layer_indices != list(range(layer_indices[-1] + 1)):
            raise SkillError("source layer.{i}.hidden keys must be contiguous from layer 0")
        ordered = ["input_features"] if mode == "asr" else ["input_ids"]
        if mode == "encoder":
            ordered.append("attention_mask")
        ordered.append("embed")
        ordered.extend(f"layer.{index}.hidden" for index in layer_indices)
        if mode == "encoder":
            ordered.extend(("final_hidden", "pooled"))
        elif mode == "asr":
            ordered.append("final_hidden")
        else:
            ordered.extend(("final_norm", "logits", "generated_token_ids"))

    exact_keys = {"input_features", "input_ids", "attention_mask", "decoder_input_ids", "generated_token_ids"}
    return [
        {
            "position": position,
            "name": key,
            "source_key": key,
            "target_key": key,
            "target_present": key in target,
            "exact": key in exact_keys,
        }
        for position, key in enumerate(ordered)
    ]


def tensor_key_mapping(rungs: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {rung["source_key"]: rung["target_key"] for rung in rungs}
    if len(mapping) != len(rungs):
        raise SkillError("parity ladder contains duplicate source keys")
    return mapping


def validate_parity_report(payload: Any) -> dict[str, Any]:
    report = _strict_fields(payload, REPORT_FIELDS, label="parity report")
    if type(report["schema_version"]) is not int or report["schema_version"] != SCHEMA_VERSION:
        raise SkillError("parity report schema_version must be integer 1")
    if not isinstance(report["ok"], bool):
        raise SkillError("parity report ok must be boolean")
    legacy_input_fields = {
        "source_model", "package", "weights", "mode", "prompts", "token_ids",
        "generate_steps", "seed", "dtype_policy", "allow_modified",
    }
    reproducible_input_fields = legacy_input_fields | {
        "input_mode", "attention_mask", "fault_inject_target",
    }
    asr_input_fields = {
        "source_model", "package", "weights", "mode", "waveform_samples",
        "seed", "dtype_policy", "allow_modified", "fault_inject_target",
    }
    inputs_value = report["inputs"]
    if not isinstance(inputs_value, dict):
        raise SkillError("parity report inputs has an invalid field set")
    if set(inputs_value) == asr_input_fields:
        inputs = inputs_value
        reproducible = True
        input_mode = "asr"
        if inputs["mode"] != "asr":
            raise SkillError("parity report contract mode is invalid")
        if type(inputs["waveform_samples"]) is not int or inputs["waveform_samples"] < 400:
            raise SkillError("parity report waveform_samples is invalid")
        if inputs["fault_inject_target"] is not None and not isinstance(
            inputs["fault_inject_target"], str
        ):
            raise SkillError("parity report fault injection target is invalid")
    else:
        if set(inputs_value) not in {
            frozenset(legacy_input_fields), frozenset(reproducible_input_fields),
        }:
            raise SkillError("parity report inputs has an invalid field set")
        inputs = inputs_value
        reproducible = set(inputs) == reproducible_input_fields
        input_mode = inputs["input_mode"] if reproducible else inputs["mode"]
        if reproducible:
            if inputs["mode"] not in {
                "dense-decoder", "encoder", "encoder-decoder", "ssm",
            }:
                raise SkillError("parity report contract mode is invalid")
            if inputs["fault_inject_target"] is not None and not isinstance(
                inputs["fault_inject_target"], str
            ):
                raise SkillError("parity report fault injection target is invalid")
            attention_mask = inputs["attention_mask"]
            if (
                not isinstance(attention_mask, list)
                or not attention_mask
                or not all(
                    isinstance(row, list)
                    and row
                    and all(type(value) is int and value in (0, 1) for value in row)
                    for row in attention_mask
                )
                or len({len(row) for row in attention_mask}) != 1
            ):
                raise SkillError(
                    "parity report attention_mask must be a non-empty 0/1 matrix"
                )
        if input_mode not in {"prompt", "token_ids"}:
            raise SkillError("parity report input mode is invalid")
        if input_mode == "token_ids":
            token_ids = inputs["token_ids"]
            if (
                inputs["prompts"] is not None
                or not isinstance(token_ids, list)
                or not token_ids
                or not all(type(value) is int and value >= 0 for value in token_ids)
            ):
                raise SkillError("parity report token-ID input is invalid")
            selected_input_count = 1
        else:
            prompts = inputs["prompts"]
            if (
                inputs["token_ids"] is not None
                or not isinstance(prompts, list)
                or not prompts
                or not all(isinstance(value, str) for value in prompts)
            ):
                raise SkillError("parity report prompt input is invalid")
            selected_input_count = len(prompts)
        if reproducible:
            attention_mask = inputs["attention_mask"]
            if len(attention_mask) != selected_input_count:
                raise SkillError(
                    "parity report attention_mask row count must match the selected input"
                )
            if input_mode == "token_ids" and len(attention_mask[0]) != len(token_ids):
                raise SkillError(
                    "parity report attention_mask width must match the token-ID input"
                )
        if type(inputs["generate_steps"]) is not int or inputs["generate_steps"] < 0:
            raise SkillError("parity report generate_steps is invalid")

    if not all(
        isinstance(inputs[field], str) and inputs[field]
        for field in ("source_model", "package", "weights")
    ):
        raise SkillError("parity report input paths must be non-empty strings")
    if type(inputs["seed"]) is not int:
        raise SkillError("parity report seed is invalid")
    if inputs["dtype_policy"] not in {"float32", "keep"}:
        raise SkillError("parity report dtype policy is invalid")
    if not isinstance(inputs["allow_modified"], bool):
        raise SkillError("parity report allow_modified must be boolean")
    tolerances = _strict_fields(
        report["tolerances"],
        {"atol", "rtol", "cosine_min"},
        label="parity report tolerances",
    )
    validate_comparison_tolerances(
        tolerances["atol"],
        tolerances["rtol"],
        tolerances["cosine_min"],
    )
    if not isinstance(report["rungs"], list) or not report["rungs"]:
        raise SkillError("parity report rungs must be a non-empty list")
    for index, raw in enumerate(report["rungs"]):
        rung = _strict_fields(raw, RUNG_FIELDS, label=f"parity report rungs[{index}]")
        if (
            rung["position"] != index
            or not isinstance(rung["name"], str)
            or not isinstance(rung["source_key"], str)
            or not isinstance(rung["target_key"], str)
            or not isinstance(rung["exact"], bool)
            or not isinstance(rung["pass"], bool)
        ):
            raise SkillError("parity report rung position/pass is invalid")
        for metric in ("max_abs", "max_rel", "cosine"):
            if rung[metric] is not None and (
                isinstance(rung[metric], bool)
                or not isinstance(rung[metric], (int, float))
                or not math.isfinite(rung[metric])
            ):
                raise SkillError(f"parity report rung {metric} must be finite or null")
    summary = _strict_fields(report["summary"], SUMMARY_FIELDS, label="parity report summary")
    if summary["status"] not in {"pass", "fail"}:
        raise SkillError("parity report summary status is invalid")
    if summary["evaluated_rungs"] != len(report["rungs"]):
        raise SkillError("parity report summary evaluated_rungs is inconsistent")
    if (
        type(summary["total_rungs"]) is not int
        or summary["total_rungs"] < summary["evaluated_rungs"]
        or not isinstance(summary["message"], str)
        or not summary["message"]
    ):
        raise SkillError("parity report summary counts/message are invalid")
    if report["ok"] != (summary["status"] == "pass"):
        raise SkillError("parity report ok and summary status disagree")
    if report["ok"] and not all(rung["pass"] for rung in report["rungs"]):
        raise SkillError("passing parity report contains a failing rung")
    if report["ok"] and summary["total_rungs"] != len(report["rungs"]):
        raise SkillError("passing parity report must evaluate every rung")
    if report["ok"] and (summary["stopped_at"] is not None or summary["debug_target"] is not None):
        raise SkillError("passing parity report must not contain a stop target")
    if not report["ok"] and not all(
        isinstance(summary[field], str) and summary[field]
        for field in ("stopped_at", "debug_target")
    ):
        raise SkillError("failing parity report must identify the stopped/debug target")
    if not report["ok"] and report["rungs"][-1]["pass"]:
        raise SkillError("failing parity report must stop on a failing rung")
    return report


def _regular_directory(value: str, *, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_dir():
        raise SkillError(f"{label} must be a local, non-symlink directory: {value}")
    return path


def _python_script(script: Path) -> list[str]:
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    command.append(str(script))
    return command


def _input_arguments(args: argparse.Namespace) -> list[str]:
    if args.mode == "asr":
        return [
            "--waveform-samples", str(args.waveform_samples),
            "--seed", str(args.seed),
            "--max-output-mb", str(args.max_output_mb),
            *(["--keep-dtype"] if args.keep_dtype else []),
        ]
    values: list[str] = []
    for prompt in args.prompt or []:
        values.extend(("--prompt", prompt))
    for prompt_file in args.prompts_file or []:
        values.extend(("--prompts-file", prompt_file))
    if args.token_ids:
        values.append("--token-ids")
        values.extend(args.token_ids)
    if args.attention_mask:
        values.append("--attention-mask")
        values.extend(args.attention_mask)
    values.extend(("--generate-steps", str(args.generate_steps)))
    values.extend(("--seed", str(args.seed)))
    values.extend(("--max-output-mb", str(args.max_output_mb)))
    if args.keep_dtype:
        values.append("--keep-dtype")
    return values


def _run_tool(command: list[str], *, timeout: float, label: str) -> None:
    completed, timed_out = run_process_capture(command, timeout=timeout)
    if timed_out:
        raise SkillError(f"{label} timed out after {timeout:g} seconds")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise SkillError(f"{label} failed with exit {completed.returncode}: {detail}")


def _debug_target(key: str, mode: str = "dense-decoder") -> tuple[str, str]:
    if key.startswith("encoder.layer."):
        layer = key.split(".")[2]
        return f"encoder layer {layer}", f"First divergence is encoder layer {layer}."
    if key.startswith("decoder.layer."):
        layer = key.split(".")[2]
        branch = " cross-attention" if key.endswith("cross_attention") else ""
        return (
            f"decoder layer {layer}{branch}",
            f"First divergence is decoder layer {layer}{branch}.",
        )
    match = LAYER_KEY_RE.fullmatch(key)
    if match is not None:
        layer = match.group(1)
        block = "encoder" if mode == "encoder" else "decoder"
        return f"layer {layer}", f"First divergence is layer {layer}; debug that {block} block."
    messages = {
        "input_features": (
            "feature input",
            "Extracted frontend features differ; debug the frozen ASR fixture plumbing.",
        ),
        "input_ids": ("input preparation", "Input IDs differ; debug tokenization or fixture plumbing."),
        "attention_mask": ("attention mask", "Attention masks differ; debug padding-mask plumbing."),
        "encoder.embed": ("encoder embedding", "First divergence is the encoder embedding stage."),
        "encoder.final_norm": ("encoder final norm", "Encoder layers passed; debug its final norm."),
        "decoder_input_ids": ("decoder start", "Decoder start IDs differ."),
        "decoder.embed": ("decoder embedding", "First divergence is the decoder embedding stage."),
        "decoder.final_norm": ("decoder final norm", "Decoder layers passed; debug its final norm."),
        "embed": ("embedding stage", "First divergence is the embedding stage; debug embedding weights and lookup."),
        "final_hidden": ("final hidden", "Encoder layers passed; debug final hidden-state capture."),
        "pooled": ("pooler", "Hidden states passed; debug CLS pooling or the dense+tanh pooler."),
        "final_norm": ("final norm", "Decoder layers passed; debug the final normalization stage."),
        "logits": ("LM head", "Hidden states passed; debug the LM head or output scaling."),
        "generated_token_ids": (
            "greedy decoding",
            "Logits passed tolerance but greedy IDs differ; debug exact argmax/decode behavior.",
        ),
    }
    return messages[key]


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_capture_limits(args.generate_steps, args.max_output_mb)
        validate_comparison_tolerances(args.atol, args.rtol, args.cosine_min)
        if (
            not math.isfinite(args.timeout_seconds)
            or args.timeout_seconds <= 0
            or args.timeout_seconds > MAX_TIMEOUT_SECONDS
        ):
            raise SkillError(
                f"--timeout-seconds must be finite, positive, and at most {MAX_TIMEOUT_SECONDS:g}"
            )
        if args.source_model is None or args.package is None or args.weights is None:
            raise SkillError("--source-model, --package, and --weights are required")
        if args.mode == "asr":
            if args.prompt or args.prompts_file or args.token_ids or args.attention_mask:
                raise SkillError("--mode asr cannot be combined with text/token inputs")
            if args.generate_steps != DEFAULT_GENERATE_STEPS:
                raise SkillError("--generate-steps is not valid with --mode asr")
            if type(args.waveform_samples) is not int or args.waveform_samples < 400:
                raise SkillError("--waveform-samples must be an integer of at least 400")
            prompts, token_ids = [], None
        else:
            if args.waveform_samples != 16000:
                raise SkillError("--waveform-samples is only valid with --mode asr")
            prompts, token_ids = validate_input_mode(
                args.token_ids, args.prompt, args.prompts_file
            )
            if args.mode == "encoder" and args.generate_steps != 0:
                raise SkillError("--mode encoder requires --generate-steps 0")
            if args.attention_mask and args.mode not in {"encoder", "encoder-decoder"}:
                raise SkillError("--attention-mask is supported only with an encoder mode")
            if args.attention_mask and token_ids is None:
                raise SkillError("--attention-mask requires --token-ids")
        source_model = _regular_directory(args.source_model, label="source model")
        package = _regular_directory(args.package, label="package")
        weights = _regular_directory(args.weights, label="weights")

        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="mlx-parity-") as raw_tmp:
            work = Path(raw_tmp).resolve()
            source_npz = work / "source.npz"
            target_npz = work / "target.npz"
            source_manifest_path = work / "source.manifest.json"
            target_manifest_path = work / "target.manifest.json"
            common_input = _input_arguments(args)
            oracle_command = [
                *_python_script(scripts / "capture_oracle.py"),
                str(source_model),
                "--mode",
                args.mode,
                *common_input,
                "--output",
                str(source_npz),
                "--manifest",
                str(source_manifest_path),
            ]
            _run_tool(
                oracle_command,
                timeout=args.timeout_seconds,
                label="capture_oracle.py",
            )
            mlx_command = [
                *_python_script(scripts / "capture_mlx.py"),
                "--mode",
                args.mode,
                "--package",
                str(package),
                "--weights",
                str(weights),
                *common_input,
                *(["--features-npz", str(source_npz)] if args.mode == "asr" else []),
                "--output",
                str(target_npz),
                "--manifest",
                str(target_manifest_path),
            ]
            if prompts and args.mode != "asr":
                mlx_command.extend(("--tokenizer", str(source_model)))
            if args.allow_modified:
                mlx_command.append("--allow-modified")
            if args.fault_inject_target:
                mlx_command.extend(("--fault-inject-target", args.fault_inject_target))
            _run_tool(
                mlx_command,
                timeout=args.timeout_seconds,
                label="capture_mlx.py",
            )

            source_manifest = validate_capture_manifest(
                read_bounded_json(
                    source_manifest_path,
                    MAX_MANIFEST_JSON_BYTES,
                    label="source capture manifest",
                )
            )
            target_manifest = validate_capture_manifest(
                read_bounded_json(
                    target_manifest_path,
                    MAX_MANIFEST_JSON_BYTES,
                    label="target capture manifest",
                )
            )
            if source_manifest["capture"] != target_manifest["capture"]:
                raise SkillError("source and target capture configurations differ")
            source_keys = [item["name"] for item in source_manifest["tensors"]]
            target_keys = [item["name"] for item in target_manifest["tensors"]]
            ladder = build_parity_ladder(source_keys, target_keys, args.mode)
            mapping_path = work / "mapping.json"
            write_strict_json(
                mapping_path,
                {"mapping": tensor_key_mapping(ladder)},
                label="parity mapping",
            )
            evaluated: list[dict[str, Any]] = []
            stopped_at: str | None = None
            for rung in ladder:
                comparison_path = work / f"compare-{rung['position']}.json"
                include = r"\A" + re.escape(rung["source_key"]) + r"\Z"
                command = [
                    *_python_script(scripts / "compare_tensors.py"),
                    str(source_npz),
                    str(target_npz),
                    "--mapping",
                    str(mapping_path),
                    "--include",
                    include,
                    "--atol",
                    str(0.0 if rung["exact"] else args.atol),
                    "--rtol",
                    str(0.0 if rung["exact"] else args.rtol),
                    "--cosine-min",
                    str(args.cosine_min),
                    "--output",
                    str(comparison_path),
                ]
                completed, timed_out = run_process_capture(command, timeout=args.timeout_seconds)
                if timed_out:
                    raise SkillError(
                        f"compare_tensors.py timed out at rung {rung['name']} after "
                        f"{args.timeout_seconds:g} seconds"
                    )
                if completed.returncode not in {0, 1}:
                    detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
                    raise SkillError(
                        f"compare_tensors.py failed at rung {rung['name']} with exit "
                        f"{completed.returncode}: {detail}"
                    )
                comparison = read_bounded_json(
                    comparison_path,
                    MAX_MANIFEST_JSON_BYTES,
                    label=f"comparison report for {rung['name']}",
                )
                rows = comparison.get("rows") if isinstance(comparison, dict) else None
                if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                    raise SkillError(f"comparison report for {rung['name']} must contain one row")
                row = rows[0]
                passed = completed.returncode == 0 and comparison.get("ok") is True and row.get("ok") is True
                evaluated.append({
                    "position": rung["position"],
                    "name": rung["name"],
                    "source_key": rung["source_key"],
                    "target_key": rung["target_key"],
                    "exact": rung["exact"],
                    "pass": passed,
                    "max_abs": _metric(row, "max_abs"),
                    "max_rel": _metric(row, "max_rel"),
                    "cosine": _metric(row, "cosine"),
                })
                if not passed:
                    stopped_at = rung["name"]
                    break

        ok = stopped_at is None
        if ok:
            debug_target = None
            message = "All parity rungs passed in runbook order."
            status = "pass"
        else:
            debug_target, message = _debug_target(stopped_at, args.mode)
            status = "fail"
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "inputs": ({
                "source_model": str(source_model),
                "package": str(package),
                "weights": str(weights),
                "mode": "asr",
                "waveform_samples": args.waveform_samples,
                "seed": args.seed,
                "dtype_policy": "keep" if args.keep_dtype else "float32",
                "allow_modified": args.allow_modified,
                "fault_inject_target": args.fault_inject_target,
            } if args.mode == "asr" else {
                "source_model": str(source_model),
                "package": str(package),
                "weights": str(weights),
                "mode": args.mode,
                "input_mode": "token_ids" if token_ids is not None else "prompt",
                "prompts": prompts if token_ids is None else None,
                "token_ids": token_ids,
                "attention_mask": target_manifest["capture"]["attention_mask"],
                "generate_steps": args.generate_steps,
                "seed": args.seed,
                "dtype_policy": "keep" if args.keep_dtype else "float32",
                "allow_modified": args.allow_modified,
                "fault_inject_target": args.fault_inject_target,
            }),
            "tolerances": {
                "atol": args.atol,
                "rtol": args.rtol,
                "cosine_min": args.cosine_min,
            },
            "rungs": evaluated,
            "summary": {
                "status": status,
                "evaluated_rungs": len(evaluated),
                "total_rungs": len(ladder),
                "stopped_at": stopped_at,
                "debug_target": debug_target,
                "message": message,
            },
        }
        validate_parity_report(report)
        if args.output:
            write_strict_json(Path(args.output), report, label="parity report")
        else:
            sys.stdout.buffer.write(strict_json_bytes(report, label="parity report"))
        return 0 if ok else 1
    except (SkillError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
