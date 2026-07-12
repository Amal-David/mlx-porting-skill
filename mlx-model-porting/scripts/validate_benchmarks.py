#!/usr/bin/env python3
"""Validate benchmark receipts and deterministically assess promotion readiness."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from _common import (
    SkillError,
    atomic_write_text,
    dump_json,
    redact_secret_text,
    redact_secrets,
    stable_mean,
    stable_median,
    stable_pstdev,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_ROOT / "assets" / "benchmarks"
ASSESSMENT_NAME = "receipt_assessments.json"
INDEX_NAME = "receipts_index.json"
REPORT_NAME = "BENCHMARK_REPORT.md"
RESERVED_JSON = {ASSESSMENT_NAME, INDEX_NAME}
MLX_METRICS = (
    "prompt_tokens",
    "prompt_tps",
    "generation_tokens",
    "generation_tps",
    "peak_memory_gb",
    "ttft_proxy_s",
)
EXTERNAL_METRICS = ("wall_seconds",)
# Kept as the historical default for callers that construct MLX-LM receipts.
METRICS = MLX_METRICS
DEFAULT_PRIMARY_METRIC = "generation_tps"
DEFAULT_MIN_RUNS = 5
DEFAULT_MAX_CV = 0.10
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
RUNNER_OPTION_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
EXTERNAL_RUNNER_ID = "external-command-wall-time"
EXTERNAL_RUNNER_SHAPE = "direct-argv-template-v1"
ATTESTED_RUNNER_ID = "attested-mlx-port-wall-time"
ATTESTED_RUNNER_SHAPE = "repository-owned-qwen2.5-workload-v1"
ATTESTED_RUNNER_PATH = "runners/attested_mlx_port.py"
ATTESTED_MODEL_ID = "local/Qwen2.5-0.5B-Instruct-standalone-mlx"
ATTESTED_WORKLOAD_ID = "qwen2.5-0.5b-load-plus-six-token-greedy-decode"
ATTESTED_WORKLOAD_PARAMETERS = {
    "batch": 1,
    "cache": "growing-kv",
    "compile": False,
    "generate_steps": 6,
    "prompt_tokens": 11,
    "timing_scope": "separate-process-load-plus-greedy-decode",
}
ATTESTED_VARIANTS = {
    "f32": {
        "revision": "7df68182b704fb0933625cdb44cc4e89e8f4cfdb5cb75b3e2251f608807f65f0",
        "weights_path": "converted-f32/model.safetensors",
        "weights_bytes": 1_976_163_149,
        "dtype_policy": "f32",
    },
    "bf16": {
        "revision": "7d325703c071cbba116c2cbd2242f20510c797fb9d9384573e5f6ca370582c7f",
        "weights_path": "converted-bf16/model.safetensors",
        "weights_bytes": 988_097_714,
        "dtype_policy": "bf16",
    },
}
EXTERNAL_ATTESTATION_BLOCKER = "missing-external-attestation-signature"
EXTERNAL_ATTESTATION_SIGNED_FIELDS = (
    "repository_commit",
    "repository_tree",
    "challenge",
    "reviewed_dependency_manifest",
    "raw_output",
    "promotion_policy",
    "timing",
)
ATTESTATION_PROMOTION_BLOCKERS = {
    "execution-semantics-unattested",
    EXTERNAL_ATTESTATION_BLOCKER,
}
EXTERNAL_INTERPRETER_FLAGS = ["-I", "-B"]
EXTERNAL_MODEL_FIELDS = {"id", "revision", "lineage_id", "source_id", "source_revision"}
SHELL_EXECUTABLES = {"bash", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "zsh"}
COMMAND_WRAPPERS = {"command", "doas", "env", "exec", "nice", "nohup", "sudo", "timeout", "xargs"}
DYNAMIC_CODE_FLAGS = {
    "node": {"-e", "--eval"},
    "python": {"-c"},
    "ruby": {"-e"},
    "perl": {"-e"},
    "php": {"-r"},
    "osascript": {"-e"},
}
DECLARATIVE_QUALITY_VALIDATOR = {"id": "mlx-benchmark-declarative-quality", "version": 1}
EXACT_OUTPUT_QUALITY_VALIDATOR = {"id": "mlx-benchmark-exact-output-parity", "version": 1}
MAX_QUALITY_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_EXTERNAL_REPORT_BYTES = 8 * 1024 * 1024
MAX_EXTERNAL_RUNS = 100
MAX_EXTERNAL_WARMUPS = 20
MAX_EXTERNAL_TIMEOUT_SECONDS = 3600.0
MAX_ATTESTATION_CHALLENGE_BYTES = 64 * 1024
MAX_ATTESTATION_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_ATTESTED_DEPENDENCY_BYTES = 16 * 1024 * 1024
MAX_ATTESTED_DEPENDENCY_SET_BYTES = 32 * 1024 * 1024


def exact_output_validator_descriptor_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"id", "version"}
        and value.get("id") == EXACT_OUTPUT_QUALITY_VALIDATOR["id"]
        and not isinstance(value.get("version"), bool)
        and isinstance(value.get("version"), int)
        and value["version"] == EXACT_OUTPUT_QUALITY_VALIDATOR["version"]
    )


def external_interpreter_environment_valid(value: Any) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "sys_path_sha256", "distributions", "startup_files", "sha256"}
        or value.get("schema_version") != 1
        or not isinstance(value.get("sys_path_sha256"), str)
        or HEX_DIGEST_RE.fullmatch(value["sys_path_sha256"]) is None
        or not isinstance(value.get("distributions"), list)
        or not isinstance(value.get("startup_files"), list)
    ):
        return False
    for distribution in value["distributions"]:
        if (
            not isinstance(distribution, dict)
            or set(distribution)
            != {
                "name",
                "version",
                "location_sha256",
                "metadata_sha256",
                "record_sha256",
                "direct_url_sha256",
            }
            or any(
                not isinstance(distribution.get(field), str)
                or not distribution[field]
                for field in ("name", "version")
            )
            or any(
                not isinstance(distribution.get(field), str)
                or HEX_DIGEST_RE.fullmatch(distribution[field]) is None
                for field in (
                    "location_sha256",
                    "metadata_sha256",
                    "record_sha256",
                    "direct_url_sha256",
                )
            )
        ):
            return False
    for startup_file in value["startup_files"]:
        if (
            not isinstance(startup_file, dict)
            or set(startup_file) != {"path_sha256", "sha256", "size_bytes"}
            or not isinstance(startup_file.get("path_sha256"), str)
            or HEX_DIGEST_RE.fullmatch(startup_file["path_sha256"]) is None
            or not isinstance(startup_file.get("sha256"), str)
            or HEX_DIGEST_RE.fullmatch(startup_file["sha256"]) is None
            or isinstance(startup_file.get("size_bytes"), bool)
            or not isinstance(startup_file.get("size_bytes"), int)
            or startup_file["size_bytes"] < 0
        ):
            return False
    payload = {
        key: value[key]
        for key in ("schema_version", "sys_path_sha256", "distributions", "startup_files")
    }
    return value.get("sha256") == canonical_hash(payload)
SECRET_COMMAND_RE = re.compile(
    r"(?i)(?:--|\b)(?:api[-_]?key|access[-_]?token|auth[-_]?token|token|secret|password)(?:=|\s)"
)
MAC_HOME_PREFIX = "/" + "Users" + "/"
EPHEMERAL_PATH_RE = re.compile(
    r"(?:<historical-ephemeral|(?:^|[\s'\"])(?:/tmp/|/private/var/|/var/folders/|"
    + re.escape(MAC_HOME_PREFIX)
    + r"|/home/|/root/))"
)
SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(?:authorization|proxy[_-]?authorization|api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"secret|password|private[_-]?key|database[_-]?url|connection[_-]?string|cookie)(?:$|[_-])|"
    r"(?:^|[_-])token$",
    re.IGNORECASE,
)
PROMPT_RE = re.compile(r"Prompt:\s*(\d+)\s+tokens,\s*([0-9]+(?:\.[0-9]+)?)\s+tokens-per-sec")
GENERATION_RE = re.compile(r"Generation:\s*(\d+)\s+tokens,\s*([0-9]+(?:\.[0-9]+)?)\s+tokens-per-sec")
PEAK_MEMORY_RE = re.compile(r"Peak memory:\s*([0-9]+(?:\.[0-9]+)?)\s+GB")
INTEGRITY_BLOCKERS = {
    "aggregate-mismatch",
    "invalid-runs",
    "raw-output-digest-mismatch",
    "target-hash-mismatch",
    "workload-hash-mismatch",
    "quality-artifact-digest-mismatch",
    "attestation-runner-digest-mismatch",
    "attestation-challenge-mismatch",
    "attestation-evidence-digest-mismatch",
    "attestation-dependency-digest-mismatch",
    "attestation-output-digest-mismatch",
}
HISTORICAL_ENABLED_METHODS = {
    "kv-4bit-8k": ["uniform-kv-quantization"],
    "pcache-warm": ["prompt-prefix-cache"],
    "quant-4bit": ["native-low-bit-weight-quantization"],
    "spec-draft-k2": ["draft-model-speculation"],
    "spec-draft-k3": ["draft-model-speculation"],
    "spec-draft-k4": ["draft-model-speculation"],
    "stack-measured-together": [
        "uniform-kv-quantization",
        "prompt-prefix-cache",
        "draft-model-speculation",
    ],
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def receipt_metrics(receipt: dict[str, Any]) -> tuple[str, ...]:
    runner = receipt.get("runner")
    if isinstance(runner, dict) and runner.get("id") in {EXTERNAL_RUNNER_ID, ATTESTED_RUNNER_ID}:
        return EXTERNAL_METRICS
    return MLX_METRICS


def _template_source_kind(source: Any) -> str | None:
    if not isinstance(source, list):
        return None
    if len(source) == 3 and source[:2] == ["models", "target"] and source[2] in EXTERNAL_MODEL_FIELDS:
        return "model"
    if len(source) == 3 and source[:2] == ["workload", "parameters"] and isinstance(source[2], str):
        return "workload"
    if (
        len(source) == 4
        and source[:2] == ["workload", "artifacts"]
        and isinstance(source[2], int)
        and not isinstance(source[2], bool)
        and source[2] >= 0
        and source[3] == "path"
    ):
        return "workload"
    if len(source) == 2 and source[0] == "variant_config" and isinstance(source[1], str):
        return "variant"
    return None


def _resolve_template_source(receipt: dict[str, Any], source: list[Any]) -> Any:
    value: Any = receipt
    for segment in source:
        if isinstance(value, dict) and isinstance(segment, str) and segment in value:
            value = value[segment]
        elif isinstance(value, list) and isinstance(segment, int) and 0 <= segment < len(value):
            value = value[segment]
        else:
            raise SkillError(f"external argv template source does not resolve: {source!r}")
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, (str, int, float)) or (isinstance(value, float) and not math.isfinite(value)):
        raise SkillError(f"external argv template source must resolve to a finite scalar: {source!r}")
    rendered = str(value)
    if not rendered:
        raise SkillError(f"external argv template source must not resolve to an empty value: {source!r}")
    return rendered


def resolve_external_argv_template(receipt: dict[str, Any], template: Any) -> list[str]:
    if not isinstance(template, list) or not template:
        raise SkillError("external argv template must be a non-empty list")
    command: list[str] = []
    binding_kinds: set[str] = set()
    target_id_bound = False
    target_revision_bound = False
    for index, entry in enumerate(template):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise SkillError("external argv template entries must contain exactly literal or source")
        if "literal" in entry:
            literal = entry["literal"]
            if not isinstance(literal, str) or not literal:
                raise SkillError("external argv template literals must be non-empty strings")
            command.append(literal)
            continue
        source = entry.get("source")
        kind = _template_source_kind(source)
        if kind is None:
            raise SkillError(f"external argv template source is not allowlisted: {source!r}")
        if index == 0:
            raise SkillError("external argv executable must be an exact literal")
        binding_kinds.add(kind)
        target_id_bound = target_id_bound or source == ["models", "target", "id"]
        target_revision_bound = target_revision_bound or source == ["models", "target", "revision"]
        command.append(_resolve_template_source(receipt, source))
    if not target_id_bound or not target_revision_bound:
        binding_kinds.discard("model")
    missing = {"model", "workload", "variant"} - binding_kinds
    if missing:
        raise SkillError(f"external argv template is missing required bindings: {', '.join(sorted(missing))}")
    return command


def external_command_is_safe(command: list[str]) -> bool:
    if not command or not all(isinstance(item, str) and item for item in command):
        return False
    executable = Path(command[0]).name.lower()
    if executable in SHELL_EXECUTABLES or executable in COMMAND_WRAPPERS:
        return False
    normalized_executable = "python" if re.fullmatch(r"python(?:3(?:\.\d+)*)?", executable) else executable
    if any(flag in command[1:] for flag in DYNAMIC_CODE_FLAGS.get(normalized_executable, set())):
        return False
    lowered = [item.lower() for item in command]
    if any(item == "--trust-remote-code" or item.startswith("--trust-remote-code=") for item in lowered):
        return False
    rendered = shlex.join(command)
    return SECRET_COMMAND_RE.search(rendered) is None and not any(EPHEMERAL_PATH_RE.search(item) for item in command)


def build_external_runner_descriptor(receipt: dict[str, Any], template: Any) -> dict[str, Any]:
    command = resolve_external_argv_template(receipt, template)
    artifacts = receipt.get("workload", {}).get("artifacts", [])
    runner_artifacts = [
        (index, artifact)
        for index, artifact in enumerate(artifacts)
        if isinstance(artifact, dict) and artifact.get("role") == "runner"
    ] if isinstance(artifacts, list) else []
    if len(runner_artifacts) != 1:
        raise SkillError("external runner requires exactly one workload artifact with role runner")
    runner_index, runner_artifact = runner_artifacts[0]
    runner_source = ["workload", "artifacts", runner_index, "path"]
    runner_positions = [
        index
        for index, entry in enumerate(template)
        if isinstance(entry, dict) and entry.get("source") == runner_source
    ]
    executable_name = command[0]
    executable = Path(executable_name).name.lower()
    if (
        runner_positions != [1]
        or len(command) < 2
        or Path(executable_name).name != executable_name
        or re.fullmatch(r"python(?:3(?:\.\d+)*)?", executable) is None
        or command[1] != runner_artifact.get("path")
    ):
        raise SkillError(
            "external runner must execute the digest-pinned Python runner as argv[1]"
        )
    bindings = [
        {"index": index, "source": entry["source"], "value_sha256": canonical_hash(command[index])}
        for index, entry in enumerate(template)
        if "source" in entry
    ]
    interpreter = receipt.get("target", {}).get("descriptor", {}).get("interpreter")
    if interpreter is not None and (
        not isinstance(interpreter, dict)
        or set(interpreter) != {"name", "sha256", "size_bytes", "version", "flags", "environment"}
        or interpreter.get("name") != command[0]
        or not isinstance(interpreter.get("sha256"), str)
        or HEX_DIGEST_RE.fullmatch(interpreter["sha256"]) is None
        or isinstance(interpreter.get("size_bytes"), bool)
        or not isinstance(interpreter.get("size_bytes"), int)
        or interpreter["size_bytes"] <= 0
        or not isinstance(interpreter.get("version"), str)
        or not interpreter["version"]
        or interpreter.get("flags") != EXTERNAL_INTERPRETER_FLAGS
        or not external_interpreter_environment_valid(interpreter.get("environment"))
    ):
        raise SkillError("external runner interpreter descriptor is invalid")
    attested = runner_artifact.get("path") == ATTESTED_RUNNER_PATH
    if attested:
        label = receipt.get("label")
        input_indexes = [
            index for index, artifact in enumerate(artifacts)
            if isinstance(artifact, dict) and artifact.get("role") == "input"
        ]
        if len(input_indexes) != 1:
            raise SkillError("attested runner requires exactly one workload input artifact")
        expected_flags = [
            "--model",
            "--revision",
            "--input",
            "--steps",
            "--mode",
            "--output",
            "--attestation-challenge",
            "--attestation-output",
        ]
        expected_sources = {
            3: ["models", "target", "id"],
            5: ["models", "target", "revision"],
            7: ["workload", "artifacts", input_indexes[0], "path"],
            9: ["workload", "parameters", "generate_steps"],
            11: ["variant_config", "mode"],
            13: ["variant_config", "quality_output_path"],
            15: ["variant_config", "attestation_challenge_path"],
            17: ["variant_config", "attestation_output_path"],
        }
        if (
            not isinstance(label, str)
            or len(command) != 18
            or command[2::2] != expected_flags
            or any(template[index].get("source") != source for index, source in expected_sources.items())
            or command[15] != f"attestations/{label}/current-challenge.json"
            or command[17] != f"attestations/{label}/current-evidence.json"
        ):
            raise SkillError("attested runner requires the exact built-in argv and label-owned evidence paths")
    return {
        "id": ATTESTED_RUNNER_ID if attested else EXTERNAL_RUNNER_ID,
        "version": 1,
        "shape": ATTESTED_RUNNER_SHAPE if attested else EXTERNAL_RUNNER_SHAPE,
        "command_sha256": canonical_hash(command),
        "argv_template": template,
        "bindings": bindings,
        "implementation": {
            key: runner_artifact.get(key)
            for key in ("path", "sha256", "size_bytes")
        },
        "interpreter": interpreter,
    }


def parse_runner_options(command: list[str]) -> dict[str, str | bool] | None:
    controlled_shape = (
        len(command) >= 4
        and re.fullmatch(r"python(?:3(?:\.\d+)*)?", command[0]) is not None
        and command[1:4] == ["-m", "mlx_lm", "generate"]
    )
    if not controlled_shape:
        return None
    options: dict[str, str | bool] = {}
    index = 4
    while index < len(command):
        token = command[index]
        if not isinstance(token, str) or not token.startswith("--"):
            return None
        rendered = token[2:]
        if "=" in rendered:
            name, value = rendered.split("=", 1)
            if not value:
                return None
        else:
            name = rendered
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                index += 1
                value = command[index]
            else:
                value = True
        if RUNNER_OPTION_RE.fullmatch(name) is None or name in options:
            return None
        options[name] = value
        index += 1
    return options


def build_runner_descriptor(command: list[str]) -> dict[str, Any]:
    options = parse_runner_options(command)
    controlled = options is not None
    return {
        "id": "mlx-lm-generate" if controlled else "uncontrolled",
        "version": 2 if controlled else 1,
        "shape": "python -m mlx_lm generate" if controlled else "uncontrolled",
        "command_sha256": canonical_hash(command),
        "arguments": options if controlled else {},
    }


def _runner_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected or str(actual).lower() == str(expected).lower()
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return isinstance(actual, str) and actual == str(expected)


def _workload_option_name(parameter: str) -> tuple[str, ...]:
    normalized = parameter.strip().replace("_", "-")
    if normalized == "temperature":
        return ("temp", "temperature")
    return (normalized,)


def runner_semantics_valid(receipt: dict[str, Any]) -> bool:
    runner = receipt.get("runner")
    arguments = runner.get("arguments") if isinstance(runner, dict) else None
    if not isinstance(arguments, dict):
        return False
    if "trust-remote-code" in arguments:
        return False
    models = receipt.get("models")
    target_model = models.get("target") if isinstance(models, dict) else None
    if not isinstance(target_model, dict) or arguments.get("model") != target_model.get("id"):
        return False

    draft_model = models.get("draft") if isinstance(models, dict) else None
    if isinstance(draft_model, dict):
        if arguments.get("draft-model") != draft_model.get("id"):
            return False
    elif "draft-model" in arguments:
        return False

    workload = receipt.get("workload")
    if not isinstance(workload, dict):
        return False
    artifacts = workload.get("artifacts")
    prompt_artifacts = [
        artifact
        for artifact in artifacts if isinstance(artifact, dict) and artifact.get("role") == "prompt"
    ] if isinstance(artifacts, list) else []
    prompt = arguments.get("prompt")
    if len(prompt_artifacts) != 1 or not isinstance(prompt, str):
        return False
    prompt_digest = prompt_artifacts[0].get("sha256")
    if not isinstance(prompt_digest, str) or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_digest:
        return False

    parameters = workload.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        return False
    invariant_names = {"model", "prompt"}
    for key, expected in parameters.items():
        aliases = _workload_option_name(str(key))
        present = [name for name in aliases if name in arguments]
        if len(present) != 1 or not _runner_value_matches(arguments[present[0]], expected):
            return False
        invariant_names.update(aliases)
    if isinstance(draft_model, dict):
        invariant_names.add("draft-model")

    variant_config = receipt.get("variant_config")
    if not isinstance(variant_config, dict):
        return False
    for name, value in arguments.items():
        if name in invariant_names:
            continue
        config_value = variant_config.get(name, variant_config.get(name.replace("-", "_")))
        if config_value is None or not _runner_value_matches(value, config_value):
            return False
    return True


def build_experiment_contract(receipt: dict[str, Any]) -> dict[str, Any]:
    runner = receipt.get("runner", {})
    target_model = receipt.get("models", {}).get("target", {})
    target = receipt.get("target", {})
    workload = receipt.get("workload", {})
    comparison = receipt.get("comparison", {})
    runs = receipt.get("runs", [])
    environment = receipt.get("environment", {})
    if not isinstance(environment, dict):
        environment = {}
    external_runner = (
        isinstance(runner, dict)
        and runner.get("id") in {EXTERNAL_RUNNER_ID, ATTESTED_RUNNER_ID}
    )
    runner_arguments = runner.get("arguments", {}) if isinstance(runner, dict) else {}
    workload_parameters = workload.get("parameters", {}) if isinstance(workload, dict) else {}
    workload_argument_names = {
        alias
        for key in workload_parameters
        for alias in _workload_option_name(str(key))
    } if isinstance(workload_parameters, dict) else set()
    prompt_artifacts = [
        artifact
        for artifact in workload.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("role") == "prompt"
    ] if isinstance(workload, dict) else []
    if external_runner:
        bindings = runner.get("bindings", []) if isinstance(runner, dict) else []
        invocation_invariant = {
            "argv_template": runner.get("argv_template"),
            "bindings": [
                {
                    "index": binding.get("index"),
                    "source": binding.get("source"),
                }
                for binding in bindings
                if isinstance(binding, dict)
                if _template_source_kind(binding.get("source") if isinstance(binding, dict) else None)
                in {"model", "workload"}
            ],
        }
        invocation_variant = {
            ".".join(str(segment) for segment in binding["source"]): binding.get("value_sha256")
            for binding in bindings
            if isinstance(binding, dict)
            and _template_source_kind(binding.get("source")) in {"model", "variant"}
            and tuple(binding.get("source", [])) not in {
                ("variant_config", "quality_output_path"),
                ("variant_config", "attestation_challenge_path"),
                ("variant_config", "attestation_output_path"),
            }
        }
        sample_shape = {
            "run_count": len(runs) if isinstance(runs, list) else 0,
            "warmup_count": receipt.get("warmup_runs"),
            "timeout_seconds": receipt.get("timeout_seconds"),
        }
    else:
        invocation_invariant = {
            "model": runner_arguments.get("model") if isinstance(runner_arguments, dict) else None,
            "prompt_sha256": prompt_artifacts[0].get("sha256") if len(prompt_artifacts) == 1 else None,
            "workload_arguments": {
                key: runner_arguments[key]
                for key in sorted(workload_argument_names)
                if isinstance(runner_arguments, dict) and key in runner_arguments
            },
        }
        invocation_variant = {
            key: runner_arguments[key]
            for key in sorted(runner_arguments)
            if key not in {"model", "prompt", *workload_argument_names}
        } if isinstance(runner_arguments, dict) else {}
        sample_shape = {
            "run_count": len(runs) if isinstance(runs, list) else 0,
            "declared_run_count": receipt.get("measured_runs"),
            "warmup_count": receipt.get("warmup_runs"),
            "timeout_seconds": receipt.get("timeout_seconds"),
            "prompt_tokens": sorted({run.get("prompt_tokens") for run in runs if isinstance(run, dict)}),
            "generation_tokens": sorted({run.get("generation_tokens") for run in runs if isinstance(run, dict)}),
        }
    invariant = {
        "runner": {
            "id": runner.get("id"),
            "version": runner.get("version"),
            "shape": runner.get("shape"),
            **({"argv_template": runner.get("argv_template")} if external_runner else {}),
        },
        "source": {
            "id": target_model.get("source_id"),
            "revision": target_model.get("source_revision"),
        },
        "invocation": invocation_invariant,
        "target_sha256": target.get("sha256"),
        "workload_sha256": workload.get("sha256"),
        "primary_metric": comparison.get("primary_metric"),
        "operating_system": {
            "platform": environment.get("platform"),
            "macos_version": environment.get("macos_version"),
        },
        "sample_shape": sample_shape,
    }
    variant = {
        "target_id": target_model.get("id"),
        "target_revision": target_model.get("revision"),
        "enabled_methods": receipt.get("enabled_methods", []),
        "config": receipt.get("variant_config", receipt.get("config_notes", {})),
        "invocation_arguments": invocation_variant,
    }
    return {
        "invariant": invariant,
        "invariant_sha256": canonical_hash(invariant),
        "variant": variant,
        "variant_sha256": canonical_hash(variant),
    }


def build_experiment_fingerprint(
    receipt: dict[str, Any],
    *,
    receipt_sha256: str,
    root: Path,
) -> dict[str, Any]:
    """Build the complete promotion identity carried from receipt to public claim.

    The receipt digest is intentionally supplied by the caller because a digest of
    canonicalized JSON is not the same thing as the SHA-256 of the reviewed receipt
    artifact.  Promotion identity binds both the experiment descriptors and the
    measured evidence, so a coherent rewrite of results cannot retain an old
    fingerprint.
    """
    if not isinstance(receipt_sha256, str) or HEX_DIGEST_RE.fullmatch(receipt_sha256) is None:
        raise SkillError("experiment fingerprint requires the candidate receipt SHA-256")
    models = receipt.get("models") if isinstance(receipt.get("models"), dict) else {}
    model_identity = {
        str(role): {
            key: model.get(key)
            for key in ("id", "revision", "lineage_id", "source_id", "source_revision")
        }
        for role, model in sorted(models.items(), key=lambda item: str(item[0]))
        if isinstance(model, dict)
    }
    target = receipt.get("target") if isinstance(receipt.get("target"), dict) else {}
    workload = receipt.get("workload") if isinstance(receipt.get("workload"), dict) else {}
    comparison = receipt.get("comparison") if isinstance(receipt.get("comparison"), dict) else {}
    quality = receipt.get("quality") if isinstance(receipt.get("quality"), dict) else {}
    quality_artifact = quality.get("artifact") if isinstance(quality.get("artifact"), dict) else {}
    quality_payload = load_json_artifact(root.resolve(), quality_artifact)
    quality_metric = quality_payload.get("metric") if isinstance(quality_payload, dict) else None
    quality_provenance = quality_payload.get("provenance") if isinstance(quality_payload, dict) else None
    if not isinstance(quality_metric, dict) or not isinstance(quality_provenance, dict):
        raise SkillError("experiment fingerprint requires controlled quality contract evidence")
    quality_contract_identity = {
        "validator": quality_payload.get("validator"),
        "metric": quality_metric,
        "provenance_mode": quality_provenance.get("mode"),
    }
    metrics = receipt_metrics(receipt)
    measured_runs = []
    for run in receipt.get("runs", []) if isinstance(receipt.get("runs"), list) else []:
        if not isinstance(run, dict):
            continue
        measured_runs.append({
            "run": run.get("run"),
            "metrics": {metric: run.get(metric) for metric in metrics},
            "raw_output": run.get("raw_output"),
        })
    payload = {
        "candidate_receipt_sha256": receipt_sha256,
        "models": model_identity,
        "target": {
            "descriptor": target.get("descriptor"),
            "sha256": target.get("sha256"),
        },
        "workload": {
            "id": workload.get("id"),
            "artifacts": workload.get("artifacts"),
            "parameters": workload.get("parameters"),
            "sha256": workload.get("sha256"),
        },
        "experiment": receipt.get("experiment"),
        "primary_metric": comparison.get("primary_metric"),
        "candidate_baseline_binding": {
            "role": comparison.get("role"),
            "baseline_receipt": comparison.get("baseline_receipt"),
            "baseline_sha256": comparison.get("baseline_sha256"),
        },
        "enabled_methods": receipt.get("enabled_methods"),
        "aggregate": receipt.get("aggregate"),
        "measured_runs": measured_runs,
        "quality": {
            "status": quality.get("status"),
            "artifact": quality_artifact,
            "result_sha256": quality_artifact.get("sha256"),
            "contract_identity": quality_contract_identity,
        },
    }
    return {
        "schema_version": 2,
        "sha256": canonical_hash(payload),
        "payload": payload,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float | int]) -> dict[str, float | int]:
    return {"median": stable_median(values), "min": min(values), "max": max(values)}


def recompute_aggregates(
    runs: list[dict[str, Any]],
    metrics: tuple[str, ...] = METRICS,
) -> dict[str, dict[str, float | int]]:
    if not isinstance(runs, list) or not runs:
        raise SkillError("benchmark receipt must contain non-empty raw runs")
    result: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        values: list[float | int] = []
        for run in runs:
            value = run.get(metric) if isinstance(run, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SkillError(f"benchmark run has invalid {metric}")
            values.append(value)
        result[metric] = summarize(values)
    return result


def aggregates_match(stored: Any, recomputed: dict[str, dict[str, float | int]]) -> bool:
    if not isinstance(stored, dict):
        return False
    for metric, summary in recomputed.items():
        candidate = stored.get(metric)
        if not isinstance(candidate, dict):
            return False
        for key, expected in summary.items():
            actual = candidate.get(key)
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                return False
            if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
                return False
    return True


def recomputed_median_ratios(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float] | None:
    try:
        metrics = receipt_metrics(candidate)
        if receipt_metrics(baseline) != metrics:
            return None
        candidate_aggregate = recompute_aggregates(candidate["runs"], metrics)
        baseline_aggregate = recompute_aggregates(baseline["runs"], metrics)
        if metrics == EXTERNAL_METRICS:
            candidate_wall = float(candidate_aggregate["wall_seconds"]["median"])
            baseline_wall = float(baseline_aggregate["wall_seconds"]["median"])
            if min(candidate_wall, baseline_wall) <= 0:
                return None
            return {"wall_seconds_inverse": baseline_wall / candidate_wall}
        candidate_decode = float(candidate_aggregate["generation_tps"]["median"])
        baseline_decode = float(baseline_aggregate["generation_tps"]["median"])
        candidate_prefill = float(candidate_aggregate["prompt_tps"]["median"])
        baseline_prefill = float(baseline_aggregate["prompt_tps"]["median"])
        candidate_ttft = float(candidate_aggregate["ttft_proxy_s"]["median"])
        baseline_ttft = float(baseline_aggregate["ttft_proxy_s"]["median"])
        candidate_memory = float(candidate_aggregate["peak_memory_gb"]["median"])
        baseline_memory = float(baseline_aggregate["peak_memory_gb"]["median"])
        if min(candidate_decode, baseline_decode, candidate_prefill, baseline_prefill, candidate_ttft, baseline_ttft, candidate_memory, baseline_memory) <= 0:
            return None
        return {
            "decode_tps": candidate_decode / baseline_decode,
            "prefill_tps": candidate_prefill / baseline_prefill,
            "ttft_proxy_inverse": baseline_ttft / candidate_ttft,
            "peak_memory_inverse": baseline_memory / candidate_memory,
        }
    except (KeyError, SkillError, TypeError, ValueError):
        return None


def stability_report(runs: list[dict[str, Any]], primary_metric: str, max_cv: float, min_runs: int) -> dict[str, Any]:
    values = [float(run[primary_metric]) for run in runs]
    mean = stable_mean(values)
    stdev = stable_pstdev(values)
    cv = stdev / mean if mean else math.inf
    return {
        "primary_metric": primary_metric,
        "run_count": len(values),
        "min_runs": min_runs,
        "mean": mean,
        "median": stable_median(values),
        "stdev": stdev,
        "cv": cv,
        "max_cv": max_cv,
        "passes": len(values) >= min_runs and cv <= max_cv,
    }


def all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in all_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in all_strings(child)]
    return []


def contains_serialized_secret(value: Any) -> bool:
    if isinstance(value, str):
        return redact_secret_text(value) != value
    if isinstance(value, dict):
        for key, child in value.items():
            if SENSITIVE_KEY_RE.search(str(key)) and child is not None and child != "" and child != "[REDACTED]":
                return True
            if contains_serialized_secret(child):
                return True
    elif isinstance(value, list):
        return redact_secrets(value) != value
    elif isinstance(value, tuple):
        return any(contains_serialized_secret(child) for child in value)
    return False


def resolve_artifact(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    resolved_root = root.resolve()
    unresolved = resolved_root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            return None
    path = unresolved.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError:
        return None
    return path


def check_artifact(root: Path, artifact: Any) -> tuple[bool, str | None]:
    if not isinstance(artifact, dict):
        return False, "missing"
    path = resolve_artifact(root, artifact.get("path"))
    expected_hash = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    if path is None or not path.is_file():
        return False, "missing"
    if not isinstance(expected_hash, str) or not HEX_DIGEST_RE.fullmatch(expected_hash):
        return False, "digest"
    if file_sha256(path) != expected_hash:
        return False, "digest"
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or path.stat().st_size != expected_size:
        return False, "digest"
    return True, None


def _controlled_quality_candidate_artifact(
    root: Path,
    receipt: dict[str, Any],
) -> dict[str, Any] | None:
    quality = receipt.get("quality")
    quality_payload = load_json_artifact(
        root,
        quality.get("artifact") if isinstance(quality, dict) else None,
    )
    provenance = quality_payload.get("provenance") if isinstance(quality_payload, dict) else None
    if not isinstance(provenance, dict) or provenance.get("mode") != "controlled-exact-output-v1":
        return None
    contract = load_json_artifact(root, provenance.get("input"))
    candidate = contract.get("candidate_artifact") if isinstance(contract, dict) else None
    if not isinstance(candidate, dict) or set(candidate) != {"path", "sha256", "size_bytes"}:
        return None
    return candidate


def _external_report_matches(root: Path, receipt: dict[str, Any], run: dict[str, Any]) -> bool:
    artifact = run.get("raw_output")
    if (
        not isinstance(artifact, dict)
        or isinstance(artifact.get("size_bytes"), bool)
        or not isinstance(artifact.get("size_bytes"), int)
        or artifact["size_bytes"] > MAX_EXTERNAL_REPORT_BYTES
    ):
        return False
    path = resolve_artifact(root, artifact.get("path")) if isinstance(artifact, dict) else None
    if path is None:
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    report_runs = report.get("runs")
    report_warmups = report.get("warmups")
    receipt_runs = receipt.get("runs")
    requested_runs = report.get("requested_runs")
    warmup_count = report.get("warmup_count")
    timeout_seconds = report.get("timeout_seconds")
    receipt_timeout_seconds = receipt.get("timeout_seconds")
    if (
        report.get("schema_version") != 1
        or report.get("command") != receipt.get("command")
        or report.get("cwd") != receipt.get("cwd")
        or report.get("ok") is not True
        or report.get("environment_overrides") != {}
        or not isinstance(report_runs, list)
        or not isinstance(report_warmups, list)
        or not isinstance(receipt_runs, list)
        or isinstance(requested_runs, bool)
        or not isinstance(requested_runs, int)
        or isinstance(warmup_count, bool)
        or not isinstance(warmup_count, int)
        or requested_runs != len(report_runs)
        or requested_runs != receipt.get("measured_runs")
        or warmup_count != len(report_warmups)
        or warmup_count != receipt.get("warmup_runs")
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0 < float(timeout_seconds) <= MAX_EXTERNAL_TIMEOUT_SECONDS
        or isinstance(receipt_timeout_seconds, bool)
        or not isinstance(receipt_timeout_seconds, (int, float))
        or not math.isfinite(float(receipt_timeout_seconds))
        or not math.isclose(
            float(timeout_seconds),
            float(receipt_timeout_seconds),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not 0 < requested_runs <= MAX_EXTERNAL_RUNS
        or not 0 <= warmup_count <= MAX_EXTERNAL_WARMUPS
        or len(report_runs) != len(receipt_runs)
        or contains_serialized_secret(report)
        or any(EPHEMERAL_PATH_RE.search(value) for value in all_strings(report))
    ):
        return False
    system = receipt.get("target", {}).get("descriptor", {}).get("benchmark_system")
    environment_sha256 = receipt.get("target", {}).get("descriptor", {}).get(
        "execution_environment_sha256"
    )
    interpreter = receipt.get("target", {}).get("descriptor", {}).get("interpreter")
    expected_quality_output = _controlled_quality_candidate_artifact(root, receipt)
    if (
        not isinstance(system, dict)
        or report.get("system") != system
        or not isinstance(environment_sha256, str)
        or HEX_DIGEST_RE.fullmatch(environment_sha256) is None
        or report.get("execution_environment_sha256") != environment_sha256
        or not isinstance(interpreter, dict)
        or report.get("interpreter") != interpreter
        or expected_quality_output is None
    ):
        return False
    for warmup in report_warmups:
        wall = warmup.get("wall_seconds") if isinstance(warmup, dict) else None
        if (
            not isinstance(warmup, dict)
            or warmup.get("phase") != "warmup"
            or warmup.get("returncode") != 0
            or warmup.get("timed_out") is not False
            or isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) <= 0
        ):
            return False
    values: list[float] = []
    attested_report = receipt.get("runner", {}).get("id") == ATTESTED_RUNNER_ID
    for index, (report_run, receipt_run) in enumerate(zip(report_runs, receipt_runs), start=1):
        if not isinstance(report_run, dict) or not isinstance(receipt_run, dict):
            return False
        wall = report_run.get("wall_seconds")
        if (
            report_run.get("phase") != "measure"
            or report_run.get("returncode") != 0
            or report_run.get("timed_out") is not False
            or isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) <= 0
            or receipt_run.get("run") != index
            or report_run.get("quality_output") != expected_quality_output
            or (
                attested_report
                and report_run.get("execution_attestation")
                != receipt_run.get("execution_attestation")
            )
            or not math.isclose(float(receipt_run.get("wall_seconds", math.inf)), float(wall), rel_tol=1e-12, abs_tol=1e-12)
        ):
            return False
        values.append(float(wall))
    summary = report.get("summary")
    expected = {
        "successful_runs": len(values),
        "wall_seconds_min": min(values),
        "wall_seconds_median": stable_median(values),
        "wall_seconds_mean": stable_mean(values),
        "wall_seconds_p95": sorted(values)[0] if len(values) == 1 else (
            lambda ordered, position: ordered[int(position)] * (1 - (position - int(position)))
            + ordered[min(int(position) + 1, len(ordered) - 1)] * (position - int(position))
        )(sorted(values), (len(values) - 1) * 0.95),
        "wall_seconds_max": max(values),
    }
    if not isinstance(summary, dict):
        return False
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, int):
            if actual != expected_value:
                return False
        elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected_value), rel_tol=1e-12, abs_tol=1e-12
        ):
            return False
    run_index = run.get("run")
    return isinstance(run_index, int) and 1 <= run_index <= len(report_runs)


def raw_metrics_match(root: Path, receipt: dict[str, Any], run: dict[str, Any]) -> bool:
    if receipt_metrics(receipt) == EXTERNAL_METRICS:
        return _external_report_matches(root, receipt, run)
    artifact = run.get("raw_output")
    path = resolve_artifact(root, artifact.get("path")) if isinstance(artifact, dict) else None
    if path is None:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    prompt = PROMPT_RE.search(text)
    generation = GENERATION_RE.search(text)
    memory = PEAK_MEMORY_RE.search(text)
    if prompt is None or generation is None or memory is None:
        return False
    expected = {
        "prompt_tokens": int(prompt.group(1)),
        "prompt_tps": float(prompt.group(2)),
        "generation_tokens": int(generation.group(1)),
        "generation_tps": float(generation.group(2)),
        "peak_memory_gb": float(memory.group(1)),
    }
    for key, value in expected.items():
        actual = run.get(key)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        if not math.isclose(float(actual), float(value), rel_tol=1e-12, abs_tol=1e-12):
            return False
    return math.isclose(
        float(run.get("ttft_proxy_s", math.inf)),
        expected["prompt_tokens"] / expected["prompt_tps"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def quality_check_result(check: Any) -> bool | None:
    if not isinstance(check, dict) or not isinstance(check.get("id"), str) or not check["id"]:
        return None
    comparator = check.get("comparator")
    observed = check.get("observed")
    threshold = check.get("threshold")
    if (
        comparator not in {"gte", "gt", "lte", "lt", "eq"}
        or isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(observed))
        or not math.isfinite(float(threshold))
    ):
        return None
    result = {
        "gte": observed >= threshold,
        "gt": observed > threshold,
        "lte": observed <= threshold,
        "lt": observed < threshold,
        "eq": math.isclose(float(observed), float(threshold), rel_tol=1e-12, abs_tol=1e-12),
    }[comparator]
    return result if check.get("passed") is result else None


def quality_check_passes(check: Any) -> bool:
    return quality_check_result(check) is True


QUALITY_RECEIPT_ROOTS = {
    "aggregate",
    "comparison",
    "experiment",
    "models",
    "runs",
    "target",
    "workload",
}


def _quality_source_value(root: Any, path: Any) -> Any:
    if not isinstance(path, list) or not path or len(path) > 16:
        raise SkillError("quality source path must be a non-empty list with at most 16 segments")
    value = root
    for segment in path:
        if isinstance(value, dict) and isinstance(segment, str) and segment in value:
            value = value[segment]
        elif (
            isinstance(value, list)
            and isinstance(segment, int)
            and not isinstance(segment, bool)
            and 0 <= segment < len(value)
        ):
            value = value[segment]
        else:
            raise SkillError(f"quality source path does not resolve: {path!r}")
    return value


def evaluate_declarative_quality_contract(
    receipt: dict[str, Any],
    contract: Any,
) -> dict[str, Any]:
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise SkillError("declarative quality contract must use schema_version 1")
    if contract.get("validator") != DECLARATIVE_QUALITY_VALIDATOR:
        raise SkillError("declarative quality contract uses an unsupported validator")
    evidence = contract.get("evidence")
    if not isinstance(evidence, dict):
        raise SkillError("declarative quality contract evidence must be an object")
    checks = contract.get("checks")
    if not isinstance(checks, list) or not checks or len(checks) > 128:
        raise SkillError("declarative quality contract must contain 1-128 checks")

    canonical_checks: list[dict[str, Any]] = []
    has_external_quality_evidence = False
    seen_ids: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            raise SkillError("declarative quality check must be an object")
        check_id = check.get("id")
        source = check.get("source")
        comparator = check.get("comparator")
        threshold = check.get("threshold")
        if not isinstance(check_id, str) or not check_id or check_id in seen_ids:
            raise SkillError("declarative quality check ids must be unique non-empty strings")
        seen_ids.add(check_id)
        if not isinstance(source, dict):
            raise SkillError(f"declarative quality check {check_id} source must be an object")
        source_kind = source.get("kind")
        source_path = source.get("path")
        reducer = source.get("reducer")
        if source_kind == "external-json":
            source_root = evidence
            has_external_quality_evidence = True
        elif source_kind == "receipt":
            if not isinstance(source_path, list) or not source_path or source_path[0] not in QUALITY_RECEIPT_ROOTS:
                raise SkillError(f"declarative quality check {check_id} uses a disallowed receipt root")
            source_root = receipt
        else:
            raise SkillError(f"declarative quality check {check_id} uses an unsupported source kind")
        raw_value = _quality_source_value(source_root, source_path)
        if reducer == "value":
            observed = raw_value
        elif reducer == "length" and isinstance(raw_value, (dict, list, str)):
            observed = len(raw_value)
        else:
            raise SkillError(f"declarative quality check {check_id} uses an invalid reducer")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or comparator not in {"gte", "gt", "lte", "lt", "eq"}
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise SkillError(f"declarative quality check {check_id} must compare finite numbers")
        evaluated = {
            "id": check_id,
            "source": {
                "kind": source_kind,
                "path": source_path,
                "reducer": reducer,
            },
            "comparator": comparator,
            "observed": observed,
            "threshold": threshold,
        }
        result = {
            "gte": observed >= threshold,
            "gt": observed > threshold,
            "lte": observed <= threshold,
            "lt": observed < threshold,
            "eq": math.isclose(float(observed), float(threshold), rel_tol=1e-12, abs_tol=1e-12),
        }[comparator]
        canonical_checks.append({**evaluated, "passed": result})
    if not has_external_quality_evidence:
        raise SkillError(
            "declarative quality contract requires at least one digest-bound external-json quality check"
        )
    return {
        "schema_version": 2,
        "validator": DECLARATIVE_QUALITY_VALIDATOR,
        "status": "pass" if all(check["passed"] for check in canonical_checks) else "fail",
        "checks": canonical_checks,
    }


def build_declarative_quality_payload(
    receipt: dict[str, Any],
    input_artifact: dict[str, Any],
    contract: Any,
) -> dict[str, Any]:
    evaluation = evaluate_declarative_quality_contract(receipt, contract)
    provenance = {
        "mode": "declarative-quality-v1",
        "input": input_artifact,
        "evaluation_sha256": canonical_hash(evaluation),
    }
    return {
        **evaluation,
        "bindings": expected_quality_bindings(receipt),
        "provenance": provenance,
    }


def _read_controlled_quality_artifact(root: Path, artifact: Any, label: str) -> bytes:
    valid, reason = check_artifact(root, artifact)
    if not valid or not isinstance(artifact, dict):
        raise SkillError(f"controlled quality {label} artifact is invalid: {reason or 'invalid'}")
    size = artifact.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size > MAX_QUALITY_ARTIFACT_BYTES:
        raise SkillError(
            f"controlled quality {label} artifact exceeds {MAX_QUALITY_ARTIFACT_BYTES} bytes"
        )
    path = resolve_artifact(root, artifact.get("path"))
    if path is None:
        raise SkillError(f"controlled quality {label} artifact path is invalid")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SkillError(f"could not read controlled quality {label} artifact: {exc}") from exc


def evaluate_controlled_exact_output_quality_contract(
    root: Path,
    contract: Any,
) -> dict[str, Any]:
    """Recompute a fixed exact-output task metric from two bound artifacts."""
    required_keys = {
        "schema_version",
        "validator",
        "metric",
        "reference_artifact",
        "candidate_artifact",
    }
    if not isinstance(contract, dict) or set(contract) != required_keys:
        raise SkillError("controlled exact-output quality contract has an invalid field set")
    if contract.get("schema_version") != 2:
        raise SkillError("controlled exact-output quality contract must use schema_version 2")
    if not exact_output_validator_descriptor_valid(contract.get("validator")):
        raise SkillError("controlled exact-output quality contract uses an unsupported validator")
    if contract.get("metric") != "exact-output-parity":
        raise SkillError("controlled exact-output quality contract metric must be exact-output-parity")
    reference_artifact = contract.get("reference_artifact")
    candidate_artifact = contract.get("candidate_artifact")
    if not isinstance(reference_artifact, dict) or not isinstance(candidate_artifact, dict):
        raise SkillError("controlled exact-output quality contract requires artifact mappings")
    if reference_artifact.get("path") == candidate_artifact.get("path"):
        raise SkillError("controlled exact-output quality artifacts must use distinct paths")
    reference = _read_controlled_quality_artifact(root, reference_artifact, "reference")
    candidate = _read_controlled_quality_artifact(root, candidate_artifact, "candidate")
    exact_match = candidate == reference
    return {
        "schema_version": 3,
        "validator": EXACT_OUTPUT_QUALITY_VALIDATOR,
        "status": "pass" if exact_match else "fail",
        "metric": {
            "id": "exact-output-parity",
            "reference_sha256": reference_artifact.get("sha256"),
            "candidate_sha256": candidate_artifact.get("sha256"),
            "reference_size_bytes": len(reference),
            "candidate_size_bytes": len(candidate),
            "exact_match": exact_match,
        },
    }


def build_controlled_exact_output_quality_payload(
    receipt: dict[str, Any],
    input_artifact: dict[str, Any],
    contract: Any,
    root: Path,
) -> dict[str, Any]:
    evaluation = evaluate_controlled_exact_output_quality_contract(root, contract)
    return {
        **evaluation,
        "bindings": expected_quality_bindings(receipt),
        "provenance": {
            "mode": "controlled-exact-output-v1",
            "input": input_artifact,
            "evaluation_sha256": canonical_hash(evaluation),
        },
    }


def expected_quality_bindings(receipt: dict[str, Any]) -> dict[str, Any]:
    runs = receipt.get("runs", [])
    raw_digests = [
        run.get("raw_output", {}).get("sha256")
        for run in runs
        if isinstance(run, dict)
    ]
    return {
        "receipt_label": receipt.get("label"),
        "model_lineage_sha256": canonical_hash(receipt.get("models", {})),
        "workload_sha256": receipt.get("workload", {}).get("sha256"),
        "baseline_receipt_sha256": receipt.get("comparison", {}).get("baseline_sha256"),
        "raw_output_set_sha256": canonical_hash(raw_digests),
    }


def build_quality_evaluator_input(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bindings": expected_quality_bindings(receipt),
        "models": receipt.get("models", {}),
        "target": receipt.get("target", {}),
        "workload": receipt.get("workload", {}),
        "comparison": receipt.get("comparison", {}),
        "experiment": receipt.get("experiment", {}),
        "runs": receipt.get("runs", []),
        "aggregate": receipt.get("aggregate", {}),
    }


def build_quality_evaluator_invocation(
    evaluator: dict[str, Any],
    input_artifact: dict[str, Any],
) -> dict[str, Any]:
    descriptor = {
        "runner": {
            "id": "python-script",
            "version": 1,
            "shape": "python evaluator.py input.json",
        },
        "evaluator_sha256": evaluator.get("sha256"),
        "input_sha256": input_artifact.get("sha256"),
    }
    return {**descriptor, "sha256": canonical_hash(descriptor), "exit_code": 0}


def load_json_artifact(root: Path, artifact: Any) -> Any:
    valid, _ = check_artifact(root, artifact)
    if not valid or not isinstance(artifact, dict):
        return None
    path = resolve_artifact(root, artifact.get("path"))
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path else None
    except (OSError, json.JSONDecodeError):
        return None


def quality_evaluator_provenance_valid(
    receipt: dict[str, Any],
    quality_payload: dict[str, Any],
    root: Path,
) -> bool:
    provenance = quality_payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("mode") != "checked-in-evaluator-v1":
        return False
    evaluator = provenance.get("evaluator")
    input_artifact = provenance.get("input")
    output_artifact = provenance.get("output")
    stderr_artifact = provenance.get("stderr")
    for artifact in (evaluator, input_artifact, output_artifact, stderr_artifact):
        valid, _ = check_artifact(root, artifact)
        if not valid:
            return False
    evaluator_path = resolve_artifact(root, evaluator.get("path")) if isinstance(evaluator, dict) else None
    if evaluator_path is None:
        return False
    try:
        relative_evaluator = evaluator_path.relative_to(root.resolve())
    except ValueError:
        return False
    expected_name = f"{evaluator.get('sha256')}.py"
    if relative_evaluator.parts[:2] != ("quality", "evaluators") or evaluator_path.name != expected_name:
        return False
    input_payload = load_json_artifact(root, input_artifact)
    output_payload = load_json_artifact(root, output_artifact)
    if input_payload != build_quality_evaluator_input(receipt):
        return False
    if not isinstance(output_payload, dict):
        return False
    if (
        output_payload.get("schema_version") != 2
        or output_payload.get("validator") != {"id": "mlx-benchmark-quality", "version": 1}
        or output_payload.get("status") != quality_payload.get("status")
        or output_payload.get("checks") != quality_payload.get("checks")
    ):
        return False
    return provenance.get("invocation") == build_quality_evaluator_invocation(evaluator, input_artifact)


def declarative_quality_payload_valid(
    receipt: dict[str, Any],
    quality_payload: Any,
    root: Path,
) -> bool:
    if not isinstance(quality_payload, dict):
        return False
    provenance = quality_payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("mode") != "declarative-quality-v1":
        return False
    input_artifact = provenance.get("input")
    valid, _ = check_artifact(root, input_artifact)
    if not valid or not isinstance(input_artifact, dict):
        return False
    input_path = resolve_artifact(root, input_artifact.get("path"))
    if input_path is None:
        return False
    try:
        relative_input = input_path.relative_to(root.resolve())
    except ValueError:
        return False
    expected_name = f"{input_artifact.get('sha256')}.json"
    if relative_input.parts[:2] != ("quality", "inputs") or input_path.name != expected_name:
        return False
    contract = load_json_artifact(root, input_artifact)
    try:
        expected = build_declarative_quality_payload(receipt, input_artifact, contract)
    except SkillError:
        return False
    return quality_payload == expected


def controlled_exact_output_quality_payload_valid(
    receipt: dict[str, Any],
    quality_payload: Any,
    root: Path,
) -> bool:
    if not isinstance(quality_payload, dict):
        return False
    provenance = quality_payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("mode") != "controlled-exact-output-v1":
        return False
    input_artifact = provenance.get("input")
    valid, _ = check_artifact(root, input_artifact)
    if not valid or not isinstance(input_artifact, dict):
        return False
    input_path = resolve_artifact(root, input_artifact.get("path"))
    if input_path is None:
        return False
    try:
        relative_input = input_path.relative_to(root.resolve())
    except ValueError:
        return False
    expected_name = f"{input_artifact.get('sha256')}.json"
    if relative_input.parts[:2] != ("quality", "inputs") or input_path.name != expected_name:
        return False
    contract = load_json_artifact(root, input_artifact)
    try:
        expected = build_controlled_exact_output_quality_payload(
            receipt,
            input_artifact,
            contract,
            root,
        )
    except SkillError:
        return False
    return quality_payload == expected


def _artifact_identity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    identity = {key: value.get(key) for key in ("path", "sha256", "size_bytes")}
    if (
        not isinstance(identity["path"], str)
        or not identity["path"]
        or not isinstance(identity["sha256"], str)
        or HEX_DIGEST_RE.fullmatch(identity["sha256"]) is None
        or isinstance(identity["size_bytes"], bool)
        or not isinstance(identity["size_bytes"], int)
        or identity["size_bytes"] < 0
    ):
        return None
    return identity


def unreferenced_attestation_files(
    root: Path,
    receipts: dict[str, dict[str, Any]],
) -> list[str]:
    """Return attestation-tree files unreachable from receipt artifact bindings."""
    referenced: set[str] = set()
    pending_snapshots: list[dict[str, Any]] = []

    def record(artifact: Any) -> None:
        identity = _artifact_identity(artifact)
        if identity is None or not identity["path"].startswith("attestations/"):
            return
        referenced.add(identity["path"])
        if identity["path"].endswith(("/challenge.json", "/evidence.json")):
            pending_snapshots.append(identity)

    def record_execution_attestation(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key in ("challenge", "evidence", "output"):
            record(value.get(key))

    def read_json(artifact: Any, *, limit: int) -> Any:
        identity = _artifact_identity(artifact)
        if identity is None or identity["size_bytes"] > limit:
            return None
        path = resolve_artifact(root, identity["path"])
        if path is None or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    for receipt in receipts.values():
        runs = receipt.get("runs")
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            record_execution_attestation(run.get("execution_attestation"))
            report = read_json(
                run.get("raw_output"),
                limit=MAX_EXTERNAL_REPORT_BYTES,
            )
            if not isinstance(report, dict):
                continue
            for phase_runs in (report.get("warmups"), report.get("runs")):
                if not isinstance(phase_runs, list):
                    continue
                for phase_run in phase_runs:
                    if isinstance(phase_run, dict):
                        record_execution_attestation(
                            phase_run.get("execution_attestation")
                        )

    inspected_snapshots: set[str] = set()
    while pending_snapshots:
        identity = pending_snapshots.pop()
        relative = identity["path"]
        if relative in inspected_snapshots:
            continue
        inspected_snapshots.add(relative)
        payload = read_json(identity, limit=MAX_ATTESTATION_ARTIFACT_BYTES)
        if not isinstance(payload, dict):
            continue
        if relative.endswith("/challenge.json"):
            record(payload.get("quality_contract"))
        elif relative.endswith("/evidence.json"):
            dependencies = payload.get("dependencies")
            if isinstance(dependencies, list):
                for dependency in dependencies:
                    if isinstance(dependency, dict):
                        record(dependency.get("artifact"))

    attestation_root = root / "attestations"
    if not attestation_root.exists():
        return []
    actual = {
        path.relative_to(root).as_posix()
        for path in attestation_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    return sorted(actual - referenced)


def _load_bounded_attestation_json(
    root: Path,
    artifact: Any,
    *,
    limit: int,
) -> dict[str, Any] | None:
    identity = _artifact_identity(artifact)
    if identity is None or identity["size_bytes"] > limit:
        return None
    valid, _ = check_artifact(root, identity)
    path = resolve_artifact(root, identity["path"])
    if not valid or path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def expected_attested_workload(receipt: dict[str, Any]) -> dict[str, Any] | None:
    workload = receipt.get("workload")
    models = receipt.get("models")
    target_model = models.get("target") if isinstance(models, dict) else None
    variant = receipt.get("variant_config")
    artifacts = workload.get("artifacts") if isinstance(workload, dict) else None
    if (
        not isinstance(target_model, dict)
        or not isinstance(variant, dict)
        or not isinstance(artifacts, list)
        or len(artifacts) != 2
        or artifacts[0].get("role") != "runner"
        or artifacts[1].get("role") != "input"
    ):
        return None
    input_artifact = _artifact_identity(artifacts[1])
    if input_artifact is None:
        return None
    return {
        "id": ATTESTED_WORKLOAD_ID,
        "model": {
            "id": target_model.get("id"),
            "revision": target_model.get("revision"),
        },
        "input": input_artifact,
        "parameters": ATTESTED_WORKLOAD_PARAMETERS,
        "variant": {
            "mode": variant.get("mode"),
            "quality_output_path": variant.get("quality_output_path"),
        },
    }


def attested_execution_state(
    receipt: dict[str, Any],
    root: Path,
) -> tuple[bool, str | None]:
    runner = receipt.get("runner")
    if not isinstance(runner, dict) or runner.get("id") != ATTESTED_RUNNER_ID:
        return False, "execution-semantics-unattested"

    trusted_path = DEFAULT_ROOT / ATTESTED_RUNNER_PATH
    if trusted_path.is_symlink() or not trusted_path.is_file():
        return False, "attestation-runner-digest-mismatch"
    trusted_runner = {
        "path": ATTESTED_RUNNER_PATH,
        "sha256": file_sha256(trusted_path),
        "size_bytes": trusted_path.stat().st_size,
    }
    workload = receipt.get("workload")
    artifacts = workload.get("artifacts") if isinstance(workload, dict) else None
    runner_artifact = _artifact_identity(artifacts[0]) if isinstance(artifacts, list) and artifacts else None
    if (
        runner.get("implementation") != trusted_runner
        or runner_artifact != trusted_runner
        or check_artifact(root, trusted_runner)[0] is not True
    ):
        return False, "attestation-runner-digest-mismatch"

    label = receipt.get("label")
    models = receipt.get("models")
    target_model = models.get("target") if isinstance(models, dict) else None
    variant = receipt.get("variant_config")
    if not isinstance(label, str) or not isinstance(target_model, dict) or not isinstance(variant, dict):
        return False, "attestation-model-mismatch"
    mode = variant.get("mode")
    variant_rule = ATTESTED_VARIANTS.get(str(mode))
    expected_variant_keys = {
        "dtype_policy",
        "mode",
        "quality_output_path",
        "weights_bytes",
        "weights_sha256",
        "attestation_challenge_path",
        "attestation_output_path",
    }
    if (
        variant_rule is None
        or target_model.get("id") != ATTESTED_MODEL_ID
        or target_model.get("revision") != variant_rule["revision"]
        or set(variant) != expected_variant_keys
        or variant.get("dtype_policy") != variant_rule["dtype_policy"]
        or variant.get("weights_sha256") != variant_rule["revision"]
        or variant.get("weights_bytes") != variant_rule["weights_bytes"]
        or variant.get("quality_output_path") != f"quality/outputs/{label}/result.json"
        or variant.get("attestation_challenge_path")
        != f"attestations/{label}/current-challenge.json"
        or variant.get("attestation_output_path")
        != f"attestations/{label}/current-evidence.json"
    ):
        return False, "attestation-model-mismatch"

    expected_workload = expected_attested_workload(receipt)
    if (
        expected_workload is None
        or not isinstance(workload, dict)
        or workload.get("id") != ATTESTED_WORKLOAD_ID
        or workload.get("parameters") != ATTESTED_WORKLOAD_PARAMETERS
        or len(workload.get("artifacts", [])) != 2
        or check_artifact(root, workload["artifacts"][1])[0] is not True
    ):
        return False, "attestation-workload-mismatch"

    quality = receipt.get("quality")
    quality_payload = load_json_artifact(
        root,
        quality.get("artifact") if isinstance(quality, dict) else None,
    )
    provenance = quality_payload.get("provenance") if isinstance(quality_payload, dict) else None
    stored_contract = load_json_artifact(
        root,
        provenance.get("input") if isinstance(provenance, dict) else None,
    )
    if not isinstance(stored_contract, dict):
        return False, "attestation-challenge-mismatch"

    command = receipt.get("command")
    runs = receipt.get("runs")
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) and item for item in command)
        or not isinstance(runs, list)
        or not runs
    ):
        return False, "attestation-evidence-missing"

    seen_nonces: set[str] = set()
    seen_challenges: set[str] = set()
    seen_evidence: set[str] = set()
    for run_index, run in enumerate(runs, start=1):
        attestation = run.get("execution_attestation") if isinstance(run, dict) else None
        if not isinstance(attestation, dict) or set(attestation) != {"challenge", "evidence", "output"}:
            return False, "attestation-evidence-missing"
        prefix = f"attestations/{label}/runs/measure-{run_index:03d}"
        expected_paths = {
            "challenge": f"{prefix}/challenge.json",
            "evidence": f"{prefix}/evidence.json",
            "output": f"{prefix}/output.json",
        }
        for key, expected_path in expected_paths.items():
            identity = _artifact_identity(attestation.get(key))
            if identity is None or identity["path"] != expected_path:
                return False, (
                    "attestation-output-digest-mismatch"
                    if key == "output"
                    else "attestation-evidence-digest-mismatch"
                )
            valid, _ = check_artifact(root, identity)
            if not valid:
                return False, (
                    "attestation-output-digest-mismatch"
                    if key == "output"
                    else "attestation-evidence-digest-mismatch"
                )

        challenge_artifact = _artifact_identity(attestation["challenge"])
        evidence_artifact = _artifact_identity(attestation["evidence"])
        output_artifact = _artifact_identity(attestation["output"])
        assert challenge_artifact and evidence_artifact and output_artifact
        challenge = _load_bounded_attestation_json(
            root,
            challenge_artifact,
            limit=MAX_ATTESTATION_CHALLENGE_BYTES,
        )
        if challenge is None:
            return False, "attestation-challenge-mismatch"
        contract_artifact = challenge.get("quality_contract")
        expected_challenge = {
            "schema_version": 1,
            "nonce": challenge.get("nonce"),
            "receipt_label": label,
            "phase": "measure",
            "run_index": run_index,
            "command": command,
            "command_sha256": canonical_hash(command),
            "runner_argv_sha256": canonical_hash(command[1:]),
            "quality_contract": contract_artifact,
        }
        nonce = challenge.get("nonce")
        contract_identity = _artifact_identity(contract_artifact)
        contract_payload = _load_bounded_attestation_json(
            root,
            contract_identity,
            limit=MAX_QUALITY_ARTIFACT_BYTES,
        ) if contract_identity is not None else None
        if (
            challenge != expected_challenge
            or not isinstance(nonce, str)
            or HEX_DIGEST_RE.fullmatch(nonce) is None
            or nonce in seen_nonces
            or challenge_artifact["sha256"] in seen_challenges
            or contract_identity is None
            or contract_identity["path"] != f"{prefix}/quality-contract.json"
            or contract_payload != stored_contract
        ):
            return False, "attestation-challenge-mismatch"
        seen_nonces.add(nonce)
        seen_challenges.add(challenge_artifact["sha256"])

        evidence = _load_bounded_attestation_json(
            root,
            evidence_artifact,
            limit=MAX_ATTESTATION_ARTIFACT_BYTES,
        )
        if evidence is None or set(evidence) != {
            "schema_version",
            "adapter",
            "challenge",
            "runner_argv_sha256",
            "workload",
            "model_artifact",
            "dependencies",
            "dependency_set_sha256",
            "output_artifact",
            "evidence_sha256",
        }:
            return False, "attestation-evidence-digest-mismatch"
        evidence_payload = {
            key: value for key, value in evidence.items() if key != "evidence_sha256"
        }
        expected_adapter = {
            "id": ATTESTED_RUNNER_ID,
            "version": 1,
            "implementation": trusted_runner,
        }
        expected_challenge_identity = {
            "sha256": challenge_artifact["sha256"],
            "size_bytes": challenge_artifact["size_bytes"],
        }
        evidence_digest = evidence.get("evidence_sha256")
        if evidence.get("challenge") != expected_challenge_identity:
            return False, "attestation-challenge-mismatch"
        if (
            evidence.get("schema_version") != 1
            or evidence.get("adapter") != expected_adapter
            or evidence.get("runner_argv_sha256") != canonical_hash(command[1:])
            or evidence_digest != canonical_hash(evidence_payload)
            or not isinstance(evidence_digest, str)
            or evidence_digest in seen_evidence
        ):
            return False, "attestation-evidence-digest-mismatch"
        seen_evidence.add(evidence_digest)

        evidence_workload = evidence.get("workload")
        if evidence_workload != {
            "descriptor": expected_workload,
            "sha256": canonical_hash(expected_workload),
        }:
            return False, "attestation-workload-mismatch"
        expected_model_artifact = {
            "path": variant_rule["weights_path"],
            "sha256": variant_rule["revision"],
            "size_bytes": variant_rule["weights_bytes"],
        }
        if evidence.get("model_artifact") != expected_model_artifact:
            return False, "attestation-model-mismatch"

        dependencies = evidence.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies or len(dependencies) > 256:
            return False, "attestation-dependency-digest-mismatch"
        dependency_total = 0
        dependency_scopes: set[str] = set()
        loaded_paths: set[tuple[str, str]] = set()
        loaded_modules: set[str] = set()
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {
                "scope",
                "module_names",
                "loaded_path",
                "artifact",
            }:
                return False, "attestation-dependency-digest-mismatch"
            scope = dependency.get("scope")
            module_names = dependency.get("module_names")
            loaded_path = dependency.get("loaded_path")
            artifact = _artifact_identity(dependency.get("artifact"))
            if (
                scope not in {"mlx-runtime", "port-package", "port-config"}
                or not isinstance(module_names, list)
                or module_names != sorted(set(module_names))
                or not all(isinstance(name, str) and name for name in module_names)
                or not isinstance(loaded_path, str)
                or not loaded_path
                or (scope, loaded_path) in loaded_paths
                or any(name in loaded_modules for name in module_names)
                or artifact is None
                or artifact["path"] != f"attestations/dependencies/{artifact['sha256']}.bin"
                or artifact["size_bytes"] > MAX_ATTESTED_DEPENDENCY_BYTES
                or check_artifact(root, artifact)[0] is not True
            ):
                return False, "attestation-dependency-digest-mismatch"
            if scope == "mlx-runtime" and (
                not module_names
                or not all(name in {"mlx", "_mlx"} or name.startswith(("mlx.", "_mlx.")) for name in module_names)
                or not loaded_path.startswith("runtime/")
            ):
                return False, "attestation-dependency-digest-mismatch"
            if scope == "port-package" and (
                not module_names
                or not all(name in {"model", "config"} for name in module_names)
                or not loaded_path.startswith("port/")
            ):
                return False, "attestation-dependency-digest-mismatch"
            if scope == "port-config" and (
                module_names != [] or loaded_path != "port/config.json"
            ):
                return False, "attestation-dependency-digest-mismatch"
            dependency_total += artifact["size_bytes"]
            if dependency_total > MAX_ATTESTED_DEPENDENCY_SET_BYTES:
                return False, "attestation-dependency-digest-mismatch"
            dependency_scopes.add(scope)
            loaded_paths.add((scope, loaded_path))
            loaded_modules.update(module_names)
        if (
            dependency_scopes != {"mlx-runtime", "port-package", "port-config"}
            or not {"model", "config"}.issubset(loaded_modules)
            or not any(name in {"mlx", "_mlx"} or name.startswith(("mlx.", "_mlx.")) for name in loaded_modules)
            or evidence.get("dependency_set_sha256") != canonical_hash(dependencies)
        ):
            return False, "attestation-dependency-digest-mismatch"

        evidence_output = _artifact_identity(evidence.get("output_artifact"))
        expected_quality_output = _controlled_quality_candidate_artifact(root, receipt)
        if (
            evidence_output is None
            or evidence_output["path"] != variant["quality_output_path"]
            or evidence_output["sha256"] != output_artifact["sha256"]
            or evidence_output["size_bytes"] != output_artifact["size_bytes"]
            or expected_quality_output is None
            or evidence_output != expected_quality_output
        ):
            return False, "attestation-output-digest-mismatch"

    # The retained challenge/evidence bundle above establishes internal
    # consistency and reproducibility-on-request only. Promotion additionally
    # requires a signature over EXTERNAL_ATTESTATION_SIGNED_FIELDS from a
    # protected Apple-Silicon signer, verified against a trust anchor that is
    # neither receipt-controlled nor checked into this repository. No such
    # external verifier exists today, so repository evidence remains sealed.
    return False, EXTERNAL_ATTESTATION_BLOCKER


def schema2_gate_state(receipt: dict[str, Any], root: Path) -> tuple[list[str], dict[str, bool], dict[str, Any]]:
    blockers: list[str] = []
    checks = {
        "aggregates_recomputed": False,
        "model_lineage_pinned": False,
        "target_hash_valid": False,
        "workload_hash_valid": False,
        "raw_outputs_valid": False,
        "quality_valid": False,
        "stability_passed": False,
        "stability_threshold_valid": False,
        "rollback_defined": False,
        "baseline_compatible": False,
        "enabled_methods_valid": False,
        "source_identity_pinned": False,
        "baseline_source_identity_match": False,
        "improvement_beyond_noise": False,
        "runner_valid": False,
        "quality_binding_valid": False,
        "experiment_descriptor_valid": False,
        "experiment_compatible": False,
        "execution_attested": False,
    }
    runs = receipt.get("runs")
    metrics = receipt_metrics(receipt)
    try:
        recomputed = recompute_aggregates(runs, metrics)
        checks["aggregates_recomputed"] = aggregates_match(receipt.get("aggregate"), recomputed)
    except SkillError:
        recomputed = {}
    if not checks["aggregates_recomputed"]:
        blockers.append("aggregate-mismatch" if isinstance(runs, list) and runs else "invalid-runs")

    command = receipt.get("command")
    runner = receipt.get("runner")
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        if (
            isinstance(runner, dict)
            and runner.get("id") in {EXTERNAL_RUNNER_ID, ATTESTED_RUNNER_ID}
        ):
            try:
                expected_runner = build_external_runner_descriptor(receipt, runner.get("argv_template"))
            except SkillError:
                expected_runner = None
            cwd = receipt.get("cwd")
            checks["runner_valid"] = (
                runner == expected_runner
                and command == resolve_external_argv_template(receipt, runner.get("argv_template"))
                and external_command_is_safe(command)
                and cwd == "."
            )
        else:
            expected_runner = build_runner_descriptor(command)
            checks["runner_valid"] = (
                runner == expected_runner
                and expected_runner["id"] == "mlx-lm-generate"
                and runner_semantics_valid(receipt)
                and isinstance(receipt.get("target", {}).get("descriptor", {}).get("software", {}).get("mlx_lm"), str)
                and bool(receipt["target"]["descriptor"]["software"]["mlx_lm"])
            )
    if not checks["runner_valid"]:
        blockers.append("uncontrolled-benchmark-runner")
    if isinstance(runner, dict) and runner.get("id") == ATTESTED_RUNNER_ID:
        checks["execution_attested"], attestation_blocker = attested_execution_state(
            receipt,
            root,
        )
        if attestation_blocker is not None:
            blockers.append(attestation_blocker)
    else:
        blockers.append("execution-semantics-unattested")

    expected_experiment = build_experiment_contract(receipt)
    checks["experiment_descriptor_valid"] = receipt.get("experiment") == expected_experiment
    if not checks["experiment_descriptor_valid"]:
        blockers.append("invalid-experiment-descriptor")

    models = receipt.get("models")
    model_items = list(models.values()) if isinstance(models, dict) else []
    checks["model_lineage_pinned"] = bool(model_items)
    for model in model_items:
        if not isinstance(model, dict) or not model.get("id") or not model.get("lineage_id"):
            checks["model_lineage_pinned"] = False
            break
        revision = model.get("revision")
        if not isinstance(revision, str) or PINNED_REVISION_RE.fullmatch(revision) is None:
            checks["model_lineage_pinned"] = False
            break
    if not checks["model_lineage_pinned"]:
        blockers.append("missing-model-lineage")
    checks["source_identity_pinned"] = bool(model_items)
    for model in model_items:
        source_id = model.get("source_id") if isinstance(model, dict) else None
        source_revision = model.get("source_revision") if isinstance(model, dict) else None
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(source_revision, str)
            or PINNED_REVISION_RE.fullmatch(source_revision) is None
        ):
            checks["source_identity_pinned"] = False
            break
    if not checks["source_identity_pinned"]:
        blockers.append("missing-source-lineage")

    target = receipt.get("target")
    if isinstance(target, dict) and isinstance(target.get("descriptor"), dict):
        descriptor = target["descriptor"]
        hardware = descriptor.get("hardware")
        software = descriptor.get("software")
        checks["target_hash_valid"] = (
            isinstance(hardware, dict)
            and bool(hardware.get("chip"))
            and isinstance(software, dict)
            and bool(software.get("mlx"))
            and target.get("sha256") == canonical_hash(descriptor)
        )
    if not checks["target_hash_valid"]:
        blockers.append("target-hash-mismatch" if isinstance(target, dict) and target.get("sha256") else "missing-target-hash")

    workload = receipt.get("workload")
    if isinstance(workload, dict):
        descriptor = {key: workload.get(key) for key in ("id", "artifacts", "parameters")}
        artifacts = workload.get("artifacts")
        artifacts_valid = isinstance(artifacts, list) and bool(artifacts)
        if artifacts_valid:
            for item in artifacts:
                valid, _ = check_artifact(root, item)
                artifacts_valid = artifacts_valid and valid
        checks["workload_hash_valid"] = (
            bool(workload.get("id"))
            and isinstance(workload.get("parameters"), dict)
            and bool(workload.get("parameters"))
            and artifacts_valid
            and workload.get("sha256") == canonical_hash(descriptor)
        )
    if not checks["workload_hash_valid"]:
        blockers.append("workload-hash-mismatch" if isinstance(workload, dict) and workload.get("sha256") else "missing-checked-in-input")

    raw_valid = isinstance(runs, list) and bool(runs)
    if raw_valid:
        for run in runs:
            raw_output = run.get("raw_output") if isinstance(run, dict) else None
            valid, _ = check_artifact(root, raw_output)
            raw_valid = raw_valid and valid and raw_output.get("truncated") is False and raw_metrics_match(root, receipt, run)
    checks["raw_outputs_valid"] = raw_valid
    if not raw_valid:
        blockers.append("raw-output-digest-mismatch" if any(
            isinstance(run, dict) and run.get("raw_output") for run in (runs if isinstance(runs, list) else [])
        ) else "missing-output-digests")

    quality = receipt.get("quality")
    quality_artifact = quality.get("artifact") if isinstance(quality, dict) else None
    quality_ok, quality_reason = check_artifact(root, quality_artifact)
    quality_payload: Any = None
    if quality_ok:
        quality_path = resolve_artifact(root, quality_artifact.get("path"))
        try:
            quality_payload = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path else None
        except (OSError, json.JSONDecodeError):
            quality_ok = False
            quality_reason = "digest"
    controlled_quality_valid = controlled_exact_output_quality_payload_valid(
        receipt,
        quality_payload,
        root,
    )
    checks["quality_valid"] = (
        quality_ok
        and quality.get("status") == "pass"
        and isinstance(quality_payload, dict)
        and quality_payload.get("schema_version") == 3
        and exact_output_validator_descriptor_valid(quality_payload.get("validator"))
        and quality_payload.get("status") == "pass"
        and isinstance(quality_payload.get("metric"), dict)
        and quality_payload["metric"].get("id") == "exact-output-parity"
        and quality_payload["metric"].get("exact_match") is True
        and controlled_quality_valid
    )
    if not checks["quality_valid"]:
        if quality_reason == "missing" or not isinstance(quality, dict):
            blockers.append("missing-quality-artifact")
        elif quality_reason == "digest":
            blockers.append("quality-artifact-digest-mismatch")
        else:
            blockers.append("quality-not-passing")
    checks["quality_binding_valid"] = (
        checks["quality_valid"]
        and quality_payload.get("bindings") == expected_quality_bindings(receipt)
        and controlled_quality_valid
    )
    if not checks["quality_binding_valid"]:
        blockers.append("quality-binding-invalid")

    stability = receipt.get("stability")
    primary_metric = stability.get("primary_metric") if isinstance(stability, dict) else DEFAULT_PRIMARY_METRIC
    max_cv = stability.get("max_cv") if isinstance(stability, dict) else DEFAULT_MAX_CV
    min_runs = stability.get("min_runs") if isinstance(stability, dict) else DEFAULT_MIN_RUNS
    default_primary_metric = "wall_seconds" if metrics == EXTERNAL_METRICS else DEFAULT_PRIMARY_METRIC
    if primary_metric not in metrics:
        primary_metric = default_primary_metric
    threshold_valid = (
        not isinstance(max_cv, bool)
        and isinstance(max_cv, (int, float))
        and 0 < float(max_cv) <= DEFAULT_MAX_CV
        and not isinstance(min_runs, bool)
        and isinstance(min_runs, int)
        and min_runs >= DEFAULT_MIN_RUNS
    )
    checks["stability_threshold_valid"] = threshold_valid
    if not threshold_valid:
        blockers.append("weak-stability-threshold")
    if isinstance(max_cv, bool) or not isinstance(max_cv, (int, float)) or max_cv <= 0 or max_cv > DEFAULT_MAX_CV:
        max_cv = DEFAULT_MAX_CV
    if isinstance(min_runs, bool) or not isinstance(min_runs, int) or min_runs < DEFAULT_MIN_RUNS:
        min_runs = DEFAULT_MIN_RUNS
    try:
        stability_result = stability_report(runs, primary_metric, float(max_cv), min_runs)
    except (SkillError, KeyError, TypeError):
        stability_result = {
            "primary_metric": primary_metric,
            "run_count": 0,
            "min_runs": min_runs,
            "max_cv": max_cv,
            "passes": False,
        }
    checks["stability_passed"] = bool(stability_result["passes"])
    if not checks["stability_passed"]:
        blockers.append("unstable-primary-metric")

    checks["rollback_defined"] = isinstance(receipt.get("rollback_condition"), str) and bool(receipt["rollback_condition"].strip())
    if not checks["rollback_defined"]:
        blockers.append("missing-rollback-condition")
    enabled_methods = receipt.get("enabled_methods")
    if receipt.get("comparison", {}).get("role") == "candidate":
        checks["enabled_methods_valid"] = (
            isinstance(enabled_methods, list)
            and bool(enabled_methods)
            and all(isinstance(method, str) and bool(method) for method in enabled_methods)
            and len(enabled_methods) == len(set(enabled_methods))
        )
        if not checks["enabled_methods_valid"]:
            blockers.append("missing-or-duplicate-enabled-methods")
    else:
        checks["enabled_methods_valid"] = True
    return blockers, checks, stability_result


def legacy_gate_state(receipt: dict[str, Any], receipts: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, bool], dict[str, Any]]:
    blockers = [
        "legacy-schema-1",
        "missing-model-lineage",
        "missing-output-digests",
        "missing-quality-artifact",
        "missing-rollback-condition",
        "missing-target-hash",
        "missing-workload-hash",
    ]
    runs = receipt.get("runs")
    try:
        recomputed = recompute_aggregates(runs)
        aggregate_ok = aggregates_match(receipt.get("aggregate"), recomputed)
    except SkillError:
        aggregate_ok = False
    if not aggregate_ok:
        blockers.append("aggregate-mismatch")
    strings = all_strings(receipt)
    if any("<historical-ephemeral" in value for value in strings):
        blockers.append("missing-checked-in-input")
    if any(EPHEMERAL_PATH_RE.search(value) for value in strings):
        blockers.append("nonportable-ephemeral-path")
    config = receipt.get("config_notes", {})
    versions = receipt.get("versions", {})
    if versions.get("mlx_lm") == "0.31.1" and config.get("draft_model"):
        blockers.append("mlx-lm-0.31.1-speculative-correctness-fix")
    label = str(receipt.get("label", ""))
    if label == "stack-measured-together":
        blockers.append("partial-invalid-stack-configuration")
    speedup = receipt.get("speedup_vs_baseline")
    baseline_label = speedup.get("baseline_label") if isinstance(speedup, dict) else None
    baseline_compatible = False
    if baseline_label:
        blockers.append("missing-baseline-digest")
        baseline = receipts.get(str(baseline_label))
        if baseline:
            candidate_model = config.get("model")
            baseline_model = baseline.get("config_notes", {}).get("model")
            prompt_counts = {run.get("prompt_tokens") for run in runs if isinstance(run, dict)}
            baseline_prompt_counts = {run.get("prompt_tokens") for run in baseline.get("runs", []) if isinstance(run, dict)}
            generation_counts = {run.get("generation_tokens") for run in runs if isinstance(run, dict)}
            baseline_generation_counts = {run.get("generation_tokens") for run in baseline.get("runs", []) if isinstance(run, dict)}
            baseline_compatible = (
                candidate_model == baseline_model
                and prompt_counts == baseline_prompt_counts
                and generation_counts == baseline_generation_counts
            )
            if label == "quant-4bit" and candidate_model != baseline_model:
                blockers.append("incompatible-quant-baseline")
            elif not baseline_compatible:
                blockers.append("baseline-workload-incompatible")
    try:
        stability_result = stability_report(runs, DEFAULT_PRIMARY_METRIC, DEFAULT_MAX_CV, DEFAULT_MIN_RUNS)
    except (KeyError, TypeError, ValueError):
        stability_result = {
            "primary_metric": DEFAULT_PRIMARY_METRIC,
            "run_count": 0,
            "min_runs": DEFAULT_MIN_RUNS,
            "max_cv": DEFAULT_MAX_CV,
            "passes": False,
        }
    if not stability_result["passes"]:
        blockers.append("unstable-primary-metric")
    checks = {
        "aggregates_recomputed": aggregate_ok,
        "model_lineage_pinned": False,
        "target_hash_valid": False,
        "workload_hash_valid": False,
        "raw_outputs_valid": False,
        "quality_valid": False,
        "stability_passed": bool(stability_result["passes"]),
        "stability_threshold_valid": False,
        "rollback_defined": False,
        "baseline_compatible": baseline_compatible,
        "enabled_methods_valid": False,
        "source_identity_pinned": False,
        "baseline_source_identity_match": False,
        "improvement_beyond_noise": False,
        "runner_valid": False,
        "quality_binding_valid": False,
        "experiment_descriptor_valid": False,
        "experiment_compatible": False,
        "execution_attested": False,
    }
    return blockers, checks, stability_result


def receipt_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.glob("*.json") if path.name not in RESERVED_JSON),
        key=lambda path: path.name,
    )


def load_receipts(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    receipts: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for path in receipt_files(root):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillError(f"could not parse benchmark receipt {path}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("label"), str):
            raise SkillError(f"benchmark receipt {path} must be an object with a label")
        label = value["label"]
        if label in receipts:
            raise SkillError(f"duplicate benchmark receipt label: {label}")
        receipts[label] = value
        paths[label] = path
    return receipts, paths


def assess_schema2_baseline(
    receipt: dict[str, Any],
    root: Path,
    receipts: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    blockers: list[str],
    checks: dict[str, bool],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    comparison = receipt.get("comparison")
    role = comparison.get("role") if isinstance(comparison, dict) else None
    if role != "candidate":
        blockers.append("baseline-role-not-promotable" if role == "baseline" else "missing-baseline-comparison")
        return None, None
    baseline_ref = comparison.get("baseline_receipt")
    baseline_path = resolve_artifact(root, baseline_ref)
    baseline_label = Path(str(baseline_ref)).stem if baseline_ref else None
    registered_baseline_path = paths.get(str(baseline_label)) if baseline_label else None
    expected_digest = comparison.get("baseline_sha256")
    if (
        baseline_path is None
        or not baseline_path.is_file()
        or baseline_label not in receipts
        or registered_baseline_path is None
        or baseline_path != registered_baseline_path.resolve()
        or baseline_ref != registered_baseline_path.name
    ):
        blockers.append("missing-baseline-receipt")
        return {"label": baseline_label, "compatible": False}, None
    actual_digest = file_sha256(baseline_path)
    if expected_digest != actual_digest:
        blockers.append("baseline-digest-mismatch")
    baseline = receipts[baseline_label]
    baseline_blockers, _, _ = schema2_gate_state(baseline, root) if baseline.get("schema_version") == 2 else (["legacy-schema-1"], {}, {})
    baseline_compatibility_blockers = [
        blocker for blocker in baseline_blockers
        if blocker not in ATTESTATION_PROMOTION_BLOCKERS
    ]
    if baseline_compatibility_blockers:
        blockers.append("baseline-not-validation-complete")
    target = receipt.get("target", {})
    baseline_target = baseline.get("target", {})
    workload = receipt.get("workload", {})
    baseline_workload = baseline.get("workload", {})
    model = receipt.get("models", {}).get("target", {})
    baseline_model = baseline.get("models", {}).get("target", {})
    primary_metric = comparison.get("primary_metric")
    baseline_metric = baseline.get("comparison", {}).get("primary_metric")
    source_identity_match = (
        bool(model.get("source_id"))
        and model.get("source_id") == baseline_model.get("source_id")
        and model.get("source_revision") == baseline_model.get("source_revision")
    )
    checks["baseline_source_identity_match"] = source_identity_match
    if not source_identity_match:
        blockers.append("baseline-source-lineage-incompatible")
    experiment = receipt.get("experiment", {})
    baseline_experiment = baseline.get("experiment", {})
    experiment_compatible = (
        isinstance(experiment, dict)
        and isinstance(baseline_experiment, dict)
        and isinstance(experiment.get("invariant_sha256"), str)
        and experiment.get("invariant_sha256") == baseline_experiment.get("invariant_sha256")
        and experiment.get("invariant") == baseline_experiment.get("invariant")
    )
    checks["experiment_compatible"] = experiment_compatible
    if not experiment_compatible:
        blockers.append("experiment-invariant-incompatible")
    candidate_variant_arguments = (
        experiment.get("variant", {}).get("invocation_arguments")
        if isinstance(experiment, dict)
        else None
    )
    baseline_variant_arguments = (
        baseline_experiment.get("variant", {}).get("invocation_arguments")
        if isinstance(baseline_experiment, dict)
        else None
    )
    external_pair = receipt_metrics(receipt) == EXTERNAL_METRICS
    execution_variant_distinct = (
        candidate_variant_arguments != baseline_variant_arguments
        if external_pair
        else receipt.get("models") != baseline.get("models")
        or candidate_variant_arguments != baseline_variant_arguments
    )
    if not execution_variant_distinct:
        checks["enabled_methods_valid"] = False
        blockers.append("candidate-invocation-not-distinct")
    workload_compatible = (
        target.get("sha256") == baseline_target.get("sha256")
        and workload.get("sha256") == baseline_workload.get("sha256")
        and model.get("lineage_id") == baseline_model.get("lineage_id")
        and receipt_metrics(baseline) == receipt_metrics(receipt)
        and primary_metric in receipt_metrics(receipt)
        and primary_metric == baseline_metric
    )
    if not workload_compatible:
        blockers.append("baseline-workload-incompatible")
    compatible = (
        expected_digest == actual_digest
        and not baseline_compatibility_blockers
        and source_identity_match
        and experiment_compatible
        and workload_compatible
    )
    checks["baseline_compatible"] = compatible
    ratios = recomputed_median_ratios(receipt, baseline)
    ratio_key = {
        "generation_tps": "decode_tps",
        "prompt_tps": "prefill_tps",
        "ttft_proxy_s": "ttft_proxy_inverse",
        "peak_memory_gb": "peak_memory_inverse",
        "wall_seconds": "wall_seconds_inverse",
    }.get(str(primary_metric))
    improvement: dict[str, Any] | None = None
    if ratio_key and ratios and isinstance(ratios.get(ratio_key), (int, float)):
        try:
            candidate_stability = stability_report(
                receipt["runs"], str(primary_metric), DEFAULT_MAX_CV, DEFAULT_MIN_RUNS
            )
            baseline_stability = stability_report(
                baseline["runs"], str(primary_metric), DEFAULT_MAX_CV, DEFAULT_MIN_RUNS
            )
            candidate_cv = float(candidate_stability["cv"])
            baseline_cv = float(baseline_stability["cv"])
            noise_margin = max(0.02, 2 * max(candidate_cv, baseline_cv))
            required_ratio = 1.0 + noise_margin
            observed_ratio = float(ratios[ratio_key])
            improvement_passes = observed_ratio > required_ratio
            checks["improvement_beyond_noise"] = improvement_passes
            improvement = {
                "primary_metric": primary_metric,
                "ratio_key": ratio_key,
                "observed_ratio": observed_ratio,
                "candidate_cv": candidate_cv,
                "baseline_cv": baseline_cv,
                "noise_margin": noise_margin,
                "required_ratio": required_ratio,
                "passes": improvement_passes,
            }
        except (KeyError, SkillError, TypeError, ValueError):
            improvement = None
    if not checks["improvement_beyond_noise"]:
        blockers.append("no-improvement-beyond-noise")
    baseline_result = {
        "label": baseline_label,
        "file": baseline_path.name,
        "expected_sha256": expected_digest,
        "actual_sha256": actual_digest,
        "compatible": compatible,
        "experiment_invariant_sha256": experiment.get("invariant_sha256") if isinstance(experiment, dict) else None,
    }
    return baseline_result, improvement


def build_assessment_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    receipts, paths = load_receipts(root)
    rows: list[dict[str, Any]] = []
    for label in sorted(receipts):
        receipt = receipts[label]
        schema_version = receipt.get("schema_version")
        strings = all_strings(receipt)
        security_blockers: list[str] = []
        command_text = shlex.join([str(item) for item in receipt.get("command", [])]) if isinstance(receipt.get("command"), list) else ""
        if SECRET_COMMAND_RE.search(command_text):
            security_blockers.append("secret-bearing-command")
        if contains_serialized_secret(receipt):
            security_blockers.append("serialized-secret")
        if any(EPHEMERAL_PATH_RE.search(value) for value in strings):
            security_blockers.append("nonportable-ephemeral-path")
        if schema_version == 2:
            blockers, checks, stability = schema2_gate_state(receipt, root)
            baseline, improvement = assess_schema2_baseline(receipt, root, receipts, paths, blockers, checks)
        else:
            blockers, checks, stability = legacy_gate_state(receipt, receipts)
            improvement = None
            speedup = receipt.get("speedup_vs_baseline")
            baseline = {
                "label": speedup.get("baseline_label"),
                "compatible": checks["baseline_compatible"],
            } if isinstance(speedup, dict) else None
        baseline_label = baseline.get("label") if isinstance(baseline, dict) else None
        ratios = recomputed_median_ratios(receipt, receipts[baseline_label]) if baseline_label in receipts else None
        blockers.extend(security_blockers)
        blockers = sorted(set(blockers))
        promotion_ready = schema_version == 2 and receipt.get("comparison", {}).get("role") == "candidate" and not blockers
        receipt_sha256 = file_sha256(paths[label])
        experiment_fingerprint = (
            build_experiment_fingerprint(receipt, receipt_sha256=receipt_sha256, root=root)
            if promotion_ready
            else None
        )
        non_attestation_blockers = [
            blocker for blocker in blockers
            if blocker not in ATTESTATION_PROMOTION_BLOCKERS
        ]
        noise_only_rejection = (
            schema_version == 2
            and receipt.get("comparison", {}).get("role") == "candidate"
            and non_attestation_blockers == ["no-improvement-beyond-noise"]
        )
        classification = (
            "rejected"
            if label == "stack-measured-together" or noise_only_rejection
            else "promotion_ready" if promotion_ready else "performance_observation"
        )
        environment = receipt.get("environment", {})
        enabled_methods = (
            receipt.get("enabled_methods", [])
            if schema_version == 2
            else HISTORICAL_ENABLED_METHODS.get(label, receipt.get("enabled_methods", []))
        )
        rows.append({
            "label": label,
            "receipt": paths[label].name,
            "receipt_sha256": receipt_sha256,
            "schema_version": schema_version,
            "timestamp": receipt.get("timestamp"),
            "classification": classification,
            "promotion_ready": promotion_ready,
            "experiment_fingerprint": experiment_fingerprint,
            "enabled_methods": enabled_methods,
            "reasons": blockers,
            "gates": checks,
            "stability": stability,
            "baseline": baseline,
            "recomputed_median_ratios": ratios,
            "improvement": improvement,
            "primary_metric": receipt.get("comparison", {}).get("primary_metric", DEFAULT_PRIMARY_METRIC),
            "chip": environment.get("cpu_brand") or environment.get("mac_model") or environment.get("machine"),
        })
    integrity_errors = [
        f"unreferenced-attestation-file:{path}"
        for path in unreferenced_attestation_files(root, receipts)
    ]
    return {
        "schema_version": 1,
        "integrity_errors": integrity_errors,
        "assessments": rows,
    }


def assessment_summary(report: dict[str, Any]) -> dict[str, int]:
    rows = report.get("assessments", [])
    return {
        "receipt_count": len(rows),
        "performance_observation_count": sum(row.get("classification") == "performance_observation" for row in rows),
        "promotion_ready_count": sum(row.get("classification") == "promotion_ready" for row in rows),
        "rejected_count": sum(row.get("classification") == "rejected" for row in rows),
        "integrity_error_count": len(report.get("integrity_errors", []))
        + sum(
            reason in INTEGRITY_BLOCKERS
            for row in rows
            for reason in row.get("reasons", [])
        ),
    }


def build_receipts_index(report: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = (root or DEFAULT_ROOT).resolve()
    receipts, _ = load_receipts(root)
    rows = []
    for assessment in report.get("assessments", []):
        receipt = receipts[assessment["label"]]
        rows.append({
            "label": assessment["label"],
            "file": assessment["receipt"],
            "date": assessment.get("timestamp"),
            "chip": assessment.get("chip"),
            "command_summary": shlex.join([str(item) for item in receipt.get("command", [])]),
            "config_notes": redact_secrets(receipt.get("config_notes", {})),
            "schema_version": assessment.get("schema_version"),
            "receipt_sha256": assessment.get("receipt_sha256"),
            "classification": assessment.get("classification"),
            "promotion_ready": assessment.get("promotion_ready"),
            "reasons": assessment.get("reasons", []),
        })
    rows.sort(key=lambda row: row["label"])
    return {"schema_version": 2, "receipts": rows}


def render_benchmark_report(report: dict[str, Any]) -> str:
    summary = assessment_summary(report)
    has_external = any(
        row.get("primary_metric") == "wall_seconds"
        for row in report.get("assessments", [])
    )
    lines = [
        "# Benchmark evidence assessment",
        "",
        "This report is generated deterministically from `assets/benchmarks/receipt_assessments.json`.",
        "Historical schema-1 measurements are retained as observations; only schema-2 candidate receipts that pass every gate are promotion-ready.",
        "MLX-LM 0.31.1 speculative-decoding observations are held because v0.31.2 fixed silent output corruption.",
        "",
        "## Summary",
        "",
        f"- Receipts: {summary['receipt_count']}",
        f"- Performance observations: {summary['performance_observation_count']}",
        f"- Promotion-ready: {summary['promotion_ready_count']}",
        f"- Rejected: {summary['rejected_count']}",
        f"- Integrity errors: {summary['integrity_error_count']}",
        "",
        "## Assessments",
        "",
        "| Receipt | Classification | Enabled methods | CV | "
        + ("Recomputed primary ratio" if has_external else "Recomputed decode ratio")
        + " | Reasons |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in report.get("assessments", []):
        methods = ", ".join(row.get("enabled_methods", [])) or "none"
        cv = row.get("stability", {}).get("cv")
        cv_text = f"{float(cv):.3f}" if isinstance(cv, (int, float)) else "n/a"
        ratio_key = "wall_seconds_inverse" if row.get("primary_metric") == "wall_seconds" else "decode_tps"
        ratio = (row.get("recomputed_median_ratios") or {}).get(ratio_key)
        ratio_text = f"{float(ratio):.4f}x" if isinstance(ratio, (int, float)) else "n/a"
        if row.get("classification") == "promotion_ready" and ratio_text != "n/a":
            ratio_text += " receipt"
        reasons = "<br>".join(row.get("reasons", [])) or "none"
        lines.append(
            f"| `{row['receipt']}` | `{row['classification']}` | {methods} | {cv_text} | {ratio_text} | {reasons} |"
        )
    lines.extend([
        "",
        "## Promotion rule",
        "",
        "A candidate is `promotion_ready` only when aggregate recomputation, pinned target/source lineage, the canonical experiment invariant, normalized target/workload hashes, bounded raw evidence, controlled quality, stability, rollback, baseline compatibility, and externally signed execution attestation all pass. The retained Qwen challenge/evidence lane establishes internal consistency and reproducibility-on-request, but SHA-256 digests are not signatures. Promotion requires a protected Apple-Silicon signer and an out-of-repository trust anchor covering the repository commit/tree, challenge, reviewed dependency manifest, raw output, promotion policy, and timing. No such signer exists today, so every checked-in receipt remains sealed. The primary-metric ratio must also exceed `1 + max(2%, 2 x max(candidate CV, baseline CV))`. Missing evidence is never inferred.",
        "",
    ])
    return "\n".join(lines)


def benchmark_report_path(root: Path) -> Path:
    return root.parent / REPORT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and assess benchmark evidence")
    parser.add_argument("action", choices=("generate", "check"))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = Path(args.root).resolve()
        report = build_assessment_report(root)
        index = build_receipts_index(report, root)
        assessment_path = root / ASSESSMENT_NAME
        index_path = root / INDEX_NAME
        human_report_path = benchmark_report_path(root)
        human_report = render_benchmark_report(report)
        if args.action == "generate":
            dump_json(report, assessment_path)
            dump_json(index, index_path)
            atomic_write_text(human_report_path, human_report)
            print(f"wrote {len(report['assessments'])} benchmark assessments")
            return 1 if assessment_summary(report)["integrity_error_count"] else 0
        drift = []
        for path, expected in ((assessment_path, report), (index_path, index)):
            if not path.is_file():
                drift.append(f"missing {path.name}")
                continue
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != expected:
                drift.append(f"stale {path.name}")
        if not human_report_path.is_file():
            drift.append(f"missing {human_report_path.name}")
        elif human_report_path.read_text(encoding="utf-8") != human_report:
            drift.append(f"stale {human_report_path.name}")
        payload = {"ok": not drift, "drift": drift, "summary": assessment_summary(report)}
        print(json.dumps(payload, indent=2))
        return 1 if drift or payload["summary"]["integrity_error_count"] else 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
