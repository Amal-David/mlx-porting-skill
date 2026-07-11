#!/usr/bin/env python3
"""Generate a parity-first MLX package from a trusted static inspection.

This generator is dependency-free. It validates JSON and writes Python source;
MLX and NumPy are imported only by the generated package at execution time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from _common import SkillError, load_structured
from make_port_plan import verify_inspection_against_artifact
from recommend_optimizations import (
    resolve_route_families,
    trusted_inspection_sha256,
    validate_trusted_inspection,
)


GENERATOR_VERSION = "1.1.0"
SCAFFOLD_MANIFEST = "scaffold-manifest.json"
DENSE_FAMILY = "dense-decoder-transformer"
DENSE_RUNBOOK = "references/runbook-decoder-transformer.md"
MOE_FAMILY = "moe-decoder-transformer"
MOE_RUNBOOK = "references/runbook-moe-transformer.md"

MOE_ROUTER_PROFILES: dict[str, dict[str, Any]] = {
    "mixtral": {
        "implemented": True,
        "router": "softmax-top-k",
        "renormalize": "always",
        "topology": "mixtral-block-sparse",
    },
    "qwen2_moe": {
        "implemented": True,
        "router": "softmax-top-k",
        "renormalize": "config",
        "topology": "qwen2-per-expert",
    },
    "granitemoe": {
        "implemented": True,
        "router": "softmax-top-k",
        "renormalize": "always",
        "topology": "granite-packed-experts",
    },
    "phimoe": {
        "implemented": False,
        "router": "sparse-mixer",
        "reason": "SparseMixer routing is not implemented",
    },
    "qwen3_moe": {
        "implemented": False,
        "router": "softmax-top-k",
        "reason": "Q/K normalization and shared-expert topology are not implemented",
    },
    "olmoe": {
        "implemented": False,
        "router": "softmax-top-k",
        "reason": "Q/K normalization topology is not implemented",
    },
    "deepseek_v2": {
        "implemented": False,
        "router": "grouped-top-k",
        "reason": "MLA, shared experts, and grouped routing are not implemented",
    },
    "deepseek_v3": {
        "implemented": False,
        "router": "grouped-top-k-with-bias",
        "reason": "MLA, shared experts, grouped routing, and router bias are not implemented",
    },
    "dbrx": {"implemented": False, "router": "unknown", "reason": "router semantics are not implemented"},
    "grok": {"implemented": False, "router": "unknown", "reason": "router semantics are not implemented"},
    "minimax": {"implemented": False, "router": "unknown", "reason": "router semantics are not implemented"},
    "ernie4_5_moe": {"implemented": False, "router": "unknown", "reason": "router semantics are not implemented"},
}

ENCODER_FAMILY = "encoder-transformer"
ENCODER_RUNBOOK = "references/runbook-encoder-transformer.md"

SSM_FAMILY = "ssm-recurrent-hybrid"
SSM_RUNBOOK = "references/runbook-ssm-hybrid.md"

# Computation-bearing keys whose semantics this generator implements. Keys that
# are not here must be known non-computational metadata or they fail closed.
DENSE_CONFIG_FEATURE_ALLOWLIST = frozenset({
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "activation_function",
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "max_window_layers",
    "max_position_embeddings",
    "mlp_bias",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pretraining_tp",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "rope_traditional",
    "sliding_window",
    "tie_word_embeddings",
    "use_sliding_window",
    "vocab_size",
})
FEATURE_ALLOWLIST = DENSE_CONFIG_FEATURE_ALLOWLIST

MOE_CONFIG_FEATURE_ALLOWLIST = DENSE_CONFIG_FEATURE_ALLOWLIST | frozenset({
    "decoder_sparse_step",
    "layer_types",
    "mlp_only_layers",
    "moe_intermediate_size",
    "norm_topk_prob",
    "num_experts",
    "num_experts_per_tok",
    "num_local_experts",
    "output_router_logits",
    "qkv_bias",
    "rope_parameters",
    "router_aux_loss_coef",
    "router_bias",
    "router_jitter_noise",
    "top_k",
})
SSM_CONFIG_FEATURE_ALLOWLIST = frozenset({
    "conv_bias",
    "d_conv",
    "d_model",
    "d_state",
    "expand",
    "n_layer",
    "rms_norm_eps",
    "ssm_variant",
    "tie_word_embeddings",
    "vocab_size",
})

MIXED_SSM_MODEL_TYPES = frozenset({
    "griffin",
    "jamba",
    "recurrent_gemma",
    "recurrentgemma",
    "zamba",
    "zamba2",
})

PURE_SSM_ARCHITECTURES = {
    "mamba": "MambaForCausalLM",
    "mamba2": "Mamba2ForCausalLM",
}

SSM_FORBIDDEN_TENSOR_SEGMENTS = frozenset({
    "experts",
    "k_proj",
    "q_proj",
    "router",
    "self_attn",
})

ENCODER_CONFIG_FEATURE_ALLOWLIST = frozenset({
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "layer_norm_eps",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "pad_token_id",
    "position_embedding_type",
    "type_vocab_size",
    "vocab_size",
})

ENCODER_INFERENCE_METADATA_KEYS = frozenset({
    "add_cross_attention",
    "attention_probs_dropout_prob",
    "classifier_dropout",
    "gradient_checkpointing",
    "hidden_dropout_prob",
})

SUPPORTED_BERT_ARCHITECTURES = frozenset({
    "BertForMaskedLM",
    "BertForMultipleChoice",
    "BertForNextSentencePrediction",
    "BertForPreTraining",
    "BertForQuestionAnswering",
    "BertForSequenceClassification",
    "BertForTokenClassification",
    "BertLMHeadModel",
    "BertModel",
})

FAMILY_FEATURE_ALLOWLISTS = {
    DENSE_FAMILY: DENSE_CONFIG_FEATURE_ALLOWLIST,
    ENCODER_FAMILY: ENCODER_CONFIG_FEATURE_ALLOWLIST,
    MOE_FAMILY: MOE_CONFIG_FEATURE_ALLOWLIST,
    SSM_FAMILY: SSM_CONFIG_FEATURE_ALLOWLIST,
}

KNOWN_METADATA_KEYS = frozenset({
    "_commit_hash",
    "_name_or_path",
    "architectures",
    "auto_map",
    "bad_words_ids",
    "begin_suppress_tokens",
    "bos_token_id",
    "decoder_start_token_id",
    "dtype",
    "early_stopping",
    "eos_token_id",
    "exponential_decay_length_penalty",
    "finetuning_task",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "id2label",
    "initializer_range",
    "is_decoder",
    "is_encoder_decoder",
    "label2id",
    "length_penalty",
    "license",
    "max_length",
    "min_length",
    "model_type",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "num_return_sequences",
    "output_attentions",
    "output_hidden_states",
    "output_scores",
    "pad_token_id",
    "prefix",
    "problem_type",
    "pruned_heads",
    "remove_invalid_values",
    "repetition_penalty",
    "return_dict",
    "return_dict_in_generate",
    "sep_token_id",
    "suppress_tokens",
    "task_specific_params",
    "temperature",
    "tf_legacy_loss",
    "tie_encoder_decoder",
    "tokenizer_class",
    "top_k",
    "top_p",
    "torch_dtype",
    "transformers_version",
    "typical_p",
    "use_bfloat16",
    "use_cache",
})

MOE_KEYS = frozenset({
    "decoder_sparse_step",
    "expert_capacity",
    "expert_interval",
    "moe_intermediate_size",
    "moe_layer_freq",
    "moe_shared_expert_intermediate_size",
    "norm_topk_prob",
    "num_experts",
    "num_experts_per_tok",
    "num_local_experts",
    "num_shared_experts",
    "router_aux_loss_coef",
    "router_jitter_noise",
    "shared_expert_intermediate_size",
})
MOE_UNSUPPORTED_KEYS = frozenset({
    "add_router_probs",
    "aux_loss_at_inference",
    "capacity_factor",
    "drop_tokens",
    "ep_size",
    "expert_capacity",
    "expert_parallel_size",
    "expert_parallelism",
    "n_group",
    "num_expert_groups",
    "routed_scaling_factor",
    "scoring_func",
    "token_drop",
    "topk_group",
    "topk_method",
})
QUANTIZATION_KEYS = frozenset({
    "bits",
    "group_size",
    "quantization",
    "quantization_config",
    "quant_method",
})
ATTENTION_VARIANT_KEYS = frozenset({
    "alibi",
    "attention_chunk_size",
    "attention_sink_size",
    "attention_type",
    "attn_logit_softcapping",
    "logit_softcap",
    "partial_rotary_factor",
    "position_embedding_type",
    "qk_norm",
    "query_pre_attn_scalar",
    "sliding_window",
    "use_alibi",
    "use_qk_norm",
    "use_sliding_window",
})
SUPPORTED_ACTIVATIONS = frozenset({"gelu", "relu", "silu", "swish"})
SUPPORTED_ROPE_TYPES = frozenset({"default", "dynamic", "linear"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a runnable parity-first MLX dense-decoder package",
    )
    parser.add_argument("inspection", help="Trusted inspection.json produced by inspect_model.py")
    parser.add_argument(
        "--artifact-root",
        help="Local inspected artifact; required for byte-bound re-verification",
    )
    parser.add_argument("--output", required=True, help="New directory for generated Python files")
    return parser.parse_args(argv)


def _meaningfully_set(value: Any) -> bool:
    return value not in (None, False, 0, 0.0, "", [], {})


def _config_int(config: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise SkillError(f"config.json {key} must be a positive integer")
    return value


def _config_float(config: dict[str, Any], key: str, *, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SkillError(f"config.json {key} must be a positive number")
    return float(value)


def _rope_type(rope_scaling: dict[str, Any]) -> str:
    first = rope_scaling.get("rope_type")
    second = rope_scaling.get("type")
    if first is not None and second is not None and first != second:
        return f"conflicting:{first!r}:{second!r}"
    value = first if first is not None else second
    return str(value or "default").lower()


def unsupported_dense_features(config: dict[str, Any]) -> list[str]:
    """Return every computation feature that this generator cannot implement."""
    errors: list[str] = []
    activation = config.get("hidden_act", config.get("activation_function", "silu"))
    if not isinstance(activation, str) or activation.lower() not in SUPPORTED_ACTIVATIONS:
        errors.append(
            f"hidden_act={activation!r} is not supported; supported activations: "
            + ", ".join(sorted(SUPPORTED_ACTIVATIONS))
        )
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0.0:
        errors.append("attention_dropout must be 0.0 for eager inference")
    if config.get("pretraining_tp", 1) != 1:
        errors.append("pretraining_tp must be 1")

    for key in sorted(MOE_KEYS):
        if key in config and _meaningfully_set(config[key]):
            errors.append(f"MoE config key {key!r} is set")

    for key in sorted(ATTENTION_VARIANT_KEYS - {"sliding_window", "use_sliding_window"}):
        if key not in config or not _meaningfully_set(config[key]):
            continue
        value = config[key]
        if key == "partial_rotary_factor" and value == 1.0:
            continue
        if key == "position_embedding_type" and str(value).lower() in {"rope", "rotary"}:
            continue
        errors.append(f"{key}={value!r} is not supported")

    for key in sorted(QUANTIZATION_KEYS):
        is_set = key in config and (
            config[key] is not None if key == "quantization_config" else _meaningfully_set(config[key])
        )
        if is_set:
            label = "quantization_config is set" if key == "quantization_config" else f"{key}={config[key]!r} is not supported"
            errors.append(label)

    rope_scaling = config.get("rope_scaling")
    if rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            errors.append("rope_scaling must be null or an object")
        else:
            rope_type = _rope_type(rope_scaling)
            if rope_type not in SUPPORTED_ROPE_TYPES:
                errors.append(
                    f"rope_scaling type {rope_type!r} is not supported; supported types: "
                    + ", ".join(sorted(SUPPORTED_ROPE_TYPES))
                )
            else:
                allowed_rope_keys = {"rope_type", "type"}
                if rope_type in {"dynamic", "linear"}:
                    allowed_rope_keys.update({"factor", "original_max_position_embeddings"})
                for key in sorted(set(rope_scaling) - allowed_rope_keys):
                    errors.append(f"rope_scaling key {key!r} is not supported for type {rope_type!r}")
                if rope_type in {"dynamic", "linear"}:
                    factor = rope_scaling.get("factor")
                    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 1.0:
                        errors.append(f"rope_scaling factor must be a number greater than 1 for type {rope_type!r}")

    use_sliding_window = config.get("use_sliding_window")
    if "use_sliding_window" in config and not isinstance(use_sliding_window, bool):
        errors.append("use_sliding_window must be boolean when present")
    sliding_window = config.get("sliding_window")
    if "sliding_window" in config and (
        type(sliding_window) is not int or sliding_window <= 0
    ):
        errors.append("sliding_window must be a positive integer when present")
    max_window_layers = config.get("max_window_layers")
    if "max_window_layers" in config and (
        type(max_window_layers) is not int or max_window_layers <= 0
    ):
        errors.append("max_window_layers must be a positive integer when present")
    if use_sliding_window is True:
        errors.append("use_sliding_window=True is not supported")
    elif _meaningfully_set(sliding_window) and use_sliding_window is not False:
        errors.append(
            f"sliding_window={sliding_window!r} requires explicit "
            "use_sliding_window=false for full attention"
        )

    classified = (
        DENSE_CONFIG_FEATURE_ALLOWLIST
        | KNOWN_METADATA_KEYS
        | MOE_KEYS
        | QUANTIZATION_KEYS
        | ATTENTION_VARIANT_KEYS
    )
    for key in sorted(set(config) - classified):
        errors.append(f"unrecognized computation-relevant config key {key!r}")
    return errors


def validate_dense_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SkillError("config.json must contain an object")
    errors = unsupported_dense_features(config)
    if errors:
        raise SkillError(
            "Unsupported config features; no code was generated. "
            f"Consult {DENSE_RUNBOOK}:\n- " + "\n- ".join(errors)
        )

    hidden_size = _config_int(config, "hidden_size")
    num_heads = _config_int(config, "num_attention_heads")
    num_kv_heads = _config_int(config, "num_key_value_heads", default=num_heads)
    head_dim = _config_int(config, "head_dim", default=hidden_size // num_heads)
    _config_int(config, "num_hidden_layers")
    _config_int(config, "intermediate_size")
    _config_int(config, "vocab_size")
    _config_int(config, "max_position_embeddings", default=2048)
    _config_float(config, "rms_norm_eps", default=1e-5)
    _config_float(config, "rope_theta", default=10000.0)
    if hidden_size != num_heads * head_dim:
        raise SkillError("config.json hidden_size must equal num_attention_heads * head_dim")
    if num_heads % num_kv_heads != 0:
        raise SkillError("config.json num_attention_heads must be divisible by num_key_value_heads")
    for key in ("attention_bias", "mlp_bias", "rope_traditional", "tie_word_embeddings"):
        if key in config and not isinstance(config[key], bool):
            raise SkillError(f"config.json {key} must be boolean")
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0.0:
        raise SkillError("Unsupported config features; no code was generated. "
                         f"Consult {DENSE_RUNBOOK}:\n- attention_dropout must be 0.0 for eager inference")
    if config.get("pretraining_tp", 1) != 1:
        raise SkillError("Unsupported config features; no code was generated. "
                         f"Consult {DENSE_RUNBOOK}:\n- pretraining_tp must be 1")
    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict) and _rope_type(rope_scaling) == "dynamic" and head_dim <= 2:
        raise SkillError("config.json dynamic RoPE requires head_dim greater than 2")
    return config


def unsupported_ssm_features(config: dict[str, Any]) -> list[str]:
    """Return config features outside the synthetic minimal selective-SSM contract."""
    errors: list[str] = []
    model_type = str(config.get("model_type", "")).lower()
    architectures = config.get("architectures")
    architecture_names = (
        [value.lower() for value in architectures]
        if isinstance(architectures, list)
        and all(isinstance(value, str) for value in architectures)
        else []
    )
    mixed_identity = model_type in MIXED_SSM_MODEL_TYPES or any(
        any(alias in name for alias in MIXED_SSM_MODEL_TYPES)
        for name in architecture_names
    )
    mixed_keys = sorted(
        key
        for key in ("attention_every_n_layers", "attention_layer_indices", "layer_types")
        if key in config and _meaningfully_set(config[key])
    )
    if mixed_identity or mixed_keys:
        details = []
        if mixed_identity:
            details.append(f"model identity {model_type or architecture_names!r}")
        if mixed_keys:
            details.append("config keys " + ", ".join(repr(key) for key in mixed_keys))
        errors.append(
            "hybrid/attention-mixed SSM config is not supported by the pure-SSM generator ("
            + "; ".join(details)
            + ")"
        )
    if config.get("ssm_variant") != "minimal_selective":
        errors.append("ssm_variant must be exactly 'minimal_selective'")
    if model_type not in {"mamba", "mamba2"}:
        errors.append("model_type must be 'mamba' or 'mamba2' for this minimal selective-SSM path")
    if not isinstance(architectures, list) or not all(
        isinstance(value, str) for value in architectures
    ):
        errors.append("architectures must be a list of strings for the pure-SSM generator")
    elif model_type in PURE_SSM_ARCHITECTURES:
        expected_architectures = [PURE_SSM_ARCHITECTURES[model_type]]
        if architectures != expected_architectures:
            errors.append(
                f"architectures must be exactly {expected_architectures!r} "
                f"for model_type {model_type!r}"
            )
    if "is_decoder" in config and config["is_decoder"] is not True:
        errors.append("is_decoder must be true when set for the decoder-only SSM generator")
    if "is_encoder_decoder" in config and config["is_encoder_decoder"] is not False:
        errors.append(
            "is_encoder_decoder must be false when set for the decoder-only SSM generator"
        )
    for key in ("conv_bias", "tie_word_embeddings"):
        if key in config and not isinstance(config[key], bool):
            errors.append(f"{key} must be boolean")
    classified = SSM_CONFIG_FEATURE_ALLOWLIST | KNOWN_METADATA_KEYS
    for key in sorted(set(config) - classified):
        if key not in mixed_keys:
            errors.append(f"unrecognized computation-relevant config key {key!r}")
    return errors


def validate_ssm_inspection(inspection: dict[str, Any]) -> None:
    """Reject attention/MoE tensor namespaces before generating a pure SSM graph."""
    forbidden: list[str] = []
    for tensor in inspection.get("tensors", []):
        if not isinstance(tensor, dict) or not isinstance(tensor.get("key"), str):
            continue
        key = tensor["key"]
        segments = {segment.lower() for segment in key.split(".")}
        if segments & SSM_FORBIDDEN_TENSOR_SEGMENTS:
            forbidden.append(key)
    if forbidden:
        raise SkillError(
            "Unsupported SSM tensor inventory; no code was generated. "
            "Pure-SSM generation rejects attention or MoE tensor namespaces: "
            + ", ".join(sorted(forbidden)[:5])
        )


def validate_ssm_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SkillError("config.json must contain an object")
    errors = unsupported_ssm_features(config)
    if errors:
        raise SkillError(
            "Unsupported SSM config features; no code was generated. "
            f"Consult {SSM_RUNBOOK}:\n- " + "\n- ".join(errors)
        )
    for key in ("d_model", "d_state", "d_conv", "expand", "n_layer", "vocab_size"):
        _config_int(config, key)
    _config_float(config, "rms_norm_eps", default=1e-5)
    return config


def _aliased_positive_int(
    config: dict[str, Any],
    primary: str,
    alias: str,
) -> int:
    first = config.get(primary)
    second = config.get(alias)
    if first is not None and second is not None and first != second:
        raise SkillError(f"config.json {primary} and {alias} must agree when both are present")
    value = first if first is not None else second
    if type(value) is not int or value <= 0:
        raise SkillError(f"config.json requires positive integer {primary} or {alias}")
    return value


def moe_router_profile(config: dict[str, Any]) -> dict[str, Any]:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise SkillError("config.json model_type must identify an explicit MoE router profile")
    alias = model_type.lower()
    profile = MOE_ROUTER_PROFILES.get(alias)
    if profile is None:
        raise SkillError(f"config.json model_type {alias!r} has no MoE router profile")
    if not profile["implemented"]:
        raise SkillError(
            f"config.json model_type {alias!r} is not supported: {profile['reason']}"
        )
    return profile


def _moe_head_dim(config: dict[str, Any], hidden: int, heads: int) -> int:
    value = config.get("head_dim")
    return hidden // heads if value is None else _config_int(config, "head_dim")


def unsupported_moe_features(config: dict[str, Any]) -> list[str]:
    """Return every unsupported computation feature for the sparse-MoE subset."""
    moe_specific = MOE_KEYS | MOE_UNSUPPORTED_KEYS | frozenset({
        "chunk_size_feed_forward",
        "layer_types",
        "mlp_only_layers",
        "output_router_logits",
        "qkv_bias",
        "rope_parameters",
        "router_bias",
        "top_k",
        "attention_multiplier",
        "embedding_multiplier",
        "logits_scaling",
        "residual_multiplier",
    })
    dense_view = {key: value for key, value in config.items() if key not in moe_specific}
    if dense_view.get("sliding_window") is None:
        dense_view.pop("sliding_window", None)
    if dense_view.get("max_window_layers") is None:
        dense_view.pop("max_window_layers", None)
    if dense_view.get("sliding_window") == 0 and dense_view.get("use_sliding_window") is False:
        dense_view.pop("sliding_window", None)
        dense_view.pop("max_window_layers", None)
    errors = unsupported_dense_features(dense_view)

    chunk_size = config.get("chunk_size_feed_forward", 0)
    if type(chunk_size) is not int or chunk_size != 0:
        errors.append("chunk_size_feed_forward must be 0")

    for key in sorted(MOE_UNSUPPORTED_KEYS):
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        if key in {"add_router_probs", "aux_loss_at_inference", "drop_tokens", "token_drop"} and value is False:
            continue
        if key == "expert_parallelism" and value is False:
            continue
        if key in {"n_group", "num_expert_groups", "topk_group"} and type(value) is int and value == 1:
            continue
        if key == "routed_scaling_factor" and not isinstance(value, bool) and isinstance(value, (int, float)) and value == 1.0:
            continue
        if key == "scoring_func" and isinstance(value, str) and value.lower() == "softmax":
            continue
        if key == "topk_method" and isinstance(value, str) and value.lower() in {"greedy", "default"}:
            continue
        errors.append(f"{key}={value!r} is not supported by the single-device softmax router")

    for key in (
        "attention_multiplier",
        "embedding_multiplier",
        "logits_scaling",
        "residual_multiplier",
    ):
        if key in config:
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 1.0:
                errors.append(f"{key} must be the neutral numeric value 1.0")

    if "pruned_heads" in config:
        pruning = config["pruned_heads"]
        if not isinstance(pruning, dict):
            errors.append("pruned_heads must be an object when present")
        elif pruning:
            errors.append("pruned_heads must be empty; attention-head pruning is not supported")

    for key in (
        "moe_shared_expert_intermediate_size",
        "num_shared_experts",
        "shared_expert_intermediate_size",
    ):
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if type(value) is int and value == 0:
            continue
        errors.append(f"shared-expert config key {key!r} is set; shared experts are not supported")

    for key in ("decoder_sparse_step", "expert_interval", "moe_layer_freq"):
        if key in config and config[key] is not None:
            value = config[key]
            if type(value) is not int or value != 1:
                errors.append(f"{key}={value!r} is not supported; every decoder layer must be MoE")
    if config.get("mlp_only_layers") not in (None, []):
        errors.append("mlp_only_layers must be empty; mixed dense/MoE layer schedules are not supported")
    layer_types = config.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list)
        or not layer_types
        or any(str(value).lower() != "full_attention" for value in layer_types)
    ):
        errors.append("layer_types must contain only 'full_attention'")
    for key in ("norm_topk_prob", "output_router_logits", "qkv_bias", "router_bias"):
        if key in config and not isinstance(config[key], bool):
            errors.append(f"{key} must be boolean when present")
    if config.get("output_router_logits") is True:
        errors.append("output_router_logits=True is not supported; auxiliary router outputs are not captured")
    if config.get("router_bias") is True:
        errors.append("router_bias=True is not supported")
    jitter = config.get("router_jitter_noise", 0.0)
    if isinstance(jitter, bool) or not isinstance(jitter, (int, float)) or jitter != 0.0:
        errors.append("router_jitter_noise must be 0.0 for deterministic eager inference")

    rope_parameters = config.get("rope_parameters")
    if rope_parameters is not None:
        if not isinstance(rope_parameters, dict):
            errors.append("rope_parameters must be null or an object")
        else:
            rope_type = _rope_type(rope_parameters)
            allowed = {"rope_type", "type", "rope_theta"}
            if rope_type in {"dynamic", "linear"}:
                allowed.update({"factor", "original_max_position_embeddings"})
            if rope_type not in SUPPORTED_ROPE_TYPES:
                errors.append(f"rope_parameters type {rope_type!r} is not supported")
            for key in sorted(set(rope_parameters) - allowed):
                errors.append(f"rope_parameters key {key!r} is not supported")
            if rope_type in {"dynamic", "linear"}:
                factor = rope_parameters.get("factor")
                if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 1.0:
                    errors.append(
                        f"rope_parameters factor must be a number greater than 1 for type {rope_type!r}"
                    )
    if rope_parameters is not None and config.get("rope_scaling") is not None:
        errors.append("rope_parameters and rope_scaling cannot both be set")
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters and "rope_theta" in config:
        if rope_parameters["rope_theta"] != config["rope_theta"]:
            errors.append("rope_parameters.rope_theta and rope_theta must agree")

    return errors


def validate_moe_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SkillError("config.json must contain an object")
    errors = unsupported_moe_features(config)
    if errors:
        raise SkillError(
            "Unsupported config features; no code was generated. "
            f"Consult {MOE_RUNBOOK}:\n- " + "\n- ".join(errors)
        )
    moe_router_profile(config)
    hidden = _config_int(config, "hidden_size")
    heads = _config_int(config, "num_attention_heads")
    kv_heads = _config_int(config, "num_key_value_heads", default=heads)
    head_dim = _moe_head_dim(config, hidden, heads)
    _config_int(config, "num_hidden_layers")
    _config_int(config, "vocab_size")
    _config_int(config, "intermediate_size")
    _config_int(config, "max_position_embeddings", default=2048)
    _config_float(config, "rms_norm_eps", default=1e-5)
    rope_parameters = config.get("rope_parameters")
    rope_theta_default = (
        float(rope_parameters.get("rope_theta", 10000.0))
        if isinstance(rope_parameters, dict) else 10000.0
    )
    _config_float(config, "rope_theta", default=rope_theta_default)
    experts = _aliased_positive_int(config, "num_experts", "num_local_experts")
    top_k = _aliased_positive_int(config, "num_experts_per_tok", "top_k")
    _config_int(
        config,
        "moe_intermediate_size",
        default=_config_int(config, "intermediate_size"),
    )
    if top_k > experts:
        raise SkillError("config.json num_experts_per_tok/top_k must not exceed the expert count")
    if hidden != heads * head_dim:
        raise SkillError("config.json hidden_size must equal num_attention_heads * head_dim")
    if heads % kv_heads:
        raise SkillError("config.json num_attention_heads must be divisible by num_key_value_heads")
    for key in ("attention_bias", "mlp_bias", "rope_traditional", "tie_word_embeddings"):
        if key in config and not isinstance(config[key], bool):
            raise SkillError(f"config.json {key} must be boolean")
    rope_config = config.get("rope_scaling", rope_parameters)
    if isinstance(rope_config, dict) and _rope_type(rope_config) == "dynamic" and head_dim <= 2:
        raise SkillError("config.json dynamic RoPE requires head_dim greater than 2")
    return config


def unsupported_encoder_features(config: dict[str, Any]) -> list[str]:
    """Return every config feature outside the supported absolute-position BERT path."""
    errors: list[str] = []
    model_type = config.get("model_type")
    if model_type != "bert":
        errors.append(f"model_type={model_type!r} is not supported; only 'bert' is accepted")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(
            isinstance(value, str) and value in SUPPORTED_BERT_ARCHITECTURES
            for value in architectures
        )
    ):
        errors.append(
            "architectures must contain only supported BERT identities: "
            + ", ".join(sorted(SUPPORTED_BERT_ARCHITECTURES))
        )
    for key in ("is_decoder", "is_encoder_decoder", "add_cross_attention"):
        value = config.get(key, False)
        if not isinstance(value, bool):
            errors.append(f"{key} must be boolean when present")
        elif value:
            errors.append(f"{key}=true is not supported by the bidirectional BERT scaffold")
    activation = config.get("hidden_act", "gelu")
    if not isinstance(activation, str) or activation.lower() != "gelu":
        errors.append("hidden_act must be 'gelu'")
    position_type = config.get("position_embedding_type", "absolute")
    if not isinstance(position_type, str) or position_type.lower() != "absolute":
        errors.append(
            f"position_embedding_type={position_type!r} is not supported; only 'absolute' is accepted"
        )
    classified = (
        ENCODER_CONFIG_FEATURE_ALLOWLIST
        | ENCODER_INFERENCE_METADATA_KEYS
        | KNOWN_METADATA_KEYS
    )
    for key in sorted(set(config) - classified):
        # Presence is computation-relevant even when the value is falsey. This is
        # intentionally not filtered through _meaningfully_set().
        errors.append(f"unrecognized computation-relevant config key {key!r}")
    return errors


def validate_encoder_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SkillError("config.json must contain an object")
    errors = unsupported_encoder_features(config)
    if errors:
        raise SkillError(
            "Unsupported config features; no code was generated. "
            f"Consult {ENCODER_RUNBOOK}:\n- " + "\n- ".join(errors)
        )
    hidden = _config_int(config, "hidden_size")
    heads = _config_int(config, "num_attention_heads")
    _config_int(config, "num_hidden_layers")
    _config_int(config, "intermediate_size")
    _config_int(config, "max_position_embeddings")
    _config_int(config, "type_vocab_size")
    _config_int(config, "vocab_size")
    _config_float(config, "layer_norm_eps", default=1e-12)
    if hidden % heads:
        raise SkillError("config.json hidden_size must be divisible by num_attention_heads")
    pad_token_id = config.get("pad_token_id", 0)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise SkillError("config.json pad_token_id must be a non-negative integer")
    return config


def _inspection_tensor_shapes(inspection: dict[str, Any]) -> dict[str, list[int]]:
    tensors = inspection.get("tensors")
    if not isinstance(tensors, list):
        raise SkillError("inspection tensor inventory must be a list")
    result: dict[str, list[int]] = {}
    for index, item in enumerate(tensors):
        if not isinstance(item, dict):
            raise SkillError(f"inspection tensor inventory entry {index} must be an object")
        key = item.get("key")
        shape = item.get("shape")
        if not isinstance(key, str) or not key or not isinstance(shape, list) or not all(
            type(value) is int and value >= 0 for value in shape
        ):
            raise SkillError(f"inspection tensor inventory entry {index} is invalid")
        if key in result:
            raise SkillError(f"inspection tensor inventory contains duplicate key {key!r}")
        result[key] = shape
    return result


def encoder_target_tensor_dict(config: dict[str, Any], *, pooler: bool) -> dict[str, list[int]]:
    hidden = _config_int(config, "hidden_size")
    intermediate = _config_int(config, "intermediate_size")
    layers = _config_int(config, "num_hidden_layers")
    tensors: dict[str, list[int]] = {
        "embeddings.LayerNorm.bias": [hidden],
        "embeddings.LayerNorm.weight": [hidden],
        "embeddings.position_embeddings.weight": [
            _config_int(config, "max_position_embeddings"), hidden
        ],
        "embeddings.token_type_embeddings.weight": [
            _config_int(config, "type_vocab_size"), hidden
        ],
        "embeddings.word_embeddings.weight": [_config_int(config, "vocab_size"), hidden],
    }
    for index in range(layers):
        prefix = f"encoder.layer.{index}"
        tensors.update({
            f"{prefix}.attention.self.query.weight": [hidden, hidden],
            f"{prefix}.attention.self.query.bias": [hidden],
            f"{prefix}.attention.self.key.weight": [hidden, hidden],
            f"{prefix}.attention.self.key.bias": [hidden],
            f"{prefix}.attention.self.value.weight": [hidden, hidden],
            f"{prefix}.attention.self.value.bias": [hidden],
            f"{prefix}.attention.output.dense.weight": [hidden, hidden],
            f"{prefix}.attention.output.dense.bias": [hidden],
            f"{prefix}.attention.output.LayerNorm.weight": [hidden],
            f"{prefix}.attention.output.LayerNorm.bias": [hidden],
            f"{prefix}.intermediate.dense.weight": [intermediate, hidden],
            f"{prefix}.intermediate.dense.bias": [intermediate],
            f"{prefix}.output.dense.weight": [hidden, intermediate],
            f"{prefix}.output.dense.bias": [hidden],
            f"{prefix}.output.LayerNorm.weight": [hidden],
            f"{prefix}.output.LayerNorm.bias": [hidden],
        })
    if pooler:
        tensors.update({
            "pooler.dense.weight": [hidden, hidden],
            "pooler.dense.bias": [hidden],
        })
    return tensors


def validate_encoder_topology(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Validate the inspected BERT tensor graph, not merely config identity."""
    actual = _inspection_tensor_shapes(inspection)
    forbidden_markers = (
        "decoder", "crossattention", "cross_attn", "cross_attn", "expert", "router",
    )
    forbidden = sorted(
        key for key in actual if any(marker in key.lower() for marker in forbidden_markers)
    )
    if forbidden:
        raise SkillError(
            "Unsupported encoder tensor topology; no code was generated. "
            "Found decoder/cross-attention/expert tensors: " + ", ".join(forbidden[:8])
        )
    pooler_keys = {"pooler.dense.weight", "pooler.dense.bias"}
    present_pooler = pooler_keys & set(actual)
    if present_pooler and present_pooler != pooler_keys:
        raise SkillError("encoder tensor topology has an incomplete pooler")
    pooler = bool(present_pooler)
    expected = encoder_target_tensor_dict(config, pooler=pooler)
    allowed_buffers = {"embeddings.position_ids", "embeddings.token_type_ids"}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected) - allowed_buffers)
    mismatched = sorted(
        key for key in set(expected) & set(actual) if expected[key] != actual[key]
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append("missing tensors: " + ", ".join(missing[:8]))
        if unexpected:
            details.append("unexpected tensors: " + ", ".join(unexpected[:8]))
        if mismatched:
            details.append(
                "shape mismatches: "
                + ", ".join(
                    f"{key} expected {expected[key]} got {actual[key]}" for key in mismatched[:8]
                )
            )
        raise SkillError(
            "Unsupported encoder tensor topology; no code was generated. " + "; ".join(details)
        )
    return pooler


CONFIG_TEMPLATE = r'''__HEADER__
"""Validated config.json parser for the generated dense decoder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _positive_int(data: dict[str, Any], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    rope_traditional: bool
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    max_position_embeddings: int
    hidden_act: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        hidden_size = _positive_int(data, "hidden_size")
        heads = _positive_int(data, "num_attention_heads")
        kv_heads = _positive_int(data, "num_key_value_heads", heads)
        head_dim = _positive_int(data, "head_dim", hidden_size // heads)
        if hidden_size != heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if heads % kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rope_scaling = data.get("rope_scaling")
        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            raise ValueError("rope_scaling must be null or an object")
        activation = data.get("hidden_act", data.get("activation_function", "silu"))
        if not isinstance(activation, str):
            raise ValueError("hidden_act must be a string")
        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=_positive_int(data, "num_hidden_layers"),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=_positive_int(data, "intermediate_size"),
            rms_norm_eps=_positive_float(data, "rms_norm_eps", 1e-5),
            rope_theta=_positive_float(data, "rope_theta", 10000.0),
            rope_scaling=dict(rope_scaling) if rope_scaling is not None else None,
            rope_traditional=bool(data.get("rope_traditional", False)),
            vocab_size=_positive_int(data, "vocab_size"),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            attention_bias=bool(data.get("attention_bias", False)),
            mlp_bias=bool(data.get("mlp_bias", False)),
            max_position_embeddings=_positive_int(data, "max_position_embeddings", 2048),
            hidden_act=activation.lower(),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(data)
'''


MODEL_TEMPLATE = r'''__HEADER__
"""Minimal eager MLX dense decoder with GQA/MHA and growing KV caches."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ModelConfig


def _require_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError("generated model requires MLX: python3 -m pip install mlx") from exc
    return mx, nn


def _rope_kind(config: ModelConfig) -> str:
    scaling = config.rope_scaling or {}
    return str(scaling.get("rope_type", scaling.get("type", "default"))).lower()


def _dynamic_rope_limit(config: ModelConfig) -> int | None:
    if _rope_kind(config) != "dynamic":
        return None
    scaling = config.rope_scaling or {}
    return int(
        scaling.get("original_max_position_embeddings", config.max_position_embeddings)
    )


def build_model(config: ModelConfig):
    """Build and return an MLX nn.Module without importing MLX at module import time."""
    mx, nn = _require_mlx()

    def activate(x):
        if config.hidden_act in {"silu", "swish"}:
            return nn.silu(x)
        if config.hidden_act == "gelu":
            return nn.gelu(x)
        if config.hidden_act == "relu":
            return nn.relu(x)
        raise ValueError(f"unsupported activation {config.hidden_act!r}")

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            query_dims = config.num_attention_heads * config.head_dim
            kv_dims = config.num_key_value_heads * config.head_dim
            self.q_proj = nn.Linear(config.hidden_size, query_dims, bias=__Q_PROJ_BIAS__)
            self.k_proj = nn.Linear(config.hidden_size, kv_dims, bias=__K_PROJ_BIAS__)
            self.v_proj = nn.Linear(config.hidden_size, kv_dims, bias=__V_PROJ_BIAS__)
            self.o_proj = nn.Linear(query_dims, config.hidden_size, bias=__O_PROJ_BIAS__)

        def _apply_rope(self, value, offset: int, total_length: int):
            scaling = config.rope_scaling or {}
            kind = _rope_kind(config)
            factor = float(scaling.get("factor", 1.0))
            base = config.rope_theta
            scale = 1.0 / factor if kind == "linear" else 1.0
            original = int(scaling.get("original_max_position_embeddings", config.max_position_embeddings))
            if kind == "dynamic" and total_length > original:
                ratio = factor * total_length / original - (factor - 1.0)
                base *= ratio ** (config.head_dim / (config.head_dim - 2.0))
            return mx.fast.rope(
                value,
                config.head_dim,
                traditional=config.rope_traditional,
                base=base,
                scale=scale,
                offset=offset,
            )

        def __call__(self, x, cache=None, attention_mask=None):
            batch, length, _ = x.shape
            q = self.q_proj(x).reshape(batch, length, config.num_attention_heads, config.head_dim)
            k = self.k_proj(x).reshape(batch, length, config.num_key_value_heads, config.head_dim)
            v = self.v_proj(x).reshape(batch, length, config.num_key_value_heads, config.head_dim)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            offset = 0 if cache is None else int(cache[0].shape[2])
            if cache is not None:
                k = mx.concatenate((cache[0], k), axis=2)
                v = mx.concatenate((cache[1], v), axis=2)
            new_cache = (k, v)
            total_length = k.shape[2]
            q = self._apply_rope(q, offset, total_length)
            rotated_k = self._apply_rope(k, 0, total_length)
            repeats = config.num_attention_heads // config.num_key_value_heads
            expanded_k = mx.repeat(rotated_k, repeats, axis=1) if repeats != 1 else rotated_k
            expanded_v = mx.repeat(v, repeats, axis=1) if repeats != 1 else v
            scores = (
                (q @ expanded_k.transpose(0, 1, 3, 2)) * (config.head_dim ** -0.5)
            ).astype(mx.float32)
            mask_value = mx.array(-1e9, dtype=mx.float32)
            key_length = expanded_k.shape[2]
            query_positions = mx.arange(offset, offset + length)[:, None]
            key_positions = mx.arange(key_length)[None, :]
            allowed = key_positions <= query_positions
            scores = mx.where(allowed[None, None, :, :], scores, mask_value)
            query_mask = None
            if attention_mask is not None:
                if attention_mask.shape[-1] != key_length:
                    raise ValueError("attention_mask length must include the complete cached context")
                scores = mx.where(
                    attention_mask[:, None, None, :].astype(mx.bool_),
                    scores,
                    mask_value,
                )
                query_mask = attention_mask[:, offset : offset + length]
            probabilities = mx.softmax(scores, axis=-1).astype(q.dtype)
            if query_mask is not None:
                probabilities = mx.where(
                    query_mask[:, None, :, None].astype(mx.bool_),
                    probabilities,
                    mx.zeros_like(probabilities),
                )
            attended = probabilities @ expanded_v
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, config.hidden_size)
            return self.o_proj(attended), new_cache

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

        def __call__(self, x):
            return self.down_proj(activate(self.gate_proj(x)) * self.up_proj(x))

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()
            self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def __call__(self, x, cache=None, attention_mask=None):
            attention, new_cache = self.self_attn(
                self.input_layernorm(x),
                cache=cache,
                attention_mask=attention_mask,
            )
            hidden = x + attention
            mlp = self.mlp(self.post_attention_layernorm(hidden))
            return hidden + mlp, new_cache, attention, mlp

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = [DecoderLayer() for _ in range(config.num_hidden_layers)]
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def __call__(self, input_ids, cache=None, attention_mask=None, capture=False):
            if cache is not None and len(cache) != len(self.layers):
                raise ValueError("cache must contain one (key, value) pair per decoder layer")
            dynamic_limit = _dynamic_rope_limit(config)
            if cache is not None and dynamic_limit is not None:
                cached_length = int(cache[0][0].shape[2])
                if cached_length + input_ids.shape[1] > dynamic_limit:
                    raise ValueError(
                        "dynamic-NTK RoPE requires full-sequence recomputation "
                        "past original_max_position_embeddings"
                    )
            hidden = self.embed_tokens(input_ids)
            captures = {"embed": hidden} if capture else {}
            updated = []
            for index, layer in enumerate(self.layers):
                layer_cache = None if cache is None else cache[index]
                hidden, layer_cache, attention, mlp = layer(
                    hidden,
                    cache=layer_cache,
                    attention_mask=attention_mask,
                )
                updated.append(layer_cache)
                if capture:
                    captures[f"layer.{index}.attention"] = attention
                    captures[f"layer.{index}.mlp"] = mlp
                    captures[f"layer.{index}.hidden"] = hidden
            hidden = self.norm(hidden)
            if capture:
                captures["final_norm"] = hidden
            return hidden, updated, captures

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.model = Backbone()
            if not config.tie_word_embeddings:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def __call__(self, input_ids, *, attention_mask=None, cache=None, capture=False):
            hidden, updated, captures = self.model(
                input_ids,
                cache=cache,
                attention_mask=attention_mask,
                capture=capture,
            )
            logits = (
                self.model.embed_tokens.as_linear(hidden)
                if config.tie_word_embeddings
                else self.lm_head(hidden)
            )
            if capture:
                captures["logits"] = logits
                return logits, updated, captures
            return logits, updated

    return CausalLM()


def load_model(config_path: str | Path, weights_path: str | Path):
    mx, _ = _require_mlx()
    config = ModelConfig.from_file(config_path)
    model = build_model(config)
    weights = mx.load(str(weights_path))
    if not isinstance(weights, dict):
        raise ValueError("converted weights must load as a name-to-array mapping")
    model.load_weights(list(weights.items()), strict=True)
    getattr(mx, "eval")(model.parameters())
    return model


def greedy_generate(model, input_ids, max_new_tokens: int):
    mx, _ = _require_mlx()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    batch = input_ids.shape[0]
    if max_new_tokens == 0:
        return mx.zeros((batch, 0), dtype=input_ids.dtype)
    logits, cache = model(input_ids)
    sequence = input_ids
    model_config = getattr(model, "config", None)
    dynamic_limit = _dynamic_rope_limit(model_config) if model_config is not None else None
    generated = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(input_ids.dtype)
    for index in range(max_new_tokens):
        generated.append(next_token)
        if index + 1 < max_new_tokens:
            sequence = mx.concatenate((sequence, next_token[:, None]), axis=1)
            if dynamic_limit is not None and sequence.shape[1] > dynamic_limit:
                logits, _ = model(sequence)
                cache = None
            else:
                logits, cache = model(next_token[:, None], cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(input_ids.dtype)
    result = mx.stack(generated, axis=1)
    getattr(mx, "eval")(result)
    return result
'''


def _replace_template_once(source: str, old: str, new: str, *, label: str) -> str:
    if source.count(old) != 1:
        raise SkillError(f"internal {label} template seam is not unique")
    return source.replace(old, new, 1)


def moe_config_template() -> str:
    """Derive the MoE config parser while preserving the dense config contract."""
    source = CONFIG_TEMPLATE.replace(
        '"""Validated config.json parser for the generated dense decoder."""',
        '"""Validated config.json parser for the generated sparse-MoE decoder."""',
        1,
    )
    source = _replace_template_once(
        source,
        '        head_dim = _positive_int(data, "head_dim", hidden_size // heads)\n',
        '        head_dim_value = data.get("head_dim")\n'
        '        head_dim = (\n'
        '            hidden_size // heads if head_dim_value is None\n'
        '            else _positive_int(data, "head_dim")\n'
        '        )\n',
        label="MoE nullable head_dim",
    )
    source = _replace_template_once(
        source,
        "    intermediate_size: int\n",
        "    intermediate_size: int\n"
        "    moe_intermediate_size: int\n"
        "    num_experts: int\n"
        "    num_experts_per_tok: int\n"
        "    norm_topk_prob: bool\n",
        label="MoE config fields",
    )
    source = _replace_template_once(
        source,
        "        activation = data.get(\"hidden_act\", data.get(\"activation_function\", \"silu\"))\n"
        "        if not isinstance(activation, str):\n"
        "            raise ValueError(\"hidden_act must be a string\")\n"
        "        return cls(\n",
        "        activation = data.get(\"hidden_act\", data.get(\"activation_function\", \"silu\"))\n"
        "        if not isinstance(activation, str):\n"
        "            raise ValueError(\"hidden_act must be a string\")\n"
        "        expert_values = (data.get(\"num_experts\"), data.get(\"num_local_experts\"))\n"
        "        if all(value is not None for value in expert_values) and expert_values[0] != expert_values[1]:\n"
        "            raise ValueError(\"num_experts and num_local_experts must agree\")\n"
        "        num_experts = expert_values[0] if expert_values[0] is not None else expert_values[1]\n"
        "        if type(num_experts) is not int or num_experts <= 0:\n"
        "            raise ValueError(\"num_experts or num_local_experts must be a positive integer\")\n"
        "        top_k_values = (data.get(\"num_experts_per_tok\"), data.get(\"top_k\"))\n"
        "        if all(value is not None for value in top_k_values) and top_k_values[0] != top_k_values[1]:\n"
        "            raise ValueError(\"num_experts_per_tok and top_k must agree\")\n"
        "        top_k = top_k_values[0] if top_k_values[0] is not None else top_k_values[1]\n"
        "        if type(top_k) is not int or top_k <= 0 or top_k > num_experts:\n"
        "            raise ValueError(\"num_experts_per_tok/top_k must be positive and at most num_experts\")\n"
        "        model_type = str(data.get(\"model_type\", \"\")).lower()\n"
        "        if model_type in {\"mixtral\", \"granitemoe\"}:\n"
        "            norm_topk_prob = True\n"
        "        else:\n"
        "            norm_topk_prob = data.get(\"norm_topk_prob\", False)\n"
        "        if not isinstance(norm_topk_prob, bool):\n"
        "            raise ValueError(\"norm_topk_prob must be boolean\")\n"
        "        rope_parameters = data.get(\"rope_parameters\")\n"
        "        if rope_scaling is None and isinstance(rope_parameters, dict):\n"
        "            rope_scaling = {key: value for key, value in rope_parameters.items() if key != \"rope_theta\"}\n"
        "        return cls(\n",
        label="MoE config validation",
    )
    source = _replace_template_once(
        source,
        '            intermediate_size=_positive_int(data, "intermediate_size"),\n',
        '            intermediate_size=_positive_int(data, "intermediate_size"),\n'
        '            moe_intermediate_size=_positive_int(\n'
        '                data, "moe_intermediate_size", _positive_int(data, "intermediate_size")\n'
        '            ),\n'
        '            num_experts=num_experts,\n'
        '            num_experts_per_tok=top_k,\n'
        '            norm_topk_prob=norm_topk_prob,\n',
        label="MoE config construction",
    )
    source = _replace_template_once(
        source,
        '            rope_theta=_positive_float(data, "rope_theta", 10000.0),\n',
        '            rope_theta=_positive_float(\n'
        '                data, "rope_theta",\n'
        '                float(rope_parameters.get("rope_theta", 10000.0))\n'
        '                if isinstance(rope_parameters, dict) else 10000.0,\n'
        '            ),\n',
        label="MoE rope parameters",
    )
    return source


def moe_model_template() -> str:
    """Reuse the dense attention/cache model and replace only its feed-forward block."""
    source = MODEL_TEMPLATE.replace(
        '"""Minimal eager MLX dense decoder with GQA/MHA and growing KV caches."""',
        '"""Minimal eager MLX sparse-MoE decoder with GQA/MHA and growing KV caches."""',
        1,
    )
    dense_mlp = '''    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

        def __call__(self, x):
            return self.down_proj(activate(self.gate_proj(x)) * self.up_proj(x))
'''
    moe_mlp = '''    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(
                config.hidden_size, config.moe_intermediate_size, bias=config.mlp_bias
            )
            self.up_proj = nn.Linear(
                config.hidden_size, config.moe_intermediate_size, bias=config.mlp_bias
            )
            self.down_proj = nn.Linear(
                config.moe_intermediate_size, config.hidden_size, bias=config.mlp_bias
            )

        def __call__(self, x):
            return self.down_proj(activate(self.gate_proj(x)) * self.up_proj(x))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.experts = [Expert() for _ in range(config.num_experts)]

        def __call__(self, x):
            router_logits = self.gate(x)
            routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=-1, precise=True)
            top_k = config.num_experts_per_tok
            selected = mx.stop_gradient(
                mx.argpartition(-routing_weights, kth=top_k - 1, axis=-1)[..., :top_k]
            )
            scores = mx.take_along_axis(routing_weights, selected, axis=-1)
            if config.norm_topk_prob:
                scores = scores / mx.sum(scores, axis=-1, keepdims=True)
            combined = mx.zeros_like(x)
            for expert_index, expert in enumerate(self.experts):
                coefficient = mx.sum(
                    mx.where(selected == expert_index, scores, mx.zeros_like(scores)),
                    axis=-1,
                )
                active = coefficient[..., None] != 0
                expert_output = expert(x)
                contribution = mx.where(
                    active,
                    expert_output * coefficient[..., None].astype(x.dtype),
                    mx.zeros_like(expert_output),
                )
                combined = combined + contribution
            return combined
'''
    return _replace_template_once(source, dense_mlp, moe_mlp, label="MoE MLP")


GENERATE_TEMPLATE = r'''__HEADER__
"""Greedy tokenizer-free generation CLI for converted MLX weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import _require_mlx, greedy_generate, load_model


def parse_token_ids(values: list[str]) -> list[int]:
    pieces = [piece.strip() for value in values for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--token-ids requires comma- or space-separated integers")
    token_ids = [int(piece, 10) for piece in pieces]
    if any(value < 0 for value in token_ids):
        raise ValueError("--token-ids must be non-negative")
    return token_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Greedy-decode token IDs with generated MLX code")
    parser.add_argument("--weights", required=True, help="Converted .safetensors or .npz weights")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--token-ids", nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    mx, _ = _require_mlx()
    input_ids = mx.array([parse_token_ids(args.token_ids)], dtype=mx.int32)
    model = load_model(args.config, args.weights)
    generated = greedy_generate(model, input_ids, args.max_new_tokens)
    print(json.dumps({"generated_token_ids": generated.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


CAPTURE_TEMPLATE = r'''__HEADER__
"""Capture target MLX tensors using the source-oracle manifest contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from model import _require_mlx, greedy_generate, load_model


def parse_token_ids(values: list[str]) -> list[int]:
    pieces = [piece.strip() for value in values for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--token-ids requires comma- or space-separated integers")
    token_ids = [int(piece, 10) for piece in pieces]
    if any(value < 0 for value in token_ids):
        raise ValueError("--token-ids must be non-negative")
    return token_ids


def file_record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"name": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def tensor_inventory(arrays) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture generated MLX intermediate tensors")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--token-ids", nargs="+", required=True)
    parser.add_argument("--generate-steps", type=int, default=4)
    parser.add_argument("--output", required=True, help="Target .npz path")
    parser.add_argument("--manifest", help="Default: <output stem>.manifest.json")
    args = parser.parse_args()
    if args.generate_steps < 0:
        raise ValueError("--generate-steps must be non-negative")
    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end with .npz")
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    config_path = Path(args.config)
    weights_path = Path(args.weights)
    token_ids = parse_token_ids(args.token_ids)
    mx, _ = _require_mlx()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("capture.py requires NumPy") from exc
    input_ids = mx.array([token_ids], dtype=mx.int32)
    attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
    model = load_model(config_path, weights_path)
    logits, _, captures = model(input_ids, attention_mask=attention_mask, capture=True)
    generated = greedy_generate(model, input_ids, args.generate_steps)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        **captures,
        "logits": logits,
        "generated_token_ids": generated,
    }
    getattr(mx, "eval")(*tensors.values())
    arrays = {}
    for name, value in sorted(tensors.items()):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        arrays[name] = np.ascontiguousarray(array)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    config_record = file_record(config_path)
    weight_record = file_record(weights_path)
    identity = json.dumps(
        {"schema_version": 1, "config": config_record, "weights": [weight_record]},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "model": {
            "directory": config_path.parent.name or ".",
            "fingerprint": "sha256:" + hashlib.sha256(identity).hexdigest(),
            "config": config_record,
            "weights": {
                "algorithm": "sha256",
                "file_count": 1,
                "total_bytes": weight_record["size_bytes"],
                "max_listed_files": 256,
                "truncated": False,
                "omitted_file_count": 0,
                "files": [weight_record],
            },
        },
        "capture": {
            "prompts": None,
            "token_ids": token_ids,
            "generate_steps": args.generate_steps,
            "seed": 0,
            "dtype_policy": "float32",
        },
        "tensors": tensor_inventory(arrays),
        "libraries": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": "not-used",
            "transformers": "not-used",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "npz": str(output), "tensor_count": len(arrays)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


README_TEMPLATE = r'''# Generated MLX dense decoder

Generated by `scaffold_port.py` version __VERSION__ from trusted inspection
`sha256:__DIGEST__` for family `dense-decoder-transformer`.

This is a **starting implementation**, not a guaranteed port. It implements the
validated eager config path, but converted weights still require tensor and
end-to-end parity validation against `capture_oracle.py` before the result is
called correct.

## Generated files

- `config.json`: canonical copy of the inspected source config.
- `config.py`: validated config parser.
- `model.py`: lazy-imported eager MLX model, GQA/MHA attention, RoPE, growing KV cache, gated MLP, and greedy decode.
- `generate.py`: tokenizer-free greedy token-ID CLI.
- `capture.py`: target tensor capture in the source-oracle NPZ/manifest schema.
- `scaffold-manifest.json`: generator/config/code digests plus the target parameter contract used by conversion and package validation.

## Stable parameter-name scheme

Linear weights use MLX/Hugging Face layout `[output_dims, input_dims]`.

- `model.embed_tokens.weight` — `[vocab_size, hidden_size]`
- `model.layers.{i}.input_layernorm.weight` — `[hidden_size]`
- `model.layers.{i}.self_attn.q_proj.weight` — `[num_attention_heads * head_dim, hidden_size]`
- `model.layers.{i}.self_attn.k_proj.weight` — `[num_key_value_heads * head_dim, hidden_size]`
- `model.layers.{i}.self_attn.v_proj.weight` — `[num_key_value_heads * head_dim, hidden_size]`
- `model.layers.{i}.self_attn.o_proj.weight` — `[hidden_size, num_attention_heads * head_dim]`
- `model.layers.{i}.post_attention_layernorm.weight` — `[hidden_size]`
- `model.layers.{i}.mlp.gate_proj.weight` — `[intermediate_size, hidden_size]`
- `model.layers.{i}.mlp.up_proj.weight` — `[intermediate_size, hidden_size]`
- `model.layers.{i}.mlp.down_proj.weight` — `[hidden_size, intermediate_size]`
- `model.norm.weight` — `[hidden_size]`
- `lm_head.weight` — `[vocab_size, hidden_size]`, untied configs only

Attention projection biases are derived independently from the inspected source
tensor inventory: __ATTENTION_BIAS_SUMMARY__. When `mlp_bias` is true, append
`.bias` to each MLP projection name. A tied LM head has no separate parameter:
conversion must map the source tie owner to `model.embed_tokens.weight` and omit
`lm_head.weight`.

## Implemented assumptions

- decoder-only pre-RMSNorm blocks with separate Q/K/V/O projections;
- causal attention, optional padding mask, MHA or GQA with a growing cache;
- standard, linear, or dynamic-NTK RoPE selected from the inspected config;
- gated SiLU/SwiGLU, GELU/GeGLU, or ReLU/ReGLU MLP;
- no dropout, active sliding window, QK normalization, MoE, quantization, soft caps, or custom attention variants;
- `use_sliding_window=false` explicitly selects full attention even when inert sliding-window metadata remains in the source config.

For dynamic-NTK RoPE, the base changes once the sequence exceeds
`original_max_position_embeddings`. Greedy decoding therefore stops reusing the
KV cache at that boundary and uses full-sequence recomputation for every later
step. This preserves one RoPE base across every key at a decode step, at the
cost of losing cached-decode speed beyond the original context limit.

## Run and verify

```bash
python3 generate.py --weights converted.safetensors --token-ids 1 42 17 --max-new-tokens 4
python3 mlx-model-porting/scripts/run_parity.py --source-model MODEL --package mlx_port --weights converted --token-ids 1 42 17 --output parity-report.json
```

Before relying on output, validate every converted parameter name and shape,
compare every stable capture key, check greedy continuation agreement, and prove
prefill-plus-step KV-cache logits match full-context logits at declared tolerances.
'''


SSM_CONFIG_TEMPLATE = r'''__HEADER__
"""Validated config parser for the generated minimal selective SSM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_state: int
    d_conv: int
    expand: int
    n_layer: int
    vocab_size: int
    rms_norm_eps: float
    conv_bias: bool
    tie_word_embeddings: bool

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        if data.get("ssm_variant") != "minimal_selective":
            raise ValueError("ssm_variant must be exactly 'minimal_selective'")
        epsilon = data.get("rms_norm_eps", 1e-5)
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("rms_norm_eps must be a positive number")
        for key in ("conv_bias", "tie_word_embeddings"):
            if key in data and not isinstance(data[key], bool):
                raise ValueError(f"{key} must be boolean")
        return cls(
            d_model=_positive_int(data, "d_model"),
            d_state=_positive_int(data, "d_state"),
            d_conv=_positive_int(data, "d_conv"),
            expand=_positive_int(data, "expand"),
            n_layer=_positive_int(data, "n_layer"),
            vocab_size=_positive_int(data, "vocab_size"),
            rms_norm_eps=float(epsilon),
            conv_bias=bool(data.get("conv_bias", True)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(data)
'''


SSM_MODEL_TEMPLATE = r'''__HEADER__
"""Readable eager MLX minimal selective SSM with explicit recurrent state."""
from __future__ import annotations

from pathlib import Path

from config import ModelConfig


def _require_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError("generated model requires MLX: python3 -m pip install mlx") from exc
    return mx, nn


def _exprel(mx, value):
    """Stable expm1(x) / x with its analytic value at x=0."""
    small = mx.abs(value) < 1e-4
    square = value * value
    series = 1.0 + value * 0.5 + square / 6.0 + square * value / 24.0
    safe_value = mx.where(small, mx.ones_like(value), value)
    quotient = mx.expm1(value) / safe_value
    return mx.where(small, series, quotient)


def build_model(config: ModelConfig):
    """Build the correctness-first loop recurrence; no fused scan is used."""
    mx, nn = _require_mlx()

    class DepthwiseConv1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((config.d_inner, config.d_conv))
            if config.conv_bias:
                self.bias = mx.zeros((config.d_inner,))

        def __call__(self, window):
            # Weight index 0 is the oldest sample and index d_conv-1 is current.
            value = mx.sum(window * self.weight.T[None, :, :], axis=1)
            return value + self.bias if config.conv_bias else value

    class SelectiveMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=False)
            self.conv1d = DepthwiseConv1d()
            # Direct per-channel dt plus token-dependent B and C.
            self.x_proj = nn.Linear(
                config.d_inner,
                config.d_inner + 2 * config.d_state,
                bias=False,
            )
            self.dt_bias = mx.zeros((config.d_inner,))
            self.A_log = mx.zeros((config.d_inner, config.d_state))
            self.D = mx.ones((config.d_inner,))
            self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

        def initial_state(self, batch: int, dtype):
            return (
                mx.zeros((batch, config.d_conv - 1, config.d_inner), dtype=dtype),
                mx.zeros((batch, config.d_inner, config.d_state), dtype=mx.float32),
            )

        def step(self, value, state=None):
            projected, gate = mx.split(self.in_proj(value), 2, axis=-1)
            projected = projected.astype(mx.float32)
            gate = gate.astype(mx.float32)
            if state is None:
                state = self.initial_state(value.shape[0], projected.dtype)
            conv_state, ssm_state = state
            if conv_state.shape != (value.shape[0], config.d_conv - 1, config.d_inner):
                raise ValueError("convolution state shape does not match batch/config")
            if ssm_state.shape != (value.shape[0], config.d_inner, config.d_state):
                raise ValueError("SSM state shape does not match batch/config")
            window = mx.concatenate((conv_state, projected[:, None, :]), axis=1)
            convolved = self.conv1d(window)
            activated = convolved * mx.sigmoid(convolved)
            parameters = self.x_proj(activated.astype(mx.float32))
            dt_raw = parameters[:, : config.d_inner]
            B = parameters[:, config.d_inner : config.d_inner + config.d_state]
            C = parameters[:, config.d_inner + config.d_state :]
            dt_input = dt_raw + self.dt_bias.astype(mx.float32)
            dt = mx.logaddexp(dt_input, mx.zeros_like(dt_input))
            A = -mx.exp(self.A_log.astype(mx.float32))
            scaled_A = dt[:, :, None] * A[None, :, :]
            decay = mx.exp(scaled_A)
            # Exact zero-order-hold input coefficient for constant B*x over dt:
            # integral_0^dt exp(A*(dt-tau)) d tau = dt * exprel(dt*A).
            input_coefficient = dt[:, :, None] * _exprel(mx, scaled_A)
            driven = (
                input_coefficient
                * B[:, None, :]
                * activated.astype(mx.float32)[:, :, None]
            )
            next_ssm = decay * ssm_state.astype(mx.float32) + driven
            y = mx.sum(next_ssm * C[:, None, :], axis=-1)
            y = y + self.D.astype(mx.float32)[None, :] * activated.astype(mx.float32)
            gated = y * (gate * mx.sigmoid(gate))
            output = self.out_proj(gated)
            next_conv = window[:, 1:, :]
            return output, (next_conv, next_ssm)

        def __call__(self, values, state=None):
            outputs = []
            current = state
            for index in range(values.shape[1]):
                output, current = self.step(values[:, index, :], current)
                outputs.append(output)
            if not outputs:
                raise ValueError("SSM input sequence must contain at least one token")
            return mx.stack(outputs, axis=1), current

    class SSMBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.mixer = SelectiveMixer()

        def __call__(self, values, state=None):
            branch, updated = self.mixer(self.norm(values), state=state)
            return values + branch, updated, branch

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = [SSMBlock() for _ in range(config.n_layer)]
            self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def __call__(self, input_ids, state=None, attention_mask=None, capture=False):
            if input_ids.ndim != 2 or input_ids.shape[1] == 0:
                raise ValueError("input_ids must have shape [batch, nonzero_length]")
            if attention_mask is not None and attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must equal input_ids shape")
            if attention_mask is not None and not bool(mx.all(attention_mask).item()):
                raise ValueError(
                    "minimal selective SSM scaffold supports unpadded fixtures only"
                )
            if state is not None and len(state) != len(self.layers):
                raise ValueError("state must contain one (conv, SSM) pair per layer")
            hidden = self.embed_tokens(input_ids)
            captures = {"embed": hidden} if capture else {}
            updated = []
            for index, layer in enumerate(self.layers):
                layer_state = None if state is None else state[index]
                hidden, layer_state, branch = layer(hidden, state=layer_state)
                updated.append(layer_state)
                if capture:
                    captures[f"layer.{index}.ssm"] = branch
                    captures[f"layer.{index}.hidden"] = hidden
            hidden = self.norm(hidden)
            if capture:
                captures["final_norm"] = hidden
            return hidden, updated, captures

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.model = Backbone()
            if not config.tie_word_embeddings:
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        def __call__(self, input_ids, *, attention_mask=None, state=None, cache=None, capture=False):
            if state is not None and cache is not None:
                raise ValueError("pass recurrent state as state or cache, not both")
            recurrent_state = state if state is not None else cache
            hidden, updated, captures = self.model(
                input_ids,
                state=recurrent_state,
                attention_mask=attention_mask,
                capture=capture,
            )
            logits = (
                self.model.embed_tokens.as_linear(hidden)
                if config.tie_word_embeddings
                else self.lm_head(hidden)
            )
            if capture:
                captures["logits"] = logits
                return logits, updated, captures
            return logits, updated

    return CausalLM()


def load_model(config_path: str | Path, weights_path: str | Path):
    mx, _ = _require_mlx()
    config = ModelConfig.from_file(config_path)
    model = build_model(config)
    weights = mx.load(str(weights_path))
    if not isinstance(weights, dict):
        raise ValueError("converted weights must load as a name-to-array mapping")
    model.load_weights(list(weights.items()), strict=True)
    getattr(mx, "eval")(model.parameters())
    return model


def greedy_generate(model, input_ids, max_new_tokens: int):
    mx, _ = _require_mlx()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return mx.zeros((input_ids.shape[0], 0), dtype=input_ids.dtype)
    logits, state = model(input_ids)
    generated = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(input_ids.dtype)
    for index in range(max_new_tokens):
        generated.append(next_token)
        if index + 1 < max_new_tokens:
            logits, state = model(next_token[:, None], state=state)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(input_ids.dtype)
    result = mx.stack(generated, axis=1)
    getattr(mx, "eval")(result)
    return result
'''


SSM_README_TEMPLATE = r'''# Generated MLX minimal selective SSM

Generated by `scaffold_port.py` version __VERSION__ from trusted inspection
`sha256:__DIGEST__` for family `ssm-recurrent-hybrid`.

This is a correctness-first **synthetic minimal-selective variant**, not a
checkpoint-compatible implementation of upstream Mamba, Mamba2, RWKV, Jamba,
Zamba, Griffin, or RecurrentGemma. Generation is allowed only when config.json
opts in with `ssm_variant: minimal_selective`. Attention-mixed/hybrid configs
fail closed.

Each pre-RMSNorm block implements an input/gate projection, causal depthwise
convolution, token-dependent dt/B/C, stable A=-exp(A_log), exact zero-order-hold
discretization, D skip, SiLU gate, output projection, and residual. The eager
full-sequence path deliberately calls the same one-token recurrence used by
decode, keeping explicit `(convolution_state, ssm_state)` per layer. MLX 0.30.4
provides `mx.conv1d` but no `mx.scan` in the validated environment; this
scaffold keeps the readable per-step loop required by the SSM runbook.

## Parameter contract

- `model.embed_tokens.weight` -- `[vocab_size, d_model]`
- `model.layers.{i}.norm.weight` -- `[d_model]`
- `model.layers.{i}.mixer.in_proj.weight` -- `[2 * d_inner, d_model]`
- `model.layers.{i}.mixer.conv1d.weight` -- `[d_inner, d_conv]`, oldest to current
- `model.layers.{i}.mixer.conv1d.bias` -- `[d_inner]` when `conv_bias=true`
- `model.layers.{i}.mixer.x_proj.weight` -- `[d_inner + 2 * d_state, d_inner]`
- `model.layers.{i}.mixer.dt_bias` -- `[d_inner]`
- `model.layers.{i}.mixer.A_log` -- `[d_inner, d_state]`
- `model.layers.{i}.mixer.D` -- `[d_inner]`
- `model.layers.{i}.mixer.out_proj.weight` -- `[d_model, d_inner]`
- `model.norm.weight` -- `[d_model]`
- `lm_head.weight` -- `[vocab_size, d_model]` only when embeddings are untied

## Proof boundary

The repository synthetic gate checks this recurrence against an independent
NumPy implementation and checks full recomputation against carried-state token
decoding on deterministic FP32 fixtures. Real-checkpoint weight conversion and parity against an
upstream torch-mamba/Mamba2 implementation remain required before claiming
checkpoint support. No tolerance is relaxed for this scaffold.
'''


MOE_README_TEMPLATE = r'''# Generated MLX sparse-MoE decoder

Generated by `scaffold_port.py` version __VERSION__ from trusted inspection
`sha256:__DIGEST__` for family `moe-decoder-transformer`.

This is a **starting implementation**, not a guaranteed port. It implements a
readable single-device sparse-expert loop so routing and expert combination can
be parity-tested before any dispatch optimization.

## Stable parameter-name scheme

Linear weights use MLX/Hugging Face layout `[output_dims, input_dims]`.

- `model.embed_tokens.weight` — `[vocab_size, hidden_size]`
- `model.layers.{i}.input_layernorm.weight` — `[hidden_size]`
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` — standard dense-decoder attention names
- `model.layers.{i}.post_attention_layernorm.weight` — `[hidden_size]`
- `model.layers.{i}.mlp.gate.weight` — `[num_experts, hidden_size]` router
- `model.layers.{i}.mlp.experts.{e}.gate_proj.weight` — `[moe_intermediate_size, hidden_size]`
- `model.layers.{i}.mlp.experts.{e}.up_proj.weight` — `[moe_intermediate_size, hidden_size]`
- `model.layers.{i}.mlp.experts.{e}.down_proj.weight` — `[hidden_size, moe_intermediate_size]`
- `model.norm.weight` — `[hidden_size]`
- `lm_head.weight` — `[vocab_size, hidden_size]`, untied configs only

Attention projection biases are derived independently from inspected tensors:
__ATTENTION_BIAS_SUMMARY__. Expert projection biases follow `mlp_bias`; the
router is bias-free.

## Implemented assumptions

- every decoder layer uses the same sparse-MoE structure;
- router softmax is evaluated in float32, then top-k experts are selected;
- Mixtral and GraniteMoE always renormalize selected probabilities; Qwen2-MoE
  follows its boolean `norm_topk_prob` value;
- expert MLPs use the configured gated activation and a readable expert loop;
- dense-decoder GQA/MHA, RoPE, padding mask, growing KV cache, and greedy decode are reused unchanged;
- shared experts, grouped routing, capacity/token dropping, expert parallelism,
  router jitter, inference auxiliary outputs, non-softmax scoring, mixed
  dense/MoE schedules, quantization, unimplemented router aliases, and custom
  attention variants fail closed;
- generation requires the complete inspected tensor inventory to match the
  selected alias profile exactly, including every layer and routed expert.

## Run and verify

```bash
python3 generate.py --weights converted.safetensors --token-ids 1 42 17 --max-new-tokens 4
python3 mlx-model-porting/scripts/run_parity.py --source-model MODEL --package mlx_port --weights converted --token-ids 1 42 17 --output parity-report.json
```

Before relying on output, validate every expert index and tensor shape, compare
every stable capture rung, check exact greedy continuation, and prove cached
decode logits match full-context logits.
'''


def _header(digest: str) -> str:
    return f"# Generated by scaffold_port.py {GENERATOR_VERSION} from inspection sha256:{digest}."


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def dense_attention_biases(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    """Derive a complete per-projection bias contract from inspected source keys."""
    layers = _config_int(config, "num_hidden_layers")
    tensor_keys = {
        str(tensor.get("key"))
        for tensor in inspection.get("tensors", [])
        if isinstance(tensor, dict) and isinstance(tensor.get("key"), str)
    }
    biases: dict[str, bool] = {}
    for projection in ATTENTION_PROJECTIONS:
        expected = {
            f"model.layers.{index}.self_attn.{projection}.bias"
            for index in range(layers)
        }
        present = expected & tensor_keys
        if present and present != expected:
            missing = sorted(expected - present)
            raise SkillError(
                f"inconsistent {projection} bias coverage across decoder layers; "
                f"missing {', '.join(missing[:5])}"
            )
        biases[projection] = bool(present)

    declared = config.get("attention_bias")
    has_any_bias = any(biases.values())
    if declared is False and has_any_bias:
        raise SkillError(
            "config.json attention_bias=false conflicts with inspected attention bias tensors"
        )
    if declared is True and not all(biases.values()):
        raise SkillError(
            "config.json attention_bias=true conflicts with incomplete inspected attention bias tensors"
        )
    return biases


def moe_attention_biases(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    biases = dense_attention_biases(inspection, config)
    qkv_bias = config.get("qkv_bias")
    if qkv_bias is not None:
        expected = all(biases[name] for name in ("q_proj", "k_proj", "v_proj"))
        if qkv_bias != expected:
            raise SkillError("config.json qkv_bias conflicts with inspected Q/K/V bias tensors")
    return biases


def dense_target_tensors(
    config: dict[str, Any],
    attention_biases: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the deterministic parameter contract implemented by the generated graph."""
    hidden = _config_int(config, "hidden_size")
    heads = _config_int(config, "num_attention_heads")
    kv_heads = _config_int(config, "num_key_value_heads", default=heads)
    head_dim = _config_int(config, "head_dim", default=hidden // heads)
    intermediate = _config_int(config, "intermediate_size")
    layers = _config_int(config, "num_hidden_layers")
    vocab = _config_int(config, "vocab_size")
    if attention_biases is None:
        attention_biases = {
            projection: bool(config.get("attention_bias", False))
            for projection in ATTENTION_PROJECTIONS
        }
    tensors: dict[str, list[int]] = {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
    }
    if not bool(config.get("tie_word_embeddings", False)):
        tensors["lm_head.weight"] = [vocab, hidden]
    for index in range(layers):
        prefix = f"model.layers.{index}"
        tensors.update({
            f"{prefix}.input_layernorm.weight": [hidden],
            f"{prefix}.post_attention_layernorm.weight": [hidden],
            f"{prefix}.self_attn.q_proj.weight": [heads * head_dim, hidden],
            f"{prefix}.self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
            f"{prefix}.self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
            f"{prefix}.self_attn.o_proj.weight": [hidden, heads * head_dim],
            f"{prefix}.mlp.gate_proj.weight": [intermediate, hidden],
            f"{prefix}.mlp.up_proj.weight": [intermediate, hidden],
            f"{prefix}.mlp.down_proj.weight": [hidden, intermediate],
        })
        attention_bias_shapes = {
            "q_proj": [heads * head_dim],
            "k_proj": [kv_heads * head_dim],
            "v_proj": [kv_heads * head_dim],
            "o_proj": [hidden],
        }
        for projection in ATTENTION_PROJECTIONS:
            if attention_biases.get(projection):
                tensors[f"{prefix}.self_attn.{projection}.bias"] = attention_bias_shapes[projection]
        if bool(config.get("mlp_bias", False)):
            tensors.update({
                f"{prefix}.mlp.gate_proj.bias": [intermediate],
                f"{prefix}.mlp.up_proj.bias": [intermediate],
                f"{prefix}.mlp.down_proj.bias": [hidden],
            })
    return [{"key": key, "shape": tensors[key]} for key in sorted(tensors)]


def moe_target_tensors(
    config: dict[str, Any],
    attention_biases: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the sparse-MoE parameter contract with explicit HF expert names."""
    hidden = _config_int(config, "hidden_size")
    heads = _config_int(config, "num_attention_heads")
    kv_heads = _config_int(config, "num_key_value_heads", default=heads)
    head_dim = _moe_head_dim(config, hidden, heads)
    layers = _config_int(config, "num_hidden_layers")
    vocab = _config_int(config, "vocab_size")
    experts = _aliased_positive_int(config, "num_experts", "num_local_experts")
    moe_intermediate = _config_int(
        config,
        "moe_intermediate_size",
        default=_config_int(config, "intermediate_size"),
    )
    if attention_biases is None:
        attention_biases = {
            projection: bool(config.get("attention_bias", False))
            for projection in ATTENTION_PROJECTIONS
        }
    tensors: dict[str, list[int]] = {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
    }
    if not bool(config.get("tie_word_embeddings", False)):
        tensors["lm_head.weight"] = [vocab, hidden]
    attention_bias_shapes = {
        "q_proj": [heads * head_dim],
        "k_proj": [kv_heads * head_dim],
        "v_proj": [kv_heads * head_dim],
        "o_proj": [hidden],
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        tensors.update({
            f"{prefix}.input_layernorm.weight": [hidden],
            f"{prefix}.post_attention_layernorm.weight": [hidden],
            f"{prefix}.self_attn.q_proj.weight": [heads * head_dim, hidden],
            f"{prefix}.self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
            f"{prefix}.self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
            f"{prefix}.self_attn.o_proj.weight": [hidden, heads * head_dim],
            f"{prefix}.mlp.gate.weight": [experts, hidden],
        })
        for projection in ATTENTION_PROJECTIONS:
            if attention_biases.get(projection):
                tensors[f"{prefix}.self_attn.{projection}.bias"] = attention_bias_shapes[projection]
        for expert in range(experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            tensors.update({
                f"{expert_prefix}.gate_proj.weight": [moe_intermediate, hidden],
                f"{expert_prefix}.up_proj.weight": [moe_intermediate, hidden],
                f"{expert_prefix}.down_proj.weight": [hidden, moe_intermediate],
            })
            if bool(config.get("mlp_bias", False)):
                tensors.update({
                    f"{expert_prefix}.gate_proj.bias": [moe_intermediate],
                    f"{expert_prefix}.up_proj.bias": [moe_intermediate],
                    f"{expert_prefix}.down_proj.bias": [hidden],
                })
    return [{"key": key, "shape": tensors[key]} for key in sorted(tensors)]


def moe_source_tensors(
    config: dict[str, Any],
    attention_biases: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the exact inspected source topology accepted for one router alias."""
    profile = moe_router_profile(config)
    topology = str(profile["topology"])
    target = {
        item["key"]: list(item["shape"])
        for item in moe_target_tensors(config, attention_biases)
    }
    if topology == "qwen2-per-expert":
        return [{"key": key, "shape": target[key]} for key in sorted(target)]

    layers = _config_int(config, "num_hidden_layers")
    experts = _aliased_positive_int(config, "num_experts", "num_local_experts")
    hidden = _config_int(config, "hidden_size")
    intermediate = _config_int(
        config,
        "moe_intermediate_size",
        default=_config_int(config, "intermediate_size"),
    )
    source = {
        key: shape
        for key, shape in target.items()
        if ".mlp." not in key
    }
    for layer in range(layers):
        target_prefix = f"model.layers.{layer}.mlp"
        if topology == "mixtral-block-sparse":
            source[f"model.layers.{layer}.block_sparse_moe.gate.weight"] = [experts, hidden]
            for expert in range(experts):
                prefix = f"model.layers.{layer}.block_sparse_moe.experts.{expert}"
                source[f"{prefix}.w1.weight"] = [intermediate, hidden]
                source[f"{prefix}.w2.weight"] = [hidden, intermediate]
                source[f"{prefix}.w3.weight"] = [intermediate, hidden]
        elif topology == "granite-packed-experts":
            prefix = f"model.layers.{layer}.block_sparse_moe"
            source[f"{prefix}.router.layer.weight"] = [experts, hidden]
            source[f"{prefix}.input_linear.weight"] = [experts, 2 * intermediate, hidden]
            source[f"{prefix}.output_linear.weight"] = [experts, hidden, intermediate]
        else:
            raise SkillError(f"internal unsupported MoE topology profile {topology!r}")
        for key in tuple(source):
            if key.startswith(target_prefix):
                raise SkillError("internal MoE source topology retained target expert names")
    return [{"key": key, "shape": source[key]} for key in sorted(source)]


def validate_moe_tensor_topology(
    inspection: dict[str, Any],
    config: dict[str, Any],
    attention_biases: dict[str, bool],
) -> None:
    """Fail closed unless every inspected tensor matches the supported source profile."""
    expected = {
        item["key"]: item["shape"]
        for item in moe_source_tensors(config, attention_biases)
    }
    tensors = inspection.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise SkillError("inspected MoE tensor topology is missing; no code was generated")
    actual: dict[str, list[int]] = {}
    malformed: list[str] = []
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            malformed.append(f"tensor[{index}] is not an object")
            continue
        key = tensor.get("key")
        shape = tensor.get("shape")
        if not isinstance(key, str) or not key:
            malformed.append(f"tensor[{index}] has no valid key")
            continue
        if key in actual:
            malformed.append(f"duplicate tensor key {key}")
            continue
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            malformed.append(f"tensor {key} has an invalid shape")
            continue
        actual[key] = shape
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        key for key in set(actual) & set(expected) if actual[key] != expected[key]
    )
    if malformed or missing or unexpected or mismatched:
        details = list(malformed)
        if missing:
            details.append("missing tensors: " + ", ".join(missing[:8]))
        if unexpected:
            details.append("unsupported tensors: " + ", ".join(unexpected[:8]))
        if mismatched:
            details.extend(
                f"shape mismatch for {key}: inspected {actual[key]}, expected {expected[key]}"
                for key in mismatched[:8]
            )
        raise SkillError(
            "Unsupported inspected MoE tensor topology; no code was generated.\n- "
            + "\n- ".join(details)
        )


def _scaffold_manifest(
    files: dict[str, str],
    inspection_digest: str,
    tensors: list[dict[str, Any]],
) -> str:
    execution_files = ("capture.py", "config.json", "config.py", "model.py")
    records = []
    for name in execution_files:
        raw = files[name].encode("utf-8")
        records.append({
            "path": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    payload = {
        "schema_version": 1,
        "generator": {"name": "scaffold_port.py", "version": GENERATOR_VERSION},
        "inspection_sha256": inspection_digest,
        "config_sha256": hashlib.sha256(files["config.json"].encode("utf-8")).hexdigest(),
        "files": records,
        "tensors": tensors,
    }
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def generate_dense_decoder(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    validate_dense_config(config)
    attention_biases = dense_attention_biases(inspection, config)
    digest = trusted_inspection_sha256(inspection)
    header = _header(digest)
    model_source = MODEL_TEMPLATE.replace("__HEADER__", header)
    for projection in ATTENTION_PROJECTIONS:
        placeholder = f"__{projection.upper()}_BIAS__"
        model_source = model_source.replace(
            placeholder,
            "True" if attention_biases[projection] else "False",
        )
    bias_summary = ", ".join(
        f"`{projection}`={'present' if attention_biases[projection] else 'absent'}"
        for projection in ATTENTION_PROJECTIONS
    )
    files = {
        "config.json": json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "config.py": CONFIG_TEMPLATE.replace("__HEADER__", header),
        "model.py": model_source,
        "generate.py": GENERATE_TEMPLATE.replace("__HEADER__", header),
        "capture.py": CAPTURE_TEMPLATE.replace("__HEADER__", header),
        "README.md": (
            README_TEMPLATE
            .replace("__VERSION__", GENERATOR_VERSION)
            .replace("__DIGEST__", digest)
            .replace("__ATTENTION_BIAS_SUMMARY__", bias_summary)
        ),
    }
    files[SCAFFOLD_MANIFEST] = _scaffold_manifest(
        files,
        digest,
        dense_target_tensors(config, attention_biases),
    )
    return files


def generate_moe_decoder(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    validate_moe_config(config)
    attention_biases = moe_attention_biases(inspection, config)
    validate_moe_tensor_topology(inspection, config, attention_biases)
    digest = trusted_inspection_sha256(inspection)
    header = _header(digest)
    model_source = moe_model_template().replace("__HEADER__", header)
    for projection in ATTENTION_PROJECTIONS:
        placeholder = f"__{projection.upper()}_BIAS__"
        model_source = model_source.replace(
            placeholder,
            "True" if attention_biases[projection] else "False",
        )
    bias_summary = ", ".join(
        f"`{projection}`={'present' if attention_biases[projection] else 'absent'}"
        for projection in ATTENTION_PROJECTIONS
    )
    files = {
        "config.json": json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "config.py": moe_config_template().replace("__HEADER__", header),
        "model.py": model_source,
        "generate.py": GENERATE_TEMPLATE.replace("__HEADER__", header),
        "capture.py": CAPTURE_TEMPLATE.replace("__HEADER__", header),
        "README.md": (
            MOE_README_TEMPLATE
            .replace("__VERSION__", GENERATOR_VERSION)
            .replace("__DIGEST__", digest)
            .replace("__ATTENTION_BIAS_SUMMARY__", bias_summary)
        ),
    }
    files[SCAFFOLD_MANIFEST] = _scaffold_manifest(
        files,
        digest,
        moe_target_tensors(config, attention_biases),
    )
    return files


ENCODER_CONFIG_TEMPLATE = r'''__HEADER__
"""Validated config.json parser for the generated BERT encoder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _positive_int(data, key):
    value = data.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int
    layer_norm_eps: float
    hidden_act: str
    vocab_size: int
    pad_token_id: int
    position_embedding_type: str

    @classmethod
    def from_dict(cls, data):
        hidden = _positive_int(data, "hidden_size")
        heads = _positive_int(data, "num_attention_heads")
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        position_type = data.get("position_embedding_type", "absolute")
        if position_type != "absolute":
            raise ValueError("position_embedding_type must be 'absolute'")
        activation = data.get("hidden_act", "gelu")
        if activation != "gelu":
            raise ValueError("hidden_act must be 'gelu'")
        epsilon = data.get("layer_norm_eps", 1e-12)
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("layer_norm_eps must be a positive number")
        pad = data.get("pad_token_id", 0)
        if type(pad) is not int or pad < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        return cls(
            hidden_size=hidden,
            num_hidden_layers=_positive_int(data, "num_hidden_layers"),
            num_attention_heads=heads,
            intermediate_size=_positive_int(data, "intermediate_size"),
            max_position_embeddings=_positive_int(data, "max_position_embeddings"),
            type_vocab_size=_positive_int(data, "type_vocab_size"),
            layer_norm_eps=float(epsilon),
            hidden_act=activation,
            vocab_size=_positive_int(data, "vocab_size"),
            pad_token_id=pad,
            position_embedding_type=position_type,
        )

    @classmethod
    def from_file(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(data)
'''


ENCODER_MODEL_TEMPLATE = r'''__HEADER__
"""Minimal eager MLX BERT encoder with bidirectional padded attention."""
from __future__ import annotations

from pathlib import Path

from config import ModelConfig


def _require_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError("generated model requires MLX: python3 -m pip install mlx") from exc
    return mx, nn


def build_model(config: ModelConfig):
    mx, nn = _require_mlx()
    head_dim = config.hidden_size // config.num_attention_heads

    class BertEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def __call__(self, input_ids, token_type_ids=None):
            batch, length = input_ids.shape
            if length > config.max_position_embeddings:
                raise ValueError("input length exceeds max_position_embeddings")
            if token_type_ids is None:
                token_type_ids = mx.zeros((batch, length), dtype=mx.int32)
            positions = mx.broadcast_to(mx.arange(length, dtype=mx.int32)[None, :], (batch, length))
            hidden = (
                self.word_embeddings(input_ids)
                + self.position_embeddings(positions)
                + self.token_type_embeddings(token_type_ids)
            )
            return self.LayerNorm(hidden)

    class BertSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        def __call__(self, hidden, attention_mask=None):
            batch, length, _ = hidden.shape
            def split(value):
                return value.reshape(
                    batch, length, config.num_attention_heads, head_dim
                ).transpose(0, 2, 1, 3)
            query = split(self.query(hidden))
            key = split(self.key(hidden))
            value = split(self.value(hidden))
            scores = (query @ key.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
            if attention_mask is not None:
                if attention_mask.shape != (batch, length):
                    raise ValueError("attention_mask must match input_ids shape")
                scores = mx.where(
                    attention_mask[:, None, None, :].astype(mx.bool_),
                    scores,
                    mx.array(-3.4028235e38, dtype=scores.dtype),
                )
            probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(value.dtype)
            return (probabilities @ value).transpose(0, 2, 1, 3).reshape(
                batch, length, config.hidden_size
            )

    class BertSelfOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def __call__(self, hidden, residual):
            return self.LayerNorm(self.dense(hidden) + residual)

    class BertAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = BertSelfAttention()
            self.output = BertSelfOutput()

        def __call__(self, hidden, attention_mask=None):
            return self.output(self.self(hidden, attention_mask=attention_mask), hidden)

    class BertIntermediate(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)

        def __call__(self, hidden):
            return nn.gelu(self.dense(hidden))

    class BertOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def __call__(self, hidden, residual):
            return self.LayerNorm(self.dense(hidden) + residual)

    class BertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = BertAttention()
            self.intermediate = BertIntermediate()
            self.output = BertOutput()

        def __call__(self, hidden, attention_mask=None):
            attended = self.attention(hidden, attention_mask=attention_mask)
            return self.output(self.intermediate(attended), attended)

    class BertEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = [BertLayer() for _ in range(config.num_hidden_layers)]

        def __call__(self, hidden, attention_mask=None, capture=False):
            captures = {}
            for index, layer in enumerate(self.layer):
                hidden = layer(hidden, attention_mask=attention_mask)
                if capture:
                    captures[f"layer.{index}.hidden"] = hidden
            return hidden, captures

    class BertPooler(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        def __call__(self, hidden):
            return mx.tanh(self.dense(hidden[:, 0]))

    class BertModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = BertEmbeddings()
            self.encoder = BertEncoder()
            if __HAS_POOLER__:
                self.pooler = BertPooler()

        def __call__(self, input_ids, *, attention_mask=None, token_type_ids=None, capture=False):
            hidden = self.embeddings(input_ids, token_type_ids=token_type_ids)
            captures = {"embed": hidden} if capture else {}
            hidden, layer_captures = self.encoder(
                hidden, attention_mask=attention_mask, capture=capture
            )
            captures.update(layer_captures)
            pooled = self.pooler(hidden) if __HAS_POOLER__ else hidden[:, 0]
            if capture:
                captures["final_hidden"] = hidden
                captures["pooled"] = pooled
                return hidden, pooled, captures
            return hidden, pooled

    return BertModel()


def load_model(config_path: str | Path, weights_path: str | Path):
    mx, _ = _require_mlx()
    model = build_model(ModelConfig.from_file(config_path))
    weights = mx.load(str(weights_path))
    if not isinstance(weights, dict):
        raise ValueError("converted weights must load as a name-to-array mapping")
    model.load_weights(list(weights.items()), strict=True)
    getattr(mx, "eval")(model.parameters())
    return model
'''


ENCODER_CAPTURE_TEMPLATE = r'''__HEADER__
"""Direct token-ID capture CLI for the generated BERT encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import _require_mlx, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture generated BERT encoder tensors")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--token-ids", nargs="+", type=int, required=True)
    parser.add_argument("--attention-mask", nargs="+", type=int)
    args = parser.parse_args()
    mx, _ = _require_mlx()
    input_ids = mx.array([args.token_ids], dtype=mx.int32)
    mask_values = args.attention_mask or [1] * len(args.token_ids)
    if len(mask_values) != len(args.token_ids) or any(value not in (0, 1) for value in mask_values):
        raise ValueError("--attention-mask must contain one 0/1 value per token ID")
    attention_mask = mx.array([mask_values], dtype=mx.int32)
    model = load_model(args.config, args.weights)
    _, _, captures = model(input_ids, attention_mask=attention_mask, capture=True)
    getattr(mx, "eval")(*captures.values())
    print(json.dumps({name: list(value.shape) for name, value in sorted(captures.items())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


ENCODER_README_TEMPLATE = r'''# Generated MLX BERT encoder

Generated by `scaffold_port.py` version __VERSION__ from trusted inspection
`sha256:__DIGEST__` for family `encoder-transformer`.

This eager standalone MLX package implements the inspected BERT text-encoder
topology: word, learned absolute position, and token-type embeddings followed by
embedding LayerNorm; __LAYERS__ bidirectional post-LayerNorm attention/FFN blocks;
and __POOLER__. It has no causal mask, KV cache, or autoregressive interface.

Parameter names intentionally mirror Hugging Face `BertModel` state-dict keys.
The scaffold manifest is the target contract for a schema-2 `WEIGHT_MAP.json`.
Validate masked and unmasked source/target captures through `run_parity.py
--mode encoder`; support for relative/rotary positions and modality-specific
ViT/CLIP frontends is deliberately excluded.
'''


def _encoder_scaffold_manifest(
    files: dict[str, str],
    config: dict[str, Any],
    inspection_digest: str,
    *,
    pooler: bool,
) -> str:
    records = []
    for name in ("capture.py", "config.json", "config.py", "model.py"):
        raw = files[name].encode("utf-8")
        records.append({
            "path": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    target = encoder_target_tensor_dict(config, pooler=pooler)
    payload = {
        "schema_version": 1,
        "generator": {"name": "scaffold_port.py", "version": GENERATOR_VERSION},
        "inspection_sha256": inspection_digest,
        "config_sha256": hashlib.sha256(files["config.json"].encode("utf-8")).hexdigest(),
        "files": records,
        "tensors": [{"key": key, "shape": target[key]} for key in sorted(target)],
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def generate_encoder_transformer(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    validate_encoder_config(config)
    pooler = validate_encoder_topology(inspection, config)
    digest = trusted_inspection_sha256(inspection)
    header = _header(digest)
    files = {
        "config.json": json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "config.py": ENCODER_CONFIG_TEMPLATE.replace("__HEADER__", header),
        "model.py": (
            ENCODER_MODEL_TEMPLATE.replace("__HEADER__", header)
            .replace("__HAS_POOLER__", "True" if pooler else "False")
        ),
        "capture.py": ENCODER_CAPTURE_TEMPLATE.replace("__HEADER__", header),
        "README.md": (
            ENCODER_README_TEMPLATE.replace("__VERSION__", GENERATOR_VERSION)
            .replace("__DIGEST__", digest)
            .replace("__LAYERS__", str(_config_int(config, "num_hidden_layers")))
            .replace("__POOLER__", "a dense+tanh CLS pooler" if pooler else "CLS pooling")
        ),
    }
    files[SCAFFOLD_MANIFEST] = _encoder_scaffold_manifest(
        files, config, digest, pooler=pooler
    )
    return files


def ssm_target_tensors(config: dict[str, Any]) -> list[dict[str, Any]]:
    d_model = _config_int(config, "d_model")
    d_state = _config_int(config, "d_state")
    d_conv = _config_int(config, "d_conv")
    d_inner = d_model * _config_int(config, "expand")
    layers = _config_int(config, "n_layer")
    vocab = _config_int(config, "vocab_size")
    tensors: dict[str, list[int]] = {
        "model.embed_tokens.weight": [vocab, d_model],
        "model.norm.weight": [d_model],
    }
    for index in range(layers):
        prefix = f"model.layers.{index}"
        tensors.update({
            f"{prefix}.norm.weight": [d_model],
            f"{prefix}.mixer.in_proj.weight": [2 * d_inner, d_model],
            f"{prefix}.mixer.conv1d.weight": [d_inner, d_conv],
            f"{prefix}.mixer.x_proj.weight": [d_inner + 2 * d_state, d_inner],
            f"{prefix}.mixer.dt_bias": [d_inner],
            f"{prefix}.mixer.A_log": [d_inner, d_state],
            f"{prefix}.mixer.D": [d_inner],
            f"{prefix}.mixer.out_proj.weight": [d_model, d_inner],
        })
        if bool(config.get("conv_bias", True)):
            tensors[f"{prefix}.mixer.conv1d.bias"] = [d_inner]
    if not bool(config.get("tie_word_embeddings", True)):
        tensors["lm_head.weight"] = [vocab, d_model]
    return [{"key": key, "shape": tensors[key]} for key in sorted(tensors)]


def generate_selective_ssm(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    validate_ssm_config(config)
    validate_ssm_inspection(inspection)
    digest = trusted_inspection_sha256(inspection)
    header = _header(digest)
    files = {
        "config.json": json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "config.py": SSM_CONFIG_TEMPLATE.replace("__HEADER__", header),
        "model.py": SSM_MODEL_TEMPLATE.replace("__HEADER__", header),
        "generate.py": GENERATE_TEMPLATE.replace("__HEADER__", header),
        "capture.py": CAPTURE_TEMPLATE.replace("__HEADER__", header),
        "README.md": (
            SSM_README_TEMPLATE
            .replace("__VERSION__", GENERATOR_VERSION)
            .replace("__DIGEST__", digest)
        ),
    }
    files[SCAFFOLD_MANIFEST] = _scaffold_manifest(
        files,
        digest,
        ssm_target_tensors(config),
    )
    return files


Generator = Callable[[dict[str, Any], dict[str, Any]], dict[str, str]]
FAMILY_GENERATORS: dict[str, Generator] = {
    DENSE_FAMILY: generate_dense_decoder,
    MOE_FAMILY: generate_moe_decoder,
    SSM_FAMILY: generate_selective_ssm,
    ENCODER_FAMILY: generate_encoder_transformer,
}


def _candidate_runbook(inspection: dict[str, Any], family: str | None) -> str:
    if family:
        for candidate in inspection.get("architecture_candidates", []):
            if isinstance(candidate, dict) and candidate.get("family") == family:
                runbook = candidate.get("runbook")
                if isinstance(runbook, str) and runbook:
                    return runbook
    runbook = inspection.get("recommended_runbook")
    return str(runbook) if isinstance(runbook, str) and runbook else "manual selection required"


def _blocked_error(inspection: dict[str, Any]) -> SkillError:
    routing = inspection.get("routing_decision", {})
    family = routing.get("winner_family") if isinstance(routing, dict) else None
    blockers = [str(item) for item in inspection.get("recommendation_blockers", [])]
    return SkillError(
        "Inspection is blocked; no code was generated.\n"
        "Recommendation blockers:\n- " + "\n- ".join(blockers) + "\n"
        f"Runbook: {_candidate_runbook(inspection, str(family) if family else None)}\n"
        "Manual work:\n"
        "- resolve every inspection blocker\n"
        "- rerun inspect_model.py against the artifact\n"
        "- regenerate the scaffold only after the trusted route is unblocked"
    )


def _unsupported_route_error(inspection: dict[str, Any], family: str) -> SkillError:
    return SkillError(
        "Unsupported architecture route; no code was generated.\n"
        f"Family: {family}\n"
        f"Runbook: {_candidate_runbook(inspection, family)}\n"
        "Manual work:\n"
        f"- implement the model graph for {family}\n"
        "- define and validate a stable parameter-name mapping\n"
        "- capture target tensors with the source-oracle key scheme\n"
        "- validate end-to-end parity and state/cache behavior"
    )


def _write_new_directory(output: Path, files: dict[str, str]) -> None:
    if output.exists() or output.is_symlink():
        raise SkillError(f"output directory already exists; refusing to overwrite: {output}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        for name in sorted(files):
            (temporary / name).write_text(files[name], encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inspection = validate_trusted_inspection(load_structured(args.inspection))
        blockers = inspection["recommendation_blockers"]
        if blockers:
            raise _blocked_error(inspection)
        families = resolve_route_families(inspection)
        if not families:
            raise SkillError(
                "No detected family; no code was generated. Complete manual architecture review first"
            )
        verify_inspection_against_artifact(inspection, args.artifact_root)
        unsupported = [family for family in families if family not in FAMILY_GENERATORS]
        if unsupported:
            raise _unsupported_route_error(inspection, unsupported[0])
        if len(families) != 1:
            raise SkillError(
                "Hybrid architecture routes require a composition-specific generator; no code was generated"
            )
        artifact_root = Path(args.artifact_root).expanduser().resolve()
        config = load_structured(artifact_root / "config.json")
        files = FAMILY_GENERATORS[families[0]](inspection, config)
        _write_new_directory(Path(args.output).expanduser(), files)
        print(json.dumps({
            "family": families[0],
            "files": sorted(files),
            "inspection_sha256": trusted_inspection_sha256(inspection),
            "output": str(Path(args.output).expanduser()),
        }, sort_keys=True))
        return 0
    except (SkillError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
